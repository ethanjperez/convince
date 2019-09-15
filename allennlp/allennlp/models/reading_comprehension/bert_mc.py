import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.functional import nll_loss, softmax, log_softmax

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from pytorch_pretrained_bert.modeling import BertEncoder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FiLM(torch.nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(-2)
        betas = betas.unsqueeze(-2)
        return (gammas * x) + betas


class BertMC(Model):
    """
    This class implements BERT for Multiple-choice QA

    The basic layout is:
    1) Encode P, Q, A with BERT
    2) Use bilinear attentions (PxQ and PxA_i) to get P, Q, A summaries
    3) Additional, global non-linear operations on BERT and summary P, Q, A features
    4) Softmax over the predicted logit for each A_i

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``passage`` ``TextFields`` we get as input to the model.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False,
                 qa_loss_weight: float = 0.,
                 influence_reward: bool = False,
                 theory_of_mind: bool = False) -> None:
        super(BertMC, self).__init__(vocab, regularizer)

        self.judge = judge
        self.is_judge = self.judge is None
        self.reward_method = None if self.is_judge else reward_method
        self.update_judge = update_judge and (self.judge is not None)
        self._detach_value_head = detach_value_head
        self._qa_loss_weight = qa_loss_weight
        self.influence_reward = influence_reward
        self.theory_of_mind = theory_of_mind
        self._text_field_embedder = text_field_embedder
        self._hidden_dim = text_field_embedder.get_output_dim()
        self.answer_type = 'mc'
        self.output_type = 'mc'
        self._config = self._text_field_embedder.token_embedder_tokens._modules['bert_model'].config

        if not self.is_judge:
            self._sent_chosen_embeddings = torch.nn.Embedding(2, self._config.hidden_size)
            self._sent_chosen_embeddings.weight.data *= 0  # Init to zero to minimally affect BERT at start
            self._policy_head = TimeDistributed(torch.nn.Linear(self._hidden_dim, 1))  # Can make MLP
            self._value_head = TimeDistributed(torch.nn.Linear(self._hidden_dim, 1))  # Can make MLP
            self._turn_film_gen = torch.nn.Linear(1, 2 * self._hidden_dim)
            self._film = FiLM()
            if self.theory_of_mind:
                final_blocks_config = deepcopy(self._config)
                final_blocks_config.num_hidden_layers = 1
                self.final_blocks_input_proj = TimeDistributed(torch.nn.Linear(self._hidden_dim * 2, self._hidden_dim))
                self.final_blocks = BertEncoder(final_blocks_config)

        # NOTE: Rename to self._accuracy (may break model loading)
        self._span_start_accuracy = CategoricalAccuracy()
        self._initializer = initializer

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                options: Dict[str, torch.LongTensor],
                answer_index: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                store_metrics: bool = True,
                valid_output_mask: torch.LongTensor = None,
                sent_targets: torch.Tensor = None,
                stance: torch.LongTensor = None,
                all_past_sent_choice_mask: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.
        answer_index : ``torch.IntTensor``, optional
            From an ``IndexField``.  This is one of the things we are trying to predict - the
            index of option that is the true answer.  If this is given, we will compute a loss
            that gets included in the output dictionary.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question ID, original passage text, and token
            offsets into the passage for each instance in the batch.  We use this for computing
            official metrics using the official SQuAD evaluation script.  The length of this list
            should be the batch size, and each dictionary should have the keys ``id``,
            ``original_passage``, and ``token_offsets``.  If you only want the best span string and
            don't care about official metrics, you can omit the ``id`` key.
        store_metrics : bool
            If true, stores metrics (if applicable) within model metric tracker.
            If false, returns resulting metrics immediately, without updating the model metric tracker.

        Returns
        -------
        An output dictionary consisting of:
        option_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, passage_length)`` representing unnormalized log
            probabilities of the span start position.
        option_probs : torch.FloatTensor
            The result of ``softmax(span_start_logits)``.
        best_answer_index : torch.IntTensor
            The result of a constrained inference over ``span_start_logits`` and
            ``span_end_logits`` to find the most probable span.  Shape is ``(batch_size, 2)``
            and each offset is a token index.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        # Precomputation
        options_to_support = None
        if not self.is_judge:
            options_to_support = torch.zeros_like(options['tokens'][:, :, 0]).float()
            if stance.dim() == options_to_support.dim():
                options_to_support = stance.float()
            else:
                for i in range(options['tokens'].size(0)):
                    if stance[i] == 1:
                        options_to_support[i, answer_index[i]] = 1.  # Support only correct option
                    else:
                        options_to_support[i] = 1. - options_to_support[i]  # Support all options
                        options_to_support[i, answer_index[i]] = 0.         # except correct one

        try:  # NB: Clean this up for BERT_Large
            sep_token = metadata[0]['[SEP]'] if '[SEP]' in metadata[0] else self.vocab._token_to_index['bert']['[SEP]']
        except:
            sep_token = 102

        # Predict answer (judge or debate+auxiliary loss)
        option_logits, policy_logits, value = self.compute_logits_and_value(
            question, passage, options, sep_token, options_to_support, all_past_sent_choice_mask)
        if option_logits is not None:
            option_probs = softmax(option_logits, dim=1)
            best_answer_index = option_probs.max(dim=1)[1]
        else:
            option_probs = None
            best_answer_index = None

        # Predict sentence to choose (debaters only)
        if policy_logits is not None:
            # Truncate option_logits, since valid_output_mask may only be defined on the passage
            # NOTE: Assumes passage always comes first.
            policy_logits = util.replace_masked_values(policy_logits[:, :valid_output_mask.size(1)], valid_output_mask, -1e7)
            policy_probs = util.masked_softmax(policy_logits, valid_output_mask)
        else:
            policy_probs = None

        # Store results
        output_dict = {
                "option_logits": option_logits,
                "option_probs": option_probs,
                "best_answer_index": best_answer_index,
                "policy_logits": policy_logits,
                "policy_probs": policy_probs,
                "value": value if (not self.is_judge) and (not self.reward_method.startswith('sl')) else None,
                "f1": None,
                "em": (best_answer_index == answer_index.squeeze(-1)).float() if best_answer_index is not None else None,
                "prob": option_probs.gather(1, answer_index).squeeze(1) if self.is_judge else None,  # prob(true ans)
                "prob_dist": option_probs if self.is_judge else policy_probs,
                }

        # Compute the loss for training.
        if (answer_index is not None) and (option_logits is not None):  # Judge SL / Debate Auxiliary SL Loss
            qa_loss = nll_loss(log_softmax(option_logits, dim=1), answer_index.squeeze(-1))
            if (not self.is_judge) and (self._qa_loss_weight > 0):
                qa_loss = qa_loss * self._qa_loss_weight
            output_dict["loss"] = output_dict.get('loss', 0) + qa_loss
            if store_metrics:
                self._span_start_accuracy(option_logits, answer_index.squeeze(-1))
        if policy_logits is not None:  # Debate SL
            if self.reward_method == 'sl':  # sent_targets should be a vector of target indices
                output_dict["loss"] = output_dict.get('loss', 0) + nll_loss(util.masked_log_softmax(policy_logits, valid_output_mask), sent_targets.squeeze(-1))
                if store_metrics:
                    self._span_start_accuracy(policy_logits, sent_targets.squeeze(-1))
            elif self.reward_method.startswith('sl-sents'):  # sent_targets should be a matrix of target values (non-zero only in EOS indices)
                sent_targets = util.replace_masked_values(sent_targets, valid_output_mask, -1e7)
                output_dict["loss"] = output_dict.get('loss', 0) + ((policy_logits - sent_targets) ** 2).sum(dim=1)  # or: util.masked_mean(((policy_logits - sent_targets) ** 2), valid_output_mask, 1)
                if store_metrics:
                    self._span_start_accuracy(policy_logits, sent_targets.max(-1)[1])

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'start_acc': self._span_start_accuracy.get_metric(reset)}

    @staticmethod
    def pack_sequences(tokens_1: torch.LongTensor, tokens_2: torch.LongTensor, sep_token: int = None,
                       maxlen: int = 512) -> torch.LongTensor:
        """
        Packs two BERT-formatted sequences into BERT format: [CLS] seq1 tokens [SEP] seq2 tokens [SEP].
        If packed sequence exceeds BERT's max input length, then the first sequence is always truncated.
        """
        assert (tokens_1.dim() == 2) and (tokens_2.dim() == 2), 'pack_sequences only supports 2-dimensional sequences.'
        batch_size = tokens_1.size(0)
        packed_seqs = torch.zeros(batch_size, maxlen, dtype=torch.long, device=tokens_1.device)
        packed_seq_lengths = []
        for i in range(batch_size):
            truncatable_length = tokens_1[i].nonzero().size(0) - 1  # Exclude terminating [SEP]
            required_length = tokens_2[i].nonzero().size(0)  # Exclude [CLS], include separating [SEP]
            seq1_target_length = min(maxlen - required_length, truncatable_length)
            packed_seq_no_padding = torch.cat([tokens_1[i, :seq1_target_length],
                                               (torch.LongTensor() if sep_token is None else torch.LongTensor([sep_token])).to(tokens_1),
                                               tokens_2[i, 1:required_length]], dim=0)
            packed_seq_length = packed_seq_no_padding.size(0)
            packed_seqs[i, :packed_seq_length] = packed_seq_no_padding
            packed_seq_lengths.append(packed_seq_length)
        return packed_seqs[:, :max(packed_seq_lengths)]  # Truncate extra padding from filling in zero matrix

    @staticmethod
    def get_token_type_ids(tokens, sep_token):
        """
        Returns the token type ids, to be used in BERT's segment embeddings
        """
        assert (tokens.dim() in [2, 3]), 'get_token_type_ids only supports {2,3}-dimensional sequences.'
        orig_size = tokens.size()
        if tokens.dim() == 3:
            tokens = util.combine_initial_dims(tokens)
        sep_token_mask = (tokens == sep_token).long()
        if sep_token_mask.nonzero().size(0) == tokens.size(0):
            return torch.zeros_like(tokens).view(orig_size)  # Use default BERT (all 0's) if there's 1 [SEP] per sample
        return (sep_token_mask.cumsum(-1) - sep_token_mask).clamp(max=1).view(orig_size)

    @staticmethod
    def tokens_to_bert_input(tokens, sep_token):
        """
        Converts tokens into a BERT-compatible dictionary format
        """
        return {
            'tokens': tokens,
            'token-type-ids': BertMC.get_token_type_ids(tokens, sep_token),
            'mask': (tokens != 0).long(),  # How BERT also gets the mask
            'tokens-offsets': None,
            # 'other-embeddings': None,
        }

    def compute_logits_and_value(self,  # type: ignore
                                 question: Dict[str, torch.LongTensor],
                                 passage: Dict[str, torch.LongTensor],
                                 options: Dict[str, torch.LongTensor],
                                 sep_token: int,
                                 options_to_support: torch.FloatTensor = None,
                                 all_past_sent_choice_mask: torch.LongTensor = None,
                                 ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass. Must be implemented by subclass.
        """
        raise NotImplementedError


@Model.register("bert-mc-gpt")
class BertMCGPT(BertMC):
    """
    Bert-for-Multiple-Choice, inspired by OpenAI GPT's RACE model. Used with BERT on RACE here:
    `BERT for Multiple Choice Machine Comprehension`: (https://github.com/NoviScl/BERT-RACE/blob/master/BERT_RACE.pdf)
    Applies BERT to each option to get each softmax logit: Logit_i = BERT([CLS] Passage [SEP] Question + Option_i [SEP])
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False,
                 qa_loss_weight: float = 0.,
                 influence_reward: bool = False,
                 theory_of_mind: bool = False) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         initializer=initializer,
                         regularizer=regularizer,
                         judge=judge,
                         update_judge=update_judge,
                         reward_method=reward_method,
                         detach_value_head=detach_value_head,
                         qa_loss_weight=qa_loss_weight,
                         influence_reward=influence_reward,
                         theory_of_mind=theory_of_mind)
        self._logit_predictor = torch.nn.Linear(self._hidden_dim, 1)
        self._initializer(self)

    def compute_logits_and_value(self,  # type: ignore
                                 question: Dict[str, torch.LongTensor],
                                 passage: Dict[str, torch.LongTensor],
                                 options: Dict[str, torch.LongTensor],
                                 sep_token: int,
                                 options_to_support: torch.FloatTensor = None,
                                 all_past_sent_choice_mask: torch.LongTensor = None,
                                 ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass
        """
        # BERT-formatting input
        batch_size, num_options, _ = options['tokens'].size()
        pqo_tokens_list = []
        pqo_token_maxlens = []
        for i in range(num_options):
            qo_tokens = self.pack_sequences(question['tokens'], options['tokens'][:, i])
            pqo_tokens_list.append(self.pack_sequences(passage['tokens'], qo_tokens, sep_token))
            pqo_token_maxlens.append(pqo_tokens_list[i].size(-1))
        pqo_tokens = torch.zeros(batch_size, num_options, max(pqo_token_maxlens), dtype=torch.long, device=passage['tokens'].device)
        for i in range(num_options):
            pqo_tokens[:, i, :pqo_tokens_list[i].size(-1)] = pqo_tokens_list[i]
        pqo = self.tokens_to_bert_input(pqo_tokens, sep_token)

        # Condition debater on stance. Also add in past debater choices
        if not self.is_judge:
            pqo['token-type-ids'][:, :, 0] = options_to_support  # Change segment embedding for [CLS] tokens only.
            if all_past_sent_choice_mask is not None:
                pqo_sent_chosen_mask = torch.zeros(batch_size, max(pqo_token_maxlens), dtype=torch.long, device=passage['tokens'].device)
                pqo_sent_chosen_mask[:, :all_past_sent_choice_mask.size(1)] = all_past_sent_choice_mask
                pqo_sent_chosen_mask = pqo_sent_chosen_mask.unsqueeze(1).expand(-1, num_options, -1)
                pqo['token-type-ids'] = (pqo['token-type-ids'] + pqo_sent_chosen_mask).clamp(max=1)
                # other_embeddings = self._sent_chosen_embeddings(pqo_sent_chosen_mask).unsqueeze(1).expand(-1, num_options, -1, -1)
                # pqo['other-embeddings'] = other_embeddings.view(-1, other_embeddings.size(-2), other_embeddings.size(-1))

        hidden_pqo = self._text_field_embedder(pqo)
        if self.is_judge:
            pred_hidden_a = hidden_pqo[:, :, 0]
            option_logits = self._logit_predictor(pred_hidden_a).squeeze(-1)
            # Expose detached hidden states for theory-of-mind debaters
            self.pqo = pqo
            self.hidden_pqo = hidden_pqo.detach()
            self.pred_hidden_a = pred_hidden_a.detach()
            self.option_logits = option_logits.detach()
            return option_logits, None, None
        else:
            # Predict answer (auxiliary SL loss)
            option_logits = None
            if self._qa_loss_weight > 0:
                pred_hidden_a = hidden_pqo[:, :, 0]
                option_logits = self._logit_predictor(pred_hidden_a).squeeze(-1)

            # Condition on option to support (again)
            agent_film_params = self._turn_film_gen(options_to_support.unsqueeze(-1))
            agent_gammas, agent_betas = torch.split(agent_film_params, self._hidden_dim, dim=-1)
            agent_hidden_pqo = self._film(hidden_pqo, 1. + agent_gammas, agent_betas) * pqo['mask'].float().unsqueeze(-1)

            # Process Judge hidden states
            if self.theory_of_mind:
                # Condition Judge states on Debater opinion (to highlight strong candidate sentences)
                cond_judge_hidden_pqo = self._film(self.judge.hidden_pqo, 1. + agent_gammas, agent_betas
                                                   ) * self.judge.pqo['mask'].float().unsqueeze(-1)
                # Align Judge states to Debater's full passage states
                shifted_judge_hidden_pqo = torch.zeros_like(agent_hidden_pqo)
                seq_lengths = util.get_lengths_from_binary_sequence_mask(pqo['mask'])
                judge_seq_lengths = util.get_lengths_from_binary_sequence_mask(self.judge.pqo['mask'])
                for i in range(batch_size):
                    for j in range(num_options):
                        shifted_judge_hidden_pqo[i, j, seq_lengths[i, j] - judge_seq_lengths[i, j]: seq_lengths[i, j]] = \
                            cond_judge_hidden_pqo[i, j, :judge_seq_lengths[i, j]]
                agent_hidden_pqo = self.final_blocks_input_proj(torch.cat([agent_hidden_pqo, shifted_judge_hidden_pqo], dim=-1))
                # Joint processing with transformer block
                extended_attention_mask = util.combine_initial_dims(pqo['mask']).unsqueeze(1).unsqueeze(2)
                extended_attention_mask = extended_attention_mask.to(dtype=next(self.final_blocks.parameters()).dtype)
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                agent_hidden_pqo = self.final_blocks(agent_hidden_pqo.view(batch_size * num_options, -1, self._hidden_dim),
                                                     extended_attention_mask, output_all_encoded_layers=False)[-1]
                # Reshape and remask
                agent_hidden_pqo = agent_hidden_pqo.view(batch_size, num_options, -1, self._hidden_dim) * pqo['mask'].float().unsqueeze(-1)

            # Predict distribution over sentence actions
            tokenwise_values = self._value_head(agent_hidden_pqo.detach() if self._detach_value_head else agent_hidden_pqo).squeeze(-1)
            value, value_option = util.replace_masked_values(tokenwise_values, pqo['mask'], -1e7).max(-1)[0].max(-1)
            policy_logits = self._policy_head(agent_hidden_pqo).squeeze(-1).sum(1)
            return option_logits, policy_logits, value


@Model.register("bert-mc-a")
class BertMCA(BertMC):
    """
    BERT Baseline which uses only answer options to make a prediction.
    Applies BERT to each option alone (without context) to get each softmax logit: Logit_i = BERT([CLS] Option_i [SEP])
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 judge: Model = None,
                 update_judge: bool = False,
                 reward_method: str = None,
                 detach_value_head: bool = False,
                 qa_loss_weight: float = 0.,
                 influence_reward: bool = False,
                 theory_of_mind: bool = False) -> None:
        super().__init__(vocab=vocab,
                         text_field_embedder=text_field_embedder,
                         initializer=initializer,
                         regularizer=regularizer,
                         judge=judge,
                         update_judge=update_judge,
                         reward_method=reward_method,
                         detach_value_head=detach_value_head,
                         qa_loss_weight=qa_loss_weight,
                         influence_reward=influence_reward,
                         theory_of_mind=theory_of_mind)
        self._logit_predictor = torch.nn.Linear(self._hidden_dim, 1)
        self._initializer(self)

    def compute_logits_and_value(self,  # type: ignore
                                 question: Dict[str, torch.LongTensor],
                                 passage: Dict[str, torch.LongTensor],
                                 options: Dict[str, torch.LongTensor],
                                 sep_token: int,
                                 options_to_support: torch.FloatTensor = None,
                                 all_past_sent_choice_mask: torch.LongTensor = None,
                                 ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Architecture-specific forward pass
        """
        if not self.is_judge:
            raise NotImplementedError

        options['tokens-offsets'] = None  # To get full BERT output (per wordpiece not word)
        options['token-type-ids'] = None  # No segment embeddings necessary here
        hidden_options = self._text_field_embedder(options)
        encoded_hidden_options = hidden_options[:, :, 0]
        option_logits = self._logit_predictor(encoded_hidden_options).squeeze(-1)
        return option_logits, None, None
