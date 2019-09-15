import copy
import logging
import math
import os
import pickle
import time
import re
import datetime
import traceback
import warnings
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, NamedTuple

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import (dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names, lazy_groups_of)
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metrics import Average
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from allennlp.training import util as training_util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@TrainerBase.register("default")
class Trainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 debate_mode: List[str],
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None,
                 eval_mode: bool = False,
                 breakpoint_level: int = 0,
                 search_outputs_path: str = None,
                 accumulation_steps: int = 1,
                 allocation_dict: Dict[str, int] = None,
                 choice_mode: str = None,
                 num_pred_rounds: int = -1,
                 x_order_prob: float = 0.,
                 require_action: bool = False,
                 single_shot: bool = False) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a ``DataIterator``, and uses the supplied ``Optimizer`` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataset and enable early stopping. There are many other bells and whistles as well.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``int``, optional (default = -1)
            An integer specifying the CUDA device to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``PytorchLRScheduler``, optional, (default = None)
            A Pytorch learning rate scheduler. The learning rate will be decayed with respect to
            this schedule at the end of each epoch. If you use
            :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`, this will use the ``validation_metric``
            provided to determine if learning has plateaued.  To support updating the learning
            rate on every batch, this can optionally implement ``step_batch(batch_num_total)`` which
            updates the learning rate given the batch number.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        log_batch_size_period : ``int``, optional, (default = ``None``)
            If defined, how often to log the average batch size.
        """
        super().__init__(serialization_dir, cuda_device, allocation_dict)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model
        self.iterator = iterator
        self._debate_mode = debate_mode
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.train_data = train_dataset
        self._validation_data = validation_dataset
        self._eval_mode = eval_mode
        self._breakpoint_level = breakpoint_level
        self._search_outputs_path = search_outputs_path
        self._accumulation_steps = accumulation_steps
        self.choice_mode = choice_mode
        self._num_pred_rounds = num_pred_rounds
        self._x_order_prob = x_order_prob
        self._require_action = require_action
        self._single_shot = single_shot

        # NOTE: 'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ' are special unicode chars. Copy and paste to use elsewhere (don't type directly).
        self._search_debate_methods = {'A', 'B', 'E', 'L', 'W', 'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ'}
        self._learning_debate_methods = {method.lower() for method in self._search_debate_methods}
        self._method_to_stance_idx = {'Ⅰ': 0, 'Ⅱ': 1, 'Ⅲ': 2, 'Ⅳ': 3}
        self._method_to_stance_idx.update({method.lower(): self._method_to_stance_idx[method]
                                           for method in self._search_debate_methods
                                           if method in self._method_to_stance_idx.keys()})

        self._single_shot_debater_output_dict_by_method = {}

        self._mask_token = '[MASK]'
        self._eos_tokens = {'.', '?', '!'}
        self._using_search = ((self.model.reward_method is not None) and (self.model.reward_method.startswith('sl'))
                              ) or (len(self._search_debate_methods.intersection(''.join(self._debate_mode))) > 0)

        self._search_outputs_is_complete = (self._search_outputs_path is not None)
        if self._search_outputs_path is None:
            self._search_outputs_path = os.path.join(serialization_dir, 'model_oracle_outputs.pkl')
        try:
            logger.info('Loading search_outputs from filepath: ' + self._search_outputs_path)
            with open(self._search_outputs_path, 'rb') as f:
                self._search_outputs = pickle.load(f)
            self._search_outputs_is_complete = True
            logger.info('Loaded search_outputs from filepath: ' + self._search_outputs_path)
        except:
            self._search_outputs = {}
            self._search_outputs_is_complete = False
            if self._using_search:
                logger.info('No search_outputs at filepath: ' + self._search_outputs_path)
                logger.info('Will save search_outputs to: ' + self._search_outputs_path)

        self._debate_logs = {}

        self._trainer_metrics = {}
        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning('You provided a validation dataset but patience was set to None, '
                               'meaning that early stopping is disabled')
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError('{} is an invalid value for "patience": it must be a positive integer '
                                     'or None (if you want to disable early stopping)'.format(patience))

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        self._checkpointer = Checkpointer(serialization_dir,
                                          keep_serialized_model_every_num_seconds,
                                          num_serialized_models_to_keep)

        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # ``_enable_activation_logging``.
        self._batch_num_total = 0

        self._tensorboard = TensorboardWriter(
                get_batch_num_total=lambda: self._batch_num_total,
                serialization_dir=serialization_dir,
                summary_interval=summary_interval,
                histogram_interval=histogram_interval,
                should_log_parameter_statistics=should_log_parameter_statistics,
                should_log_learning_rate=should_log_learning_rate)

        self._log_batch_size_period = log_batch_size_period

        self._last_log = 0.0  # time of last logging

        # Enable activation logging.
        if histogram_interval is not None:
            self._tensorboard.enable_activation_logging(self.model)

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    def _forward(self, batch_group: List[TensorDict], model: Model) -> Dict[str, Any]:
        """
        Does a forward pass on the appropriate model and device(s) and returns the result.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]

            # Move batch to appropriate GPU
            if model.is_judge:
                # Check allocation_dict
                if self._allocation_dict is not None:
                    batch = nn_util.move_to_device(batch, self._allocation_dict['judge'])
                else:
                    batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            else:
                # Check allocation_dict
                if self._allocation_dict is not None:
                    batch = nn_util.move_to_device(batch, self._allocation_dict['debate'])
                else:
                    batch = nn_util.move_to_device(batch, self._cuda_devices[0])

            # Run batch through model
            output_dict = model(**batch)

        return output_dict

    def _slice_or_copy_batch(self, batch: TensorDict, idxs: slice = slice(None), copy: bool = False) -> TensorDict:
        """
        Slices and copies an existing batch into a smaller batch.
        Use a single integer or a slice for idxs to get sample(s) in the batch.
        """
        sliced_batch = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                sliced_batch[k] = {}
                for inner_k, inner_v in v.items():
                    sliced_batch[k][inner_k] = inner_v[idxs].clone().detach() if copy else inner_v[idxs]
            elif v is None:
                sliced_batch[k] = v
            elif isinstance(v, torch.Tensor):
                sliced_batch[k] = v[idxs].clone().detach() if copy else v[idxs]
            elif isinstance(v, list):
                sliced_batch[k] = v[idxs].copy() if copy else v[idxs]
            elif isinstance(v, bool):
                sliced_batch[k] = v
            else:
                raise NotImplementedError('Unimplemented slice for key, value:', k, v)
        return sliced_batch

    def _create_batch_from_idx(self, batch: TensorDict, idx: int, num_repeat) -> TensorDict:
        """
        Slices and copies an existing batch into a smaller batch. Repeats the slice num_repeat times.
        Use a single integer to get sample in the batch.
        """
        sliced_batch = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                sliced_batch[k] = {}
                for inner_k, inner_v in v.items():
                    sliced_batch[k][inner_k] = inner_v[idx].repeat(num_repeat, *[1 for _ in range(inner_v[idx].dim())])
            elif v is None:
                sliced_batch[k] = v
            elif isinstance(v, torch.Tensor):
                sliced_batch[k] = v[idx].repeat(num_repeat, 1)
            elif isinstance(v, list):
                sliced_batch[k] = [copy.deepcopy(v[idx]) for _ in range(num_repeat)]
            else:
                raise NotImplementedError('Unimplemented slice for key, value:', k, v)
        return sliced_batch

    def _update_trainer_metrics(self, metric_name: str, new_value: torch.tensor) -> None:
        """
        Updates the trainer's metrics for metric_name with new_value
        """
        self._trainer_metrics[metric_name] = self._trainer_metrics.get(metric_name, Average())
        self._trainer_metrics[metric_name](new_value.detach().cpu())

    def _modify_input_passage(self, batch: TensorDict, required_text_mask: TensorDict,
                              sent_choice_input_masks: torch.Tensor) -> TensorDict:
        """
        Modifies a passage according to sentence selections made elsewhere.
        Supports e.g. masking and deleting portions of the passage.
        Used before passing the batch to the judge.
        """
        has_chars = 'token_characters' in batch['passage'].keys()
        bsz = batch['passage']['tokens'].size(0)
        if self.choice_mode in {'delete', 'concat'}:
            new_maxlen = -1
            post_delete_toks = torch.zeros_like(batch['passage']['tokens'])
            if has_chars:
                post_delete_tok_chars = torch.zeros_like(batch['passage']['token_characters'])
            for i in range(bsz):
                if self.choice_mode == 'delete':
                    sent_reveal_input_mask = 1. - sent_choice_input_masks[i]
                elif self.choice_mode == 'concat':
                    sent_reveal_input_mask = (sent_choice_input_masks[i] + required_text_mask['input'][i]).clamp(max=1)
                toks = batch['passage']['tokens'][i]
                reveal_idxs = (toks * sent_reveal_input_mask).nonzero().squeeze(-1)
                sample_len = toks[reveal_idxs].size(0)
                new_maxlen = max(new_maxlen, sample_len)
                post_delete_toks[i][:sample_len] = toks[reveal_idxs]
                if has_chars:  # NOTE: Doesn't handle BERT and token_characters simultaneously
                    tok_chars = batch['passage']['token_characters'][i]
                    post_delete_tok_chars[i][:sample_len] = tok_chars[reveal_idxs]
            # Detaching just in case to prevent gradient flow back to agents modifying passage
            batch['passage']['tokens'] = post_delete_toks[:, :new_maxlen].detach()
            batch['passage']['mask'] = (batch['passage']['tokens'] != 0).long()
            if has_chars:
                batch['passage']['token_characters'] = post_delete_tok_chars[:, :new_maxlen].detach()
        elif self.choice_mode == 'reveal':
            batch['passage']['tokens'] = batch['passage']['tokens'].masked_fill(
                sent_choice_input_masks.byte(), self.get_token_index(self._mask_token)).detach()
            if has_chars:
                batch['passage']['token_characters'] = batch['passage']['token_characters'].masked_fill(
                    sent_choice_input_masks.byte().unsqueeze(-1), 0).detach()
        else:
            raise NotImplementedError('Modifying passages via choice_mode ' + self.choice_mode + ' not supported.')
        return batch

    def _log_verdicts(self, batch: TensorDict, ver_dicts: List[TensorDict], debate_mode: List[str]):
        """
        Neatly prints and logs all judge predictions from a batch. Used for "f" full passage debate mode only.
        Simplified version of _log_debate.
        """
        bsz = batch['passage']['tokens'].size(0)
        for i in range(bsz):
            # Set up useful variables
            qid = batch['metadata'][i]['id']
            passage = ' '.join(batch['metadata'][i]['passage_tokens'])
            question = ' '.join(batch['metadata'][i]['question_tokens'])

            # Initialize and store values in debate logs
            self._debate_logs[qid] = {'passage': passage, 'question': question, 'debate_mode': debate_mode,
                                      'sentences_chosen': [], 'stances': [], 'num_sents': [],
                                      'prob_dist': [ver_dict['prob_dist'][i].tolist() for ver_dict in ver_dicts]}
            if 'option_logits' in ver_dicts[-1]:
                self._debate_logs[qid]['option_logits'] = [ver_dict['option_logits'][i].tolist()
                                                           for ver_dict in ver_dicts]

            # Pretty print initial debate info
            print('\n**ID**\n', qid)
            print('\n**Passage**\n', passage)
            print('\n**Question**\n', question)
            toks = batch['metadata'][i]['passage_tokens']
            if 'options' in batch:
                options = [' '.join(batch['metadata'][i]['options_tokens'][j])
                           for j in range(len(batch['metadata'][i]['options_tokens']))]
                self._debate_logs[qid]['options'] = options
                print('\n**Options**\n', options)
                true_answer_index = batch['answer_index'][i]
                self._debate_logs[qid]['answer_index'] = true_answer_index.item()
                print('\n**True Answer**\n', true_answer_index.item(),
                      ' '.join(batch['metadata'][i]['options_tokens'][true_answer_index]))
                final_pred_answer_index = ver_dicts[-1]['best_answer_index'][i]
                self._debate_logs[qid]['final_pred_answer_index'] = final_pred_answer_index.item()
                print('\n**Final Predicted Answer**\n', final_pred_answer_index.item(),
                      ' '.join(batch['metadata'][i]['options_tokens'][final_pred_answer_index]))
            else:
                print('\n**Answers**\n', [answer if isinstance(answer, str) else ' '.join(answer)
                                          for answer in batch['metadata'][i]['answer_texts']])

            # Pretty print each debate round
            for round_no in range(len(ver_dicts)):
                print('\n**J**:')
                if 'prob_dist' in ver_dicts[round_no]:
                    print('*PROB DIST*:', ver_dicts[round_no]['prob_dist'][i])
                for k in ['prob', 'em']:
                    if ver_dicts[round_no].get(k) is not None:
                        self._debate_logs[qid][k] = ver_dicts[round_no][k][i].item()
        return

    def _log_debate(self, batch: TensorDict, sent_idxs: TensorDict, ver_dicts: List[TensorDict],
                    sent_choice_idxs: List[torch.Tensor], debate_mode: List[str],
                    stances: List[torch.Tensor], num_sents: torch.Tensor, advantage: torch.Tensor = None,
                    ver_dict_prior: TensorDict = None) -> None:
        """
        Neatly prints and logs all debates from a batch.
        """
        bsz = batch['passage']['tokens'].size(0)
        sent_choice_output_masks = [(sent_idxs['output'] == sent_choice_idx) for sent_choice_idx in sent_choice_idxs]
        for i in range(bsz):
            # Set up useful variables
            qid = batch['metadata'][i]['id']
            passage = ' '.join(batch['metadata'][i]['passage_tokens'])
            question = ' '.join(batch['metadata'][i]['question_tokens'])

            # Initialize and store values in debate logs
            self._debate_logs[qid] = {'passage': passage, 'question': question, 'debate_mode': debate_mode,
                                      'sentences_chosen': [], 'stances': [], 'num_sents': int(num_sents[i]),
                                      'prob_dist': [ver_dict['prob_dist'][i].tolist() for ver_dict in ver_dicts]}
            if 'option_logits' in ver_dicts[-1]:
                self._debate_logs[qid]['option_logits'] = [ver_dict['option_logits'][i].tolist()
                                                           for ver_dict in ver_dicts]
            if ver_dict_prior is not None:
                self._debate_logs[qid]['prior'] = {'prob_dist': ver_dict_prior['prob_dist'][i].tolist()}
                if 'option_logits' in ver_dict_prior:
                    self._debate_logs[qid]['prior']['option_logits'] = ver_dict_prior['option_logits'][i].tolist()

            # Pretty print initial debate info
            print('\n**ID**\n', qid)
            print('\n**Passage**\n', passage)
            print('\n**Question**\n', question)
            toks = batch['metadata'][i]['passage_tokens']
            if 'options' in batch:
                options = [' '.join(batch['metadata'][i]['options_tokens'][j])
                           for j in range(len(batch['metadata'][i]['options_tokens']))]
                self._debate_logs[qid]['options'] = options
                print('\n**Options**\n', options)
                true_answer_index = batch['answer_index'][i]
                self._debate_logs[qid]['answer_index'] = true_answer_index.item()
                print('\n**True Answer**\n', true_answer_index.item(),
                      ' '.join(batch['metadata'][i]['options_tokens'][true_answer_index]))
                final_pred_answer_index = ver_dicts[-1]['best_answer_index'][i]
                self._debate_logs[qid]['final_pred_answer_index'] = final_pred_answer_index.item()
                print('\n**Final Predicted Answer**\n', final_pred_answer_index.item(),
                      ' '.join(batch['metadata'][i]['options_tokens'][final_pred_answer_index]))
            else:
                print('\n**Answers**\n', [answer if isinstance(answer, str) else ' '.join(answer)
                                          for answer in batch['metadata'][i]['answer_texts']])

            # Pretty print prior (if applicable)
            if ver_dict_prior is not None:
                if 'prob_dist' in ver_dict_prior:
                    print('\n**J**')
                    print('*PROB DIST PRIOR*:', ver_dict_prior['prob_dist'][i])
            # Pretty print each debate round
            turns_completed = 0
            for round_no, round_methods in enumerate(debate_mode):
                for round_turn_no, method in enumerate(round_methods):
                    turn_no = turns_completed + round_turn_no
                    turn_sent_idxs = {'output': sent_choice_output_masks[turn_no][i].nonzero().squeeze(-1)}
                    sent_str = None
                    if len(turn_sent_idxs['output']) > 0:
                        sent_str = ' '.join(toks[turn_sent_idxs['output'].min(): turn_sent_idxs['output'].max() + 1])
                    self._debate_logs[qid]['sentences_chosen'].append(sent_str)
                    stance_str = str(stances[method][i].nonzero().item()) \
                        if method.lower() in {'e', 'l', 'w'} else method
                    self._debate_logs[qid]['stances'].append(stance_str)
                    print('\n**' + method + '**', '*(Stance: ' + stance_str + ')*',
                          ': Sentence', int(sent_choice_idxs[turn_no][i]), '\n', sent_str)
                print('\n**J**:')
                if advantage is not None:
                    self._debate_logs[qid]['advantage'] = float(advantage[i])
                    print('*ADVANTAGE*:', round(float(advantage[i]), 4))
                if 'prob_dist' in ver_dicts[round_no]:
                    print('*PROB DIST*:', ver_dicts[round_no]['prob_dist'][i])
                for k in ['prob', 'em']:
                    if ver_dicts[round_no].get(k) is not None:
                        self._debate_logs[qid][k] = ver_dicts[round_no][k][i].item()
                turns_completed += len(round_methods)
        return

    def _print_tokens(self, tokens) -> None:
        """
        Prints BERT wordpiece tokens from token indices.
        """
        print(' '.join([self.get_index_token(tok.item()) for tok in tokens]))
        return

    @staticmethod
    def _get_output_dim(batch: TensorDict) -> int:
        """
        Returns the output (word-level, not sub-word level) dimension of a model.
        """
        output_field = 'tokens-offsets' if 'tokens-offsets' in batch['passage'] else 'tokens'
        return batch['passage'][output_field].size(1)

    @staticmethod
    def _get_last_output_token_idx(batch: TensorDict, sample_no: int) -> int:
        """
        Returns the index of the last non-padding token in a batch's passage.
        """
        output_field = 'tokens-offsets' if 'tokens-offsets' in batch['passage'] else 'tokens'
        return batch['passage'][output_field][sample_no].nonzero().max()

    @staticmethod
    def _output_to_input_idx(batch: TensorDict, sample_no: int, output_idx: int) -> int:
        """
        Converts a index on the output (always word-level) to one on the input (often sub-word level, shifted, etc.)
        """
        if 'tokens-offsets' in batch['passage']:
            return batch['passage']['tokens-offsets'][sample_no][output_idx].item()
        else:
            return output_idx

    def get_index_token(self, index):
        """
        Returns the token corresponding to a vocab index for this model.
        Within Training, this should always be used to get the index (i.e., instead of the model directly)
        """
        return self.model.vocab._index_to_token['bert'][index]

    def get_token_index(self, token: str):
        """
        Returns the index of a token, regardless of the text field embedder.
        Within Training, this should always be used to get the index (i.e., instead of the model directly)
        """
        return self.model.vocab._token_to_index['bert'][token]

    def _get_search_output_dict(self, batch: TensorDict, sent_idxs: TensorDict, judge_answer_mask: TensorDict,
                                required_text_mask: TensorDict, past_sent_choice_idxs: List[torch.Tensor],
                                num_sents: torch.Tensor, sample_no: int, debate_mode: List[str], round_no: int) -> TensorDict:
        """
        Returns the output dict from running all possible decisions on the Judge. Used to get search decisions.
        Batches together all possible next outcomes for one sample.
        """
        sample_id = batch['metadata'][sample_no]['id']
        prev_turns_str = '_'.join(debate_mode[:round_no])

        if sample_id in self._search_outputs:
            if prev_turns_str in self._search_outputs[sample_id]:  # New save format
                # NB: For multi-gpu, fix cuda device
                return nn_util.move_to_device(self._search_outputs[sample_id][prev_turns_str], self._cuda_devices[0])
        elif self._search_outputs_is_complete:
            raise Exception('Recalculating Oracle despite _search_outputs_is_complete = True ! For sample_id:',
                            sample_id)

        judge = self.model if self.model.is_judge else self.model.judge
        bsz = batch['passage']['tokens'].size(0)
        num_sent_options = num_sents[sample_no]
        search_batch = self._create_batch_from_idx(batch, sample_no, num_sent_options)
        search_batch['store_metrics'] = False  # Do not update judge metrics
        search_sent_choice_idxs = torch.arange(num_sent_options).unsqueeze(1)
        if (past_sent_choice_idxs is not None) and (len(past_sent_choice_idxs) > 0):
            past_idxs_repeat = past_sent_choice_idxs[sample_no].repeat(num_sent_options, 1)
            search_sent_choice_idxs = torch.cat([past_idxs_repeat, search_sent_choice_idxs], 1)

        # Modify Judge's input
        sample_required_text_mask = {k: v[sample_no].unsqueeze(0).expand(num_sent_options, -1)
                                     for k, v in required_text_mask.items()}
        search_sent_choice_input_masks = [sent_idxs['input'][sample_no].unsqueeze(0).expand(num_sent_options, -1) ==
                                          search_sent_choice_idxs[:, i].unsqueeze(1)
                                          for i in range(search_sent_choice_idxs.size(1))]
        search_all_sent_choice_input_mask = torch.stack(search_sent_choice_input_masks).sum(0).clamp(max=1)
        search_batch = self._modify_input_passage(
            search_batch, sample_required_text_mask, search_all_sent_choice_input_mask)

        # Get judge results (May require multiple batches)
        # Slice batch based on batch_size. Do several separate forward passes.
        num_search_batch_slices = math.ceil(num_sent_options.item() / float(bsz))
        search_output_dict = None
        for search_batch_slice_num in range(num_search_batch_slices):
            feed_slice = slice(search_batch_slice_num * bsz, (search_batch_slice_num + 1) * bsz)
            search_batch_slice = self._slice_or_copy_batch(search_batch, idxs=feed_slice)
            search_batch_slice_output_dict = self._forward([search_batch_slice], judge)
            # Add results to overall results dictionary
            if search_output_dict is None:
                search_output_dict = search_batch_slice_output_dict
            else:
                for k, v in search_batch_slice_output_dict.items():
                    if isinstance(v, torch.Tensor) and v.dim() > 0:
                        search_output_dict[k] = torch.cat(
                            [search_output_dict[k], search_batch_slice_output_dict[k]], dim=0)
                    else:
                        if k in search_output_dict.keys():
                            search_output_dict.pop(k)
        # Cache for later use and saving to file
        self._search_outputs[sample_id] = self._search_outputs.get(sample_id, {})
        self._search_outputs[sample_id][prev_turns_str] = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                                           for k, v in search_output_dict.items()}

        return search_output_dict

    @staticmethod
    def _get_sent_choice_prob_from_dist(batch: TensorDict, sent_idxs: TensorDict, sent_choice_idx: torch.Tensor,
                                        prob_dist: torch.Tensor) -> torch.Tensor:
        """
        Returns the probability of the selected sentence according to the given distribution.
        """
        sent_choice_input_mask = (sent_idxs['input'] == sent_choice_idx)
        masked_sent_choice_probs = (prob_dist.to(sent_choice_input_mask.device) * sent_choice_input_mask.float())
        assert masked_sent_choice_probs.nonzero().size(0) == batch['passage']['tokens'].size(0), \
            'Sentence choice masking did not mask exactly one non-zero probability value for each sample in batch:' + \
            str(batch['metadata'])
        sent_choice_prob = masked_sent_choice_probs.sum(1)
        if nn_util.tensors_equal(sent_choice_prob, torch.zeros_like(sent_choice_prob)):
            logger.warning(
                'Likely masked out possible answer on accident: sent_choice_prob is exactly zero. Problem batch:',
                batch['metadata'])
        return sent_choice_prob

    def _get_sent_choice_prob_value(self, batch: TensorDict, sent_idxs: TensorDict, judge_answer_mask: TensorDict,
                                    debate_choice_mask: TensorDict, required_text_mask: TensorDict,
                                    past_sent_choice_idxs: List[torch.Tensor], stance: torch.Tensor,
                                    sent_answer_idx: torch.Tensor, num_sents: torch.Tensor, turn_str: str,
                                    for_training: bool, method: str, debate_mode: List[str], round_no: int) -> Tuple[
                                    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                    List[torch.Tensor]]:
        """
        Returns the sentence chosen by a particular policy.
        """
        choice_start_time = time.time()
        debater = None if self.model.is_judge else self.model
        judge = self.model if self.model.is_judge else self.model.judge
        bsz = batch['passage']['tokens'].size(0)
        value = torch.zeros(bsz)  # Default value (RL/Oracle agents only set this)
        turn_loss = None  # Optional variable to return (SL agents only)
        advantage = None  # Optional variable to return (Oracle only)
        all_values = None
        if method == 'r':  # Random selection
            sent_choice_idx = (torch.rand_like(num_sents.float()) * (num_sents.float())).trunc().long().unsqueeze(1)
            sent_choice_prob = torch.ones(bsz) / (num_sents.float())
        elif method == 'g':  # Ground truth, answer-containing selection
            assert sent_answer_idx is not None, 'No "Ground Truth" answer-supporting sentences provided.'
            sent_choice_idx = sent_answer_idx
            sent_choice_prob = torch.ones(bsz)
        elif method in self._search_debate_methods:
            # NOTE: Set below to None to make search selection simultaneous with other selections
            past_sent_choice_idxs = torch.cat(past_sent_choice_idxs, 1) if len(past_sent_choice_idxs) > 0 else None
            opt_sent_idxs = []
            advantage = []
            value = []
            all_values = []  # Will be returned for SL per-sentence targets (sl-sents). Only need to return for Oracle.
            judge_was_training = judge.training
            judge.eval()
            for sample_no in range(bsz):
                search_output_dict = self._get_search_output_dict(batch, sent_idxs, judge_answer_mask,
                    required_text_mask, past_sent_choice_idxs, num_sents, sample_no, debate_mode, round_no)
                search_metrics = self._get_reward(search_output_dict, stance[sample_no].unsqueeze(0), method, 'prob')
                if self._require_action and (past_sent_choice_idxs is not None):
                    search_metrics_for_valid_actions = search_metrics.clone()
                    search_metrics_for_valid_actions[past_sent_choice_idxs[sample_no]] = float('-inf')
                    search_metrics_for_valid_actions = search_metrics_for_valid_actions.tolist()
                    search_metrics = search_metrics.tolist()
                    opt_reward = float(max(search_metrics_for_valid_actions))
                    if opt_reward == float('-inf'):  # Full passage revealed, so just give no-action reward
                        opt_reward = float(max(search_metrics))
                else:
                    search_metrics = search_metrics.tolist()
                    opt_reward = float(max(search_metrics))
                baseline_reward = sum(search_metrics) / len(search_metrics)
                opt_sent_idxs.append(search_metrics.index(opt_reward))
                advantage.append(opt_reward - baseline_reward)
                all_values.append(search_metrics)
                value.append(opt_reward)
            if judge_was_training:
                judge.train()

            sent_choice_idx = torch.LongTensor(opt_sent_idxs).unsqueeze(1)
            sent_choice_prob = torch.ones(bsz)
            value = torch.Tensor(value)
            advantage = torch.Tensor(advantage)
        elif (method in self._learning_debate_methods) and (debater.reward_method == 'sl-random'):
            sent_choice_idx = (torch.rand_like(num_sents.float()) * (num_sents.float())).trunc().long().unsqueeze(1)
            sent_choice_prob = torch.ones(bsz) / (num_sents.float())
        elif method in self._learning_debate_methods:
            assert debater is not None, 'Cannot use debate method ' + method + ' without debate agents!'

            # Add some debater-specific batch info.
            batch['sent_targets'] = None
            batch['stance'] = stance
            # NOTE: input-level mask to predict a span over Judge's input tokens
            batch['valid_output_mask'] = debate_choice_mask['input']
            batch['all_past_sent_choice_mask'] = self._get_all_sent_choice_input_mask(sent_idxs, past_sent_choice_idxs)

            if debater.reward_method.startswith('sl'):
                search_round_no = round_no
                if self._single_shot and (round_no > 0):
                    search_round_no = 0
                # Get Oracle results
                search_sent_choice_idx, _, _, _, search_advantage, all_values = self._get_sent_choice_prob_value(
                    batch, sent_idxs, judge_answer_mask, debate_choice_mask, required_text_mask, past_sent_choice_idxs,
                    stance, sent_answer_idx, num_sents, turn_str, for_training, method.upper(), debate_mode, search_round_no)
                if debater.reward_method == 'sl':
                    answer_token_mask_input = ((search_sent_choice_idx == sent_idxs['input']).long() *
                                               batch['valid_output_mask'])
                    batch['sent_targets'] = answer_token_mask_input.nonzero()[:, 1].unsqueeze(-1)
                elif debater.reward_method.startswith('sl-sents'):
                    # Place all_values in appropriate EOS indices
                    search_token_values = sent_idxs['input'].clone().detach().float()
                    for i in range(bsz):
                        if num_sents[i].item() != len(all_values[i]):
                            logger.warning('Discrepancy in loaded Oracle num_sents ' + str(num_sents[i].item()) +
                                           ' and SL num_sents ' + str(len(all_values[i])) + ' for sample: ' +
                                           str(batch['metadata'][i]['id']))
                        target_shift = (sum(all_values[i]) / float(len(all_values[i]))) \
                            if debater.influence_reward else 0.
                        for sent_no in range(num_sents[i]):
                            search_token_values[i] = search_token_values[i].masked_fill(
                                sent_idxs['input'][i] == sent_no, all_values[i][sent_no] - target_shift)
                    search_token_values = nn_util.replace_masked_values(
                        search_token_values, batch['valid_output_mask'], -1e7)
                    batch['sent_targets'] = search_token_values

            if self._single_shot and (round_no > 0):
                # Use initial round's predictions
                debater_output_dict = self._single_shot_debater_output_dict_by_method[method]
            else:
                # Debate forward pass
                debater_output_dict = self._forward([batch], debater)
                if self._single_shot and (round_no == 0):  # Store result for later use
                    self._single_shot_debater_output_dict_by_method[method] = debater_output_dict

            # Set SL / debate-auxiliary losses for SGD
            turn_loss = debater_output_dict.get('loss')
            if turn_loss is not None:
                self._update_trainer_metrics('sl_loss' + turn_str, turn_loss.float().mean())
            if debater_output_dict['em'] is not None:
                self._update_trainer_metrics('debater_answer_acc' + turn_str, debater_output_dict['em'].float().mean())

            # Remove debater-specific batch info.
            batch['stance'] = None
            batch['valid_output_mask'] = None
            batch['sent_targets'] = None
            batch['all_past_sent_choice_mask'] = None

            if debater.reward_method.startswith('sl'):  # SL: Add predictions and loss
                # Get SL results
                word_choice_idx = torch.argmax(debater_output_dict['prob_dist'], dim=1, keepdim=True)
                if self._single_shot and (round_no > 0):
                    sorted_word_choice_idxs = torch.sort(debater_output_dict['prob_dist'], descending=True)
                    for i in range(bsz):
                        if sorted_word_choice_idxs[0][i, round_no] > 0:
                            word_choice_idx[i, 0] = sorted_word_choice_idxs[1][i, round_no]
                sent_choice_idx = sent_idxs['input'].gather(1, word_choice_idx.to(sent_idxs['input'].device))
                sent_choice_prob = self._get_sent_choice_prob_from_dist(  # Not required but good for logging/debugging
                    batch, sent_idxs, sent_choice_idx, debater_output_dict['prob_dist'])
                # Log statistics comparing SL to Oracle
                sl_acc = (search_sent_choice_idx == sent_choice_idx).float()
                self._update_trainer_metrics('sl_acc' + turn_str, sl_acc.mean())
                self._update_trainer_metrics('sl_first_sent_acc' + turn_str,
                                             (search_sent_choice_idx == 0).float().mean())
                self._update_trainer_metrics('sl_first_sent_preds' + turn_str, (sent_choice_idx == 0).float().mean())
                last_sent_idxs = sent_idxs['input'].max(dim=-1)[0]
                self._update_trainer_metrics('sl_last_sent_acc' + turn_str,
                                             (search_sent_choice_idx == last_sent_idxs).float().mean())
                self._update_trainer_metrics('sl_last_sent_preds' + turn_str,
                                             (sent_choice_idx == last_sent_idxs).float().mean())
                for i in range(-1, 10):
                    thres_start = i / 10.
                    thres_end = (i + 1) / 10.
                    thres_start_mask = (search_advantage.abs() > thres_start).float()
                    thres_end_mask = (thres_end >= search_advantage.abs()).float()
                    search_sc_diff_in_thres_idxs = (thres_start_mask * thres_end_mask).nonzero()
                    self._update_trainer_metrics('sl_num_per_batch_MaxScoreDrop_in_' + str(thres_end) + '_' +
                                                 str(thres_start) + turn_str,
                                                 torch.tensor(float(len(search_sc_diff_in_thres_idxs))))
                    for idx in search_sc_diff_in_thres_idxs:
                        self._update_trainer_metrics('sl_acc_where_MaxScoreDrop_in_' + str(thres_end) + '_' +
                                                     str(thres_start) + turn_str, sl_acc[idx])
            else:  # RL: Add predictions and value (no loss calculated yet)
                if for_training:  # Use probability of sampled sentence to calculate loss
                    word_choice_idx = torch.multinomial(debater_output_dict['prob_dist'], 1)
                else:  # Use best predicted sentence during evaluation
                    word_choice_idx = torch.argmax(debater_output_dict['prob_dist'], dim=1, keepdim=True)
                sent_choice_idx = sent_idxs['input'].gather(1, word_choice_idx.to(sent_idxs['input'].device))
                sent_choice_prob = self._get_sent_choice_prob_from_dist(
                    batch, sent_idxs, sent_choice_idx, debater_output_dict['prob_dist'])
                value = debater_output_dict['value']
        else:
            raise NotImplementedError('Unimplemented answer selection debate method', method)

        if sent_answer_idx is not None:
            answer_sent_chosen = (sent_choice_idx == sent_answer_idx).float()
            self._update_trainer_metrics('answer_sent_chosen' + turn_str, answer_sent_chosen.mean())

        self._update_trainer_metrics('time' + turn_str, torch.Tensor([time.time() - choice_start_time]))
        return sent_choice_idx, sent_choice_prob, value, turn_loss, advantage, all_values

    def _judge_text_masks(self, batch: TensorDict) -> Tuple[TensorDict, TensorDict]:
        """
        Returns the necessary input/output text masks for judge:
        limiting the output distributions to certain tokens and certain input tokens from being masked/changed.
        """
        bsz, input_dim = batch['passage']['tokens'].size()
        output_dim = self._get_output_dim(batch)

        # Ensure Judge receives question
        question_mask = {
            'input': torch.zeros(bsz, input_dim, dtype=torch.long),
            'output': torch.zeros(bsz, output_dim, dtype=torch.long)
        }
        required_text_mask = {
            'output': question_mask['output'],
            'input': question_mask['input']
        }

        # Ensure Judge receives answers. Limit where Judge can answer (if applicable)
        passage_mask = {'output': nn_util.get_text_field_mask(batch['passage'], 0)}
        judge_answer_mask = {'output': (passage_mask['output'] - question_mask['output']).clamp(min=0)}
        required_bert_tokens = {'[CLS]', '[SEP]'}
        for i in range(bsz):
            last_output_token_idx = self._get_last_output_token_idx(batch, i)
            for output_idx, output_token in enumerate(batch['metadata'][i]['passage_tokens']):
                if (output_token in required_bert_tokens) and (output_idx <= last_output_token_idx):
                    # NOTE: Mask will change after deletion (for final [SEP]).
                    required_text_mask['output'][i, output_idx] = 1.
        for required_bert_tok in required_bert_tokens:
            required_text_mask['input'] += (batch['passage']['tokens'] ==
                                            self.get_token_index(required_bert_tok)).long()
        # Clamp in case of double-counting (which shouldn't happen)
        required_text_mask['output'] = required_text_mask['output'].clamp(max=1)
        required_text_mask['input'] = required_text_mask['input'].clamp(max=1)
        return required_text_mask, judge_answer_mask

    def _debater_text_masks(self, batch: TensorDict, required_text_mask: TensorDict) -> TensorDict:
        """
        Returns where debaters can select sentences to quote/delete
        """
        bsz, input_dim = batch['passage']['tokens'].size()
        output_dim = self._get_output_dim(batch)
        debate_choice_mask = {
            'output': torch.zeros(bsz, output_dim, dtype=torch.long),
            'input': torch.zeros(bsz, input_dim, dtype=torch.long)
        }
        for i in range(bsz):
            last_output_token_idx = self._get_last_output_token_idx(batch, i)
            # Allow all EOS tokens
            for output_idx, output_token in enumerate(batch['metadata'][i]['passage_tokens']):
                if (output_token in self._eos_tokens) and (output_idx <= last_output_token_idx):
                    debate_choice_mask['output'][i, output_idx] = 1.
                    debate_choice_mask['input'][i, self._output_to_input_idx(batch, i, output_idx)] = 1.
            # Force last non-padding token to be an eos token in the mask
            debate_choice_mask['output'][i, last_output_token_idx] = 1.
            last_input_token_idx = self._output_to_input_idx(batch, i, last_output_token_idx)
            debate_choice_mask['input'][i, last_input_token_idx] = 1.
        debate_choice_mask['output'] *= (1. - required_text_mask['output'])
        debate_choice_mask['input'] *= (1. - required_text_mask['input'])
        return debate_choice_mask

    def _get_num_sents(self, debate_choice_mask: TensorDict) -> torch.Tensor:
        """
        Calculates number of choosable passage/input sentences. Also tracks mean number of sentences per passage.
        """
        num_output_sents = debate_choice_mask['output'].sum(1)
        num_input_sents = debate_choice_mask['input'].sum(1)
        assert nn_util.tensors_equal(num_output_sents, num_input_sents), \
            'Error: Discrepancy in # of output/input sentences:' + str(num_output_sents) + ', ' + str(num_input_sents)
        self._update_trainer_metrics('num_sents', num_output_sents.float().mean())
        return num_output_sents

    @staticmethod
    def _get_sent_idxs(required_text_mask: TensorDict, debate_choice_mask: TensorDict, num_sents: torch.Tensor,
                       version: str) -> TensorDict:
        """
        Returns the sentence indices of each word in the sequence (input or output -level).
        Padding regions have sent_idxs == num_sents. Required regions have -1.
        """
        sent_idxs = debate_choice_mask[version].cumsum(1) - debate_choice_mask[version]
        sent_idxs = sent_idxs.masked_fill(required_text_mask[version].byte(), -1.)
        sent_idxs = sent_idxs.masked_fill(sent_idxs == num_sents.unsqueeze(-1), -1.)
        return sent_idxs

    @staticmethod
    def _get_all_sent_choice_input_mask(sent_idxs: TensorDict, sent_choice_idxs: torch.Tensor) -> torch.Tensor:
        """
        Returns a mask over the input given all debater-chosen sentence indexes.
        """
        if len(sent_choice_idxs) == 0:
            return None
        sent_choice_input_masks = [(sent_idxs['input'] == sent_choice_idx) for sent_choice_idx in sent_choice_idxs]
        return torch.stack(sent_choice_input_masks).sum(0).clamp(max=1)

    @staticmethod
    def _get_reward(ver_dict, stance, debate_method, reward_method) -> torch.Tensor:
        """
        Returns the reward (without baseline) based on Judge score and debating method.
        """
        debate_method = debate_method.lower()  # Evaluate Oracle rewards too
        j_score = ver_dict[reward_method]
        if debate_method in {'a', 'r'}:
            rewards = j_score
        elif debate_method in {'b'}:
            rewards = (1. - j_score)
        elif debate_method in {'e', 'l', 'w', 'ⅰ', 'ⅱ', 'ⅲ', 'ⅳ'}:
            rewards = (ver_dict['prob_dist'] * stance.to(ver_dict['prob_dist'])).sum(dim=1)
        else:
            raise NotImplementedError('_get_reward not implemented for debate_method', debate_method)
        return rewards.detach()  # Judge shouldn't get gradients through j_score, used to reward debaters

    def _get_stance(self, batch: TensorDict, method: str) -> torch.Tensor:
        """
        Returns a new stance vector or matrix for a given agent method, if applicable.
        """
        method = method.lower()  # To get Oracle stances too
        bsz = batch['passage']['tokens'].size(0)
        if method in {'a', 'b'}:
            return torch.tensor([method == 'a'] * bsz).to(batch['passage']['tokens'])
        elif method in {'e'}:
            stance = torch.zeros_like(batch['options']['tokens'][:, :, 0])
            for i in range(bsz):
                sample_debate_mode = batch['metadata'][i]['debate_mode']
                # NOTE: Multi-turn debate_mode not yet supported here
                assert (len(sample_debate_mode) == 1) and (len(sample_debate_mode[0]) == 1)
                stance[i, self._method_to_stance_idx[sample_debate_mode[0][0]]] = 1
            return stance
        elif method in {'l', 'w'}:
            possible_stances = torch.ones_like(batch['options']['tokens'][:, :, 0])
            if method == 'w':  # Don't support correct option
                for i in range(bsz):
                    possible_stances[i, batch['answer_index'][i]] = 0
            stance_idx = torch.multinomial(possible_stances.float(), 1)
            stance = torch.zeros_like(batch['options']['tokens'][:, :, 0])  # stance later returned for loss func
            for i in range(bsz):
                stance[i, stance_idx[i]] = 1
            return stance
        elif method in {'ⅰ', 'ⅱ', 'ⅲ', 'ⅳ'}:
            stance = torch.zeros_like(batch['options']['tokens'][:, :, 0])
            stance[:, self._method_to_stance_idx[method]] = 1
            return stance
        elif method in {'r'}:
            return None
        else:
            if method == 'i':
                logger.warning('Did you mean to use method ⅰ (Roman Numeral) instead of i (Letter)?')
            raise NotImplementedError('_get_stance not implemented for method', method)

    def _get_all_stances(self, batch: TensorDict, debate_mode: List[str]) -> List[torch.Tensor]:
        """
        Gets stances for each agent and turn. Stances stay consistent across turns.
        """
        stances = {}
        for round_methods in debate_mode:
            for method in round_methods:
                stances[method] = stances.get(method, self._get_stance(batch, method))
        return stances

    def debate_batch_loss(self, batch: TensorDict, for_training: bool, debate_mode: List[str]) -> torch.Tensor:
        """
        Does a debate-style forward pass on a single batch in the group
        """
        # Useful aliases
        debater = None if self.model.is_judge else self.model
        judge = self.model if self.model.is_judge else self.model.judge
        bsz = batch['passage']['tokens'].size(0)
        output_dim = self._get_output_dim(batch)

        # Add token info to batch for BERT
        for i in range(bsz):
            batch['metadata'][i]['[SEP]'] = self.get_token_index('[SEP]')

        # Precomputation for all rounds
        required_text_mask, judge_answer_mask = self._judge_text_masks(batch)
        debate_choice_mask = self._debater_text_masks(batch, required_text_mask)
        num_sents = self._get_num_sents(debate_choice_mask)
        sent_idxs = {
            'output': self._get_sent_idxs(required_text_mask, debate_choice_mask, num_sents, 'output'),
            'input': self._get_sent_idxs(required_text_mask, debate_choice_mask, num_sents, 'input')
        }
        sent_answer_idx = None

        # Get Judge baseline opinion
        prev_round_ver_dict = None
        ver_dict_prior = None
        if (debater is not None) and (debater.influence_reward or debater.theory_of_mind):
            # Copy and modify Judge's input
            no_rounds_batch = self._slice_or_copy_batch(batch, copy=True)
            no_sent_choice_input_mask = torch.zeros_like(required_text_mask['input'])
            no_rounds_batch = self._modify_input_passage(no_rounds_batch, required_text_mask, no_sent_choice_input_mask)

            # Judge forward pass
            choice_start_time = time.time()
            ver_dict_prior = self._forward([no_rounds_batch], judge)
            prev_round_ver_dict = ver_dict_prior
            self._update_trainer_metrics('time_j_baseline', torch.Tensor([time.time() - choice_start_time]))
            # NOTE: Optional: if self.model.update_judge: loss += prev_round_ver_dict['loss']

        if (torch.rand([1]) < self._x_order_prob).item():
            debate_mode.reverse()

        if self._num_pred_rounds > 0:
            pred_round_nos = torch.multinomial(torch.ones([len(debate_mode)]), self._num_pred_rounds, replacement=False)
            debate_mode = [round_mode if (round_no in pred_round_nos) else round_mode.upper()
                           for round_no, round_mode in enumerate(debate_mode[:pred_round_nos.max() + 1])]

        # Execute turns and accumulate loss across rounds.
        # Decisions/turns within a single round are sequential for Oracles, simultaneous otherwise.
        loss = torch.Tensor([0]).cuda() if torch.cuda.is_available() else torch.Tensor([0])
        sent_choice_idxs, sent_choice_probs, values, advantage = [], [], [], None
        turns_completed = 0
        ver_dicts = []
        stances = self._get_all_stances(batch, debate_mode)
        for round_no in range(len(debate_mode)):
            round_sent_choice_idxs, round_sent_choice_probs, round_values = [], [], []
            for round_turn_no, method in enumerate(debate_mode[round_no]):
                # Execute a single player's turn to determine decisions
                turn_no = turns_completed + round_turn_no
                turn_str = "_turn_" + str(turn_no) + "_agent_" + method
                sent_choice_idx, sent_choice_prob, value, turn_loss, advantage, _ = \
                    self._get_sent_choice_prob_value(batch, sent_idxs, judge_answer_mask, debate_choice_mask,
                                                     required_text_mask, sent_choice_idxs, stances[method],
                                                     sent_answer_idx, num_sents, turn_str, for_training, method,
                                                     debate_mode, round_no)
                round_sent_choice_idxs.append(sent_choice_idx)
                round_sent_choice_probs.append(sent_choice_prob)
                round_values.append(value)
                if turn_loss is not None:
                    loss += turn_loss  # Add Loss: SL (primary or auxiliary losses)
            sent_choice_idxs += round_sent_choice_idxs
            sent_choice_probs += round_sent_choice_probs
            values += round_values

            # Modify Judge's input
            all_sent_choice_input_mask = self._get_all_sent_choice_input_mask(sent_idxs, sent_choice_idxs)
            judge_batch = self._slice_or_copy_batch(batch, copy=True)
            judge_batch = self._modify_input_passage(judge_batch, required_text_mask, all_sent_choice_input_mask)

            # Judge forward pass
            choice_start_time = time.time()
            ver_dict = self._forward([judge_batch], judge)
            ver_dicts.append(ver_dict)
            self._update_trainer_metrics('time_j', torch.Tensor([time.time() - choice_start_time]))

            # Add Loss: Judge
            if (debater is None) or self.model.update_judge:
                loss += ver_dict['loss']

            # Add Loss: RL agents
            for round_turn_no, method in enumerate(debate_mode[round_no]):
                # Log rewards and stats for all methods
                turn_no = turns_completed + round_turn_no
                turn_str = "_turn_" + str(turn_no) + "_agent_" + method
                reward_to_track = 'prob' if ((debater is None) or (debater.reward_method.startswith('sl'))) \
                    else debater.reward_method
                rewards = self._get_reward(ver_dict, stances[method], method, reward_to_track)
                self._update_trainer_metrics('reward' + turn_str, rewards.mean())
                self._update_trainer_metrics(
                    'reward_std' + turn_str,
                    (rewards.cpu() - self._trainer_metrics['reward' + turn_str].get_metric()).abs().mean())
                self._update_trainer_metrics('sent_choice_prob' + turn_str, sent_choice_probs[turn_no].mean())
                if 'em' in ver_dict:
                    self._update_trainer_metrics('judge_em' + turn_str, ver_dict['em'].mean())
                if 'prob' in ver_dict:
                    self._update_trainer_metrics('judge_prob' + turn_str, ver_dict['prob'].mean())
                # Log statistics based on if stance-conditioned debater was defending correct stance
                if method.lower() in {'e', 'l', 'w', 'ⅰ', 'ⅱ', 'ⅲ', 'ⅳ'}:
                    stance_was_correct = stances[method].gather(1, judge_batch['answer_index']).squeeze(1).float()
                    self._update_trainer_metrics('stance_was_correct' + turn_str, stance_was_correct.mean())
                    stance_was_correct = stance_was_correct.tolist()
                    correctness_str = {0: '_incorrect_stance', 1: '_correct_stance'}
                    for i in range(bsz):
                        self._update_trainer_metrics(
                            'reward' + turn_str + correctness_str[stance_was_correct[i]], rewards[i])
                        if 'em' in ver_dict:
                            self._update_trainer_metrics(
                                'judge_em' + turn_str + correctness_str[stance_was_correct[i]], ver_dict['em'][i])
                        if 'prob' in ver_dict:
                            self._update_trainer_metrics(
                                'judge_prob' + turn_str + correctness_str[stance_was_correct[i]], ver_dict['prob'][i])

                if (debater is not None) and debater.influence_reward:
                    prev_round_rewards = self._get_reward(prev_round_ver_dict, stances[method], method, reward_to_track)
                    rewards = rewards - prev_round_rewards
                    self._update_trainer_metrics('influence_reward' + turn_str, rewards.mean())
                    # Log statistics based on if stance-conditioned debater was defending correct stance
                    if method.lower() in {'e', 'l', 'w', 'ⅰ', 'ⅱ', 'ⅲ', 'ⅳ'}:
                        for i in range(bsz):
                            self._update_trainer_metrics(
                                'influence_reward' + turn_str + correctness_str[stance_was_correct[i]], rewards[i])

                if (method not in self._learning_debate_methods) or (debater is None) or (
                        debater.reward_method.startswith('sl')):
                    continue  # Don't apply RL losses in the above cases

                # Calculate RL losses
                baselines = values[turn_no]
                policy_losses = -(torch.log(sent_choice_probs[turn_no]) * (rewards - baselines.detach()))
                loss += policy_losses.mean()
                value_losses = 0.5 * ((baselines - rewards) ** 2)
                loss += value_losses.mean()

                # Log RL-only stats
                self._update_trainer_metrics('policy_loss' + turn_str, policy_losses.mean())
                self._update_trainer_metrics('value_loss' + turn_str, value_losses.mean())  # Upper bound ~= .125
                self._update_trainer_metrics('baseline' + turn_str, baselines.mean())
                self._update_trainer_metrics(
                    'baseline_std' + turn_str,
                    (baselines - self._trainer_metrics['baseline' + turn_str].get_metric()
                     ).abs().mean())  # Should be high
                self._update_trainer_metrics('advantage' + turn_str, (rewards - baselines).mean())
                self._update_trainer_metrics('advantage_abs' + turn_str,
                                             (rewards - baselines).abs().mean())  # Should have low variance
                # Log stats based on if stance-conditioned debater was defending correct stance
                if method in {'e', 'l', 'w', 'ⅰ', 'ⅱ', 'ⅲ', 'ⅳ'}:
                    for i in range(bsz):
                        self._update_trainer_metrics(
                            'policy_loss' + turn_str + correctness_str[stance_was_correct[i]],policy_losses[i])
                        self._update_trainer_metrics(
                            'baseline' + turn_str + correctness_str[stance_was_correct[i]], baselines[i])
                        self._update_trainer_metrics(
                            'value_loss' + turn_str + correctness_str[stance_was_correct[i]],
                            value_losses[i])  # Upper bound ~= .125
                        if debater.influence_reward:
                            self._update_trainer_metrics(
                                'influence_reward' + turn_str + correctness_str[stance_was_correct[i]], rewards[i])

            prev_round_ver_dict = ver_dict  # Update baseline Judge scores for next round
            turns_completed += len(debate_mode[round_no])

        if self._eval_mode:
            self._log_debate(batch, sent_idxs, ver_dicts, sent_choice_idxs, debate_mode, stances, num_sents, advantage,
                             ver_dict_prior)

        return loss.cpu()

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool, debate_mode: List[str] = None
                   ) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        # If overriding default passage choosing method
        if debate_mode is None:
            debate_mode = self._debate_mode

        # Print IDs to know which sample(s) are problematic before a forward pass crash
        if self._breakpoint_level >= 1:
            for batch in batch_group:
                for i in range(batch['passage']['tokens'].size(0)):
                    print('ID:', batch['metadata'][i]['id'], '...')

        # Set output_dict['loss'] to do gradient descent on.
        if debate_mode[0] == "f":  # Full passage training: Normal SL training (always on Judge)
            judge = self.model if self.model.is_judge else self.model.judge
            output_dict = self._forward(batch_group, judge)
            if (not for_training) and (len(batch_group) == 1):  # len(batch_group) > 1 not tested for logging
                self._log_verdicts(batch_group[0], [output_dict], debate_mode)
        else:  # Training on subset of sentence (judge or debate training)
            losses = [self.debate_batch_loss(batch, for_training, debate_mode) for batch in batch_group]
            losses = torch.cat([loss.unsqueeze(0) for loss in losses], 0)
            output_dict = {'loss': losses.mean()}
        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0

        # Set the model to "train" mode.
        self.model.train()
        if (not self.model.update_judge) and (self.model.judge is not None):
            self.model.judge.eval()

        # As this is only used for GPU Data Parallel Mode, check _allocation_dict first
        num_gpus = len(self._cuda_devices) if (self._allocation_dict is None) else 1

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data,
                                            num_epochs=1,
                                            shuffle=self.shuffle)
        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data)/num_gpus)
        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator, total=num_training_batches)

        cumulative_batch_size = 0
        for batch_group in train_generator_tqdm:
            batches_this_epoch += 1
            grad_steps_this_epoch = (batches_this_epoch // self._accumulation_steps)
            self._batch_num_total += 1  # NOTE: Should use _grad_steps_num_total instead but may affect model loading
            batch_num_total = self._batch_num_total

            if (batch_num_total - 1) % self._accumulation_steps == 0:
                self.optimizer.zero_grad()

            loss = self.batch_loss(batch_group, for_training=True)

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")
            if not loss.requires_grad:
                raise RuntimeError("Training loss does not require gradient. Cannot do backward pass.")

            loss = loss / self._accumulation_steps
            backward_start_time = time.time()
            loss.backward()
            self._update_trainer_metrics('time_backward', torch.Tensor([time.time() - backward_start_time]))

            train_loss += loss.item()

            if batch_num_total % self._accumulation_steps != 0:
                continue  # Don't step with optimizer: accumulate gradients

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using an
            # LRScheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)

            if self._tensorboard.should_log_histograms_this_batch():
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                param_updates = {name: param.detach().cpu().clone()
                                 for name, param in self.model.named_parameters()}
                self.optimizer.step()
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                       update_norm / (param_norm + 1e-7))
            else:
                self.optimizer.step()

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(self.model, train_loss, grad_steps_this_epoch, self._trainer_metrics)
            description = training_util.description_from_metrics(metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                # self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self.model, self.optimizer)

                self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
                self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self.model, histogram_parameters)

            if self._log_batch_size_period:  # NOTE: Could be inaccurate due to gradient accumulation, but not used now
                cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
                cumulative_batch_size += cur_batch
                if (grad_steps_this_epoch - 1) % self._log_batch_size_period == 0:
                    average = cumulative_batch_size/grad_steps_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    self._tensorboard.add_train_scalar("mean_batch_size", average)

            # Save model if needed.
            if self._model_save_interval is not None and (
                    time.time() - last_save_time > self._model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time)))
                )
        metrics = training_util.get_metrics(
            self.model, train_loss, grad_steps_this_epoch, self._trainer_metrics, reset=True)
        metrics['cpu_memory_MB'] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory
        return metrics

    def _validation_loss(self, debate_mode: List[str] = None) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")
        if debate_mode is None:
            debate_mode = self._debate_mode

        self.model.eval()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        # As this is only used for GPU Data Parallel Mode, check _allocation_dict first
        num_gpus = len(self._cuda_devices) if (self._allocation_dict is None) else 1

        raw_val_generator = val_iterator(self._validation_data, num_epochs=1, shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(val_iterator.get_num_batches(self._validation_data)/num_gpus)
        val_generator_tqdm = Tqdm.tqdm(val_generator, total=num_validation_batches)

        batches_this_epoch = 0
        val_loss = 0
        for batch_group in val_generator_tqdm:

            loss = self.batch_loss(batch_group, for_training=False, debate_mode=debate_mode)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch, self._trainer_metrics)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        for epoch in range(epoch_counter, self._num_epochs + self._eval_mode):
            if not self._eval_mode:
                epoch_start_time = time.time()
                train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0),
                                                    train_metrics['cpu_memory_MB'])
            for key, value in train_metrics.items():
                if key.startswith('gpu_'):
                    metrics["peak_"+key] = max(metrics.get("peak_"+key, 0), value)

            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, self._trainer_metrics, reset=True)

                    # Check validation metric for early stopping
                    if not self._eval_mode:
                        this_epoch_val_metric = val_metrics[self._validation_metric]
                        self._metric_tracker.add_metric(this_epoch_val_metric)

                        if self._metric_tracker.should_stop_early():
                            logger.info("Ran out of patience.  Stopping training.")
                            break

            if not self._eval_mode:
                self._tensorboard.log_metrics(train_metrics, val_metrics=val_metrics, log_to_console=True)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = time.strftime("%H:%M:%S", time.gmtime(training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if (not self._eval_mode) and self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

            if self._serialization_dir:
                # Save search_outputs mapping if it's newly computed
                if (not self._search_outputs_is_complete) and self._using_search:
                    self._search_outputs_is_complete = True
                    with open(self._search_outputs_path, 'wb') as f:
                        pickle.dump(self._search_outputs, f, pickle.HIGHEST_PROTOCOL)
                    logger.info('Saved search_outputs to: ' + self._search_outputs_path)
                if self._eval_mode:
                    dump_metrics(os.path.join(self._serialization_dir,
                                              f'metrics_epoch_{epoch}.d=' + '_'.join(self._debate_mode) + '.json'),
                                 metrics)
                    dump_metrics(os.path.join(self._serialization_dir,
                                              f'debate_logs.d=' + '_'.join(self._debate_mode) + '.json'),
                                 self._debate_logs)
                else:
                    dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            if self._eval_mode:
                return metrics

            if self._learning_rate_scheduler:
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # These are the training states we need to persist.
        training_states = {
                "metric_tracker": self._metric_tracker.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "batch_num_total": self._batch_num_total
        }

        # If we have a learning rate scheduler, we should persist that too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = (
                    self._learning_rate_scheduler.lr_scheduler.state_dict()
            )

        self._checkpointer.save_checkpoint(
                model_state=self.model.state_dict(),
                epoch=epoch,
                training_states=training_states,
                is_best_so_far=self._metric_tracker.is_best_so_far())

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self._learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
            self._learning_rate_scheduler.lr_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the ``training_state`` contains a serialized ``MetricTracker``.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metrics_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metrics_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(cls,  # type: ignore
                    model: Model,
                    serialization_dir: str,
                    debate_mode: List[str],
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    params: Params,
                    validation_iterator: DataIterator = None,
                    eval_mode: bool = False,
                    breakpoint_level: int = 0,
                    search_outputs_path: str = None,
                    accumulation_steps: int = 1,
                    allocation_dict: Dict[str, int] = None,
                    choice_mode: str = None,
                    num_pred_rounds: int = -1,
                    x_order_prob: float = 0.,
                    require_action: bool = False,
                    single_shot: bool = False) -> 'Trainer':

        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)

        # Move models to appropriate devices (if running in multi-gpu mode) => ensures optimizer created properly
        if len(allocation_dict) == 3:
            # Move model depending on whether or not it is the judge
            if model.is_judge:
                model = model.cuda(allocation_dict['judge'])
            else:
                # Move entire model to debate GPU (gross, but best you can do)
                model = model.cuda(allocation_dict['debate'])

                # Move model.judge to judge GPU
                model.judge = model.judge.cuda(allocation_dict['judge'])

        # Otherwise, if not multi-gpu mode, run with the default behavior
        else:
            model_device = cuda_device if isinstance(cuda_device, int) else cuda_device[0]
            if model_device >= 0:
                model = model.cuda(model_device)

        # Creates Optimizer Parameters on the appropriate GPU
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        if lr_scheduler_params:
            scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            scheduler = None

        num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
        keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   debate_mode,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=scheduler,
                   num_serialized_models_to_keep=num_serialized_models_to_keep,
                   keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   log_batch_size_period=log_batch_size_period,
                   eval_mode=eval_mode,
                   breakpoint_level=breakpoint_level,
                   search_outputs_path=search_outputs_path,
                   accumulation_steps=accumulation_steps,
                   allocation_dict=allocation_dict,
                   choice_mode=choice_mode,
                   num_pred_rounds=num_pred_rounds,
                   x_order_prob=x_order_prob,
                   require_action=require_action,
                   single_shot=single_shot)


class TrainerPieces(NamedTuple):
    """
    We would like to avoid having complex instantiation logic taking place
    in `Trainer.from_params`. This helper class has a `from_params` that
    instantiates a model, loads train (and possibly validation and test) datasets,
    constructs a Vocabulary, creates data iterators, and handles a little bit
    of bookkeeping. If you're creating your own alternative training regime
    you might be able to use this.
    """
    model: Model
    iterator: DataIterator
    train_dataset: Iterable[Instance]
    validation_dataset: Iterable[Instance]
    test_dataset: Iterable[Instance]
    validation_iterator: DataIterator
    params: Params

    @staticmethod
    def from_params(params: Params,
                    serialization_dir: str,
                    cuda_device: Union[int, List],
                    recover: bool = False,
                    judge_filename: str = None,
                    update_judge: bool = False,
                    eval_mode: bool = False,
                    reward_method: str = None,
                    detach_value_head: bool = False,
                    allocation_dict: Dict[str, int] = None,
                    qa_loss_weight: float = 0.,
                    influence_reward: bool = False,
                    theory_of_mind: bool = False) -> 'TrainerPieces':

        all_datasets = training_util.datasets_from_params(params)
        if eval_mode:
            params["datasets_for_vocab_creation"] = ['train', 'validation']
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

        for dataset in datasets_for_vocab_creation:
            if dataset not in all_datasets:
                raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

        logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                    ", ".join(datasets_for_vocab_creation))

        if recover and os.path.exists(os.path.join(serialization_dir, "vocabulary")):
            vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
        else:
            vocab = Vocabulary.from_params(
                    params.pop("vocabulary", {}),
                    (instance for key, dataset in all_datasets.items()
                     for instance in dataset
                     if key in datasets_for_vocab_creation)
            )

        # Debate: Load judge model from archive (if applicable)
        judge = None
        update_judge = update_judge and (judge_filename is not None)
        if judge_filename is not None:
            judge_file_ending = judge_filename.split('.')[-1]
            if judge_file_ending == 'gz':
                # Place on appropriate GPU if allocation_dict is specified
                judge_device = allocation_dict.get('judge',
                                                   cuda_device if isinstance(cuda_device, int) else cuda_device[-1])
                archive = load_archive(judge_filename, cuda_device=judge_device)
                config = archive.config
                prepare_environment(config)
                judge = archive.model

            elif judge_file_ending == 'json' or judge_file_ending == 'jsonnet':
                # NB: No overrides possible for judge (keeps training config). Also, only 'model' field is used.
                judge_params = Params.from_file(judge_filename, params_overrides='')

                # NB: No judge inside this model
                judge = Model.from_params(vocab=vocab, params=judge_params.get('model'),
                                          judge=None, update_judge=False, reward_method=None,
                                          detach_value_head=False)
                if not update_judge:
                    warnings.warn('Provided Judge file was a training config file. '
                                  'Training from scratch even though -u was not specified.', UserWarning)
                    update_judge = True

            # Whether to use judge only for black-box reward (no gradient signal)
            if not update_judge:
                judge.eval()
            for parameter in judge.parameters():
                parameter.requires_grad_(update_judge)

        model = Model.from_params(vocab=vocab, params=params.pop('model'),
                                  judge=judge, update_judge=update_judge, reward_method=reward_method,
                                  detach_value_head=detach_value_head, qa_loss_weight=qa_loss_weight,
                                  influence_reward=influence_reward, theory_of_mind=theory_of_mind)

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(model.vocab)
        validation_iterator_params = params.pop("validation_iterator", None)
        if validation_iterator_params:
            validation_iterator = DataIterator.from_params(validation_iterator_params)
            validation_iterator.index_with(model.vocab)
        else:
            validation_iterator = None

        train_data = all_datasets['train']
        validation_data = all_datasets.get('validation')
        test_data = all_datasets.get('test')
        if eval_mode and (test_data is not None):
            validation_data = test_data

        trainer_params = params.pop("trainer")
        no_grad_regexes = trainer_params.pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        return TrainerPieces(model, iterator,
                             train_data, validation_data, test_data,
                             validation_iterator, trainer_params)
