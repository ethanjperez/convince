import json
import logging
import os
from typing import Dict, List

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, IndexField, MetadataField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("dream-mc")
class DreamMCReader(DatasetReader):
    """
    Reads a JSON-formatted Race file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 debate_mode: List[str] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._letter_to_answer_idx = {'A': 0, 'B': 1, 'C': 2}  # NB: Dream Only Has 3 Answers
        self._debate_modes = {
            ('e',): [['ⅰ'], ['ⅱ'], ['ⅲ']],
            ('E',): [['Ⅰ'], ['Ⅱ'], ['Ⅲ']],
        }.get(tuple(debate_mode), [None])

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Iterate through Dialogues
        for diag in data:
            passage, qas, diag_id = diag

            # Concatenate Passage Turns + Tokenize
            passage_text = " ".join(passage)
            tokenized_passage = self._tokenizer.tokenize(passage_text)

            # Iterate through QAs
            for idx, qa in enumerate(qas):
                question, choices, answer = qa['question'], qa['choice'], qa['answer']
                answer_index = choices.index(answer)

                # Generate a Question ID by adding to diag_id
                qid = diag_id + "-q%d" % idx

                # Yield!!!
                for debate_mode in self._debate_modes:
                    yield self.text_to_instance(question, passage_text, choices, qid, tokenized_passage, answer_index,
                                                debate_mode)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         options_text: List[str],
                         qa_id: str,
                         passage_tokens: List[Token] = None,
                         answer_index: int = None,
                         debate_mode: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        additional_metadata = {'id': qa_id, 'debate_mode': debate_mode}
        fields: Dict[str, Field] = {}

        # Tokenize and calculate token offsets (i.e., for wordpiece)
        question_tokens = self._tokenizer.tokenize(question_text)
        if passage_tokens is None:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        options_tokens = self._tokenizer.batch_tokenize(options_text)
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

        # This is separate so we can reference it later with a known type.
        options_field = ListField([TextField(option_tokens, self._token_indexers) for option_tokens in options_tokens])
        fields['passage'] = TextField(passage_tokens, self._token_indexers)
        fields['question'] = TextField(question_tokens, self._token_indexers)
        fields['options'] = options_field
        metadata = {'original_passage': passage_text, 'token_offsets': passage_offsets,
                    'question_tokens': [token.text for token in question_tokens],
                    'passage_tokens': [token.text for token in passage_tokens],
                    'options_tokens': [[token.text for token in option_tokens] for option_tokens in options_tokens]}
        if answer_index is not None:
            metadata['answer_texts'] = options_text[answer_index]

        fields['answer_index'] = IndexField(answer_index, options_field)

        metadata.update(additional_metadata)
        fields['metadata'] = MetadataField(metadata)
        return Instance(fields)
