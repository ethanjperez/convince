import json
import logging
import os
from typing import Dict, List, Tuple

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("babi_single")
class BabiSingleReader(DatasetReader):
    """
    Reads a JSON-formatted bAbI file and returns a ``Dataset`` where the ``Instances`` have four
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
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for article in dataset:
            for paragraph_json in article['paragraphs']:
                paragraph = paragraph_json["context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                for question_answer in paragraph_json['qas']:
                    question_text = question_answer["question"].strip().replace("\n", "")
                    answer_texts = [answer['text'] for answer in question_answer['answers']]
                    span_starts = [answer['answer_start'] for answer in question_answer['answers']]
                    span_ends = [start + len(answer) for start, answer in zip(span_starts, answer_texts)]
                    instance = self.text_to_instance(question_text,
                                                     paragraph,
                                                     zip(span_starts, span_ends),
                                                     answer_texts,
                                                     tokenized_paragraph,
                                                     question_answer['id'])
                    yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         question_text: str,
                         passage_text: str,
                         char_spans: List[Tuple[int, int]] = None,
                         answer_texts: List[str] = None,
                         passage_tokens: List[Token] = None,
                         qa_id: str = None) -> Instance:
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans: List[Tuple[int, int]] = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                logger.debug("Passage: %s", passage_text)
                logger.debug("Passage tokens: %s", passage_tokens)
                logger.debug("Question text: %s", question_text)
                logger.debug("Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug("Token span: (%d, %d)", span_start, span_end)
                logger.debug("Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug("Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        return util.make_reading_comprehension_instance(self._tokenizer.tokenize(question_text),
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts,
                                                        {'id': qa_id})


if __name__ == "__main__":
    # Parses babi Race Files into Single JSON Format
    # Assumes race_raw directory lives in "datasets/race_raw" and you're running from allenlp dir
    race_raw_path = "datasets/babi_raw"
    babi_path = "datasets/babi"

    if not os.path.exists(babi_path):
        os.mkdir(babi_path)

    # Create Data Dictionary
    train_data, val_data, test_data = {"data": []}, {"data": []}, {"data": []}

    for dt in ['train', 'valid', 'test']:
        dt_path = os.path.join(race_raw_path, "qa1_%s.txt" % dt)
        with open(dt_path, 'r') as f:
            lines = f.readlines()

        # Iterate through lines, and start building story, question, answer, supporting fact pairs
        stories, questions, answers, sub_story = [], [], [], []
        for i in range(len(lines)):
            # Question/Answer/Supporting Fact Line
            if '\t' in lines[i]:
                _, line = lines[i].split(' ', 1)
                q, a, sup_fact = map(str.strip, line.split('\t'))

                # Assemble story so far, get correct answer span (in appropriate answer fact)
                ids, sentences = map(list, list(zip(*sub_story)))

                # Add spaces to beginning of each sentence starting with the second
                for j in range(1, len(sentences)):
                    sentences[j] = " " + sentences[j]

                # Find supporting fact sentence
                s_idx, span_start = ids.index(int(sup_fact)), 0

                # Create final story, identify answer start span
                composite_story, span_tracker, answer_start_span = "", 0, 0
                for j in range(len(sentences)):
                    if j == s_idx:
                        answer_start_span = span_tracker + sentences[j].index(a)

                    composite_story += sentences[j]
                    span_tracker += len(sentences[j])

                # Add to trackers
                stories.append(composite_story)
                questions.append(q)
                answers.append((answer_start_span, a))

            # Regular (Story) Line
            else:
                # Get line number, and processed line (strip)
                n_id, line = lines[i].split(' ', 1)
                n_id, line = int(n_id), line.strip()

                # Reset stories, sub_story if n_id == 1
                if n_id == 1:
                    sub_story = []

                # Add n_id, line to substory
                sub_story.append((n_id, line))

        # Assemble JSON Files
        for i in range(len(stories)):
            story, question, answer = stories[i], questions[i], answers[i]

            # Create necessary fields
            article_dict = {"title": "story_%d" % i, "paragraphs": []}

            # Create single-entry paragraph dictionary
            paragraph_dict = {"context": story,
                              "qas": [{"answers": [{"answer_start": answer[0], "text": answer[1]}],
                                       "question": question,
                                       "id": hex(hash("story_%d_query_%s" % (i, question)))[2:]}]
                              }
            article_dict["paragraphs"].append(paragraph_dict)

            # Add article dict to data
            if dt == 'train':
                train_data["data"].append(article_dict)
            elif dt == 'valid':
                val_data["data"].append(article_dict)
            elif dt == 'test':
                test_data["data"].append(article_dict)

    # Dump JSON Files
    with open(os.path.join(babi_path, "babi-train-v1.0.json"), 'w') as f:
        json.dump(train_data, f)

    with open(os.path.join(babi_path, "babi-dev-v1.0.json"), 'w') as f:
        json.dump(val_data, f)

    with open(os.path.join(babi_path, "babi-test-v1.0.json"), 'w') as f:
        json.dump(test_data, f)
