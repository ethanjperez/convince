#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FixedDialogTeacher
from .build import build

import json
import os


class IndexTeacher(FixedDialogTeacher):
    """Hand-written dataset teacher, which loads the json squad data and
    implements its own `act()` method for interacting with student agent,
    rather than inheriting from the core Dialog Teacher. This code is here as
    an example of rolling your own without inheritance.

    This teacher also provides access to the "answer_start" indices that
    specify the location of the answer in the context.
    """

    def __init__(self, opt, shared=None):
        build(opt)
        super().__init__(opt, shared)
        self.id = 'race'
        self._letter_to_answer_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self._answer_idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

        self.use_bad_qid = ((opt['evaluation_data_dir'] is not None) and ('test' in opt['evaluation_data_dir'])
                            and ('fasttext.o' in opt['evaluation_data_dir']))

        datapath = os.path.join(
            opt['datapath'],
            'RACE',
            self.datatype
        )
        self.data = self._setup_data(datapath, opt['question_split_no'], opt['question_splits'])

        self.reset()

    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        return self.examples[episode_idx]

    def _setup_data(self, file_path, split_no=0, num_splits=1):
        self.examples = []
        for level in ['high', 'middle']:
            # Get all articles
            file_level_path = os.path.join(file_path, level)
            if not os.path.exists(file_level_path):
                continue
            articles = os.listdir(file_level_path)
            articles.sort()
            for article in articles:
                art_file = os.path.join(file_level_path, article)
                with open(art_file, 'rb') as f:
                    art_data = json.load(f)

                # Article-level info
                title = art_data["id"]
                passage_text = art_data["article"]

                # Iterate through questions
                for q in range(len(art_data["questions"])):
                    question_text = art_data["questions"][q].strip().replace("\n", "")
                    options_text = art_data["options"][q]
                    for option_no in range(len(options_text)):
                        options_text[option_no] = self._answer_idx_to_letter[option_no] + ': ' + options_text[option_no]
                    answer_index = self._letter_to_answer_idx[art_data["answers"][q]]
                    qid = self._filepath_to_id(art_file, q)
                    if self.use_bad_qid:
                        qid = qid.replace('test/', 'dev/')  # Add bug to QID to match files
                    self.examples.append({
                        'id': self.id,
                        'text': '\n'.join([passage_text + '\n', question_text + '\n'] + options_text),
                        'labels': options_text[answer_index],
                        'episode_done': True,
                        'answer_starts': answer_index,
                        'title': title,
                        'qid': qid,
                        'passage': passage_text,
                        'question': question_text,
                        'options': options_text,
                        'question_type_labels': art_data['question_type_labels'][q],
                    })
        self.examples = self.examples[split_no::num_splits]

    @staticmethod
    def _filepath_to_id(filepath: str, q_no: int) -> str:
        file_parts = os.path.join(filepath, str(q_no)).split('/')[-4:]
        for split in ['train', 'dev', 'test']:
            if split in file_parts[0]:
                file_parts[0] = split
            elif file_parts[0] in {'A', 'B', 'C', 'D', 'E'}:
                file_parts[0] = 'test'  # Question-type datasets come from test
        return '/'.join(file_parts)
