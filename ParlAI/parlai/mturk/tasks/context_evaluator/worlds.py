#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkTaskWorld
import random


class ContextEvaluationWorld(MTurkTaskWorld):
    """
    World for recording a turker's question and answer given a context.
    Assumes the context is a random context from a given task, e.g.
    from SQuAD, CBT, etc.
    """

    def __init__(self, opt, task, mturk_agent, evaluation_data):
        self.reward = opt['reward']
        self.is_sandbox = opt['is_sandbox']
        self.question_split_no = opt['question_split_no']
        self.option_split_no = opt['option_split_no']
        self.dataset = opt['dataset']
        self.task = task
        self.mturk_agent = mturk_agent
        self.evaluation_data = evaluation_data
        self.episodeDone = False

        # Maintain data, counts, and stats
        self.max_collected = len(self.task.examples)
        self.cur_example_no = 1  # 1-indexed (shown to user)
        self.num_collected = 0
        self.num_tested = 0
        self.num_changed_responses = None
        self.num_debate_mode_responses = None
        self.data = []
        self.num_correct_on_labeled = {}
        self.num_collected_on_labeled = {}
        self.accuracy = {}
        self.answer_to_count_by_prompt = {}
        self.durations = {}
        self.reject_reasons = []
        self.block_reasons = []
        self.bonus_reasons = []
        self.quote_rating = None
        self.quote_description = None
        self.task_rating = None
        self.feedback = None
        self.hit_done = False
        self.freq_changed_responses = None

        # Prompt type differences
        self.prompt_types = [opt['prompt_type']]
        assert len(self.prompt_types) == 1, 'Using multiple prompt_types not yet supported.'
        self.prompt_type_description = {
            'question': 'just the questions and answers',
            'quote and question': 'just a short quote from the passage',
            'question and quotes': 'just a few quotes from the passage',
            'quotes and question': 'just a few quotes from the passage',
            'passage and question': 'the passage',
        }
        self.accuracy_bonus_multiplier = {
            'question': .5,
            'quote and question': .5,
            'question and quotes': 1.0,
            'quotes and question': 1.0,
            'passage and question': 1.0,
        }[self.prompt_types[0]]
        self.accuracy_bonus_threshold = {
            'dream': {
                'question': .5,
                'quote and question': .6,
                'question and quotes': .8,
                'quotes and question': .85,
                'passage and question': .95,
            },
            'race': {
                'question': .55,
                'quote and question': .6,
                'question and quotes': .7,
                'quotes and question': .75,
                'passage and question': .9,
            },
        }[self.dataset]
        self.median_sample_ms_reject_threshold = {
            'dream': {
                'question': 4000,
                'quote and question': 4500,
                'question and quotes': 7000,
                'quotes and question': 7000,
                'passage and question': 7000,
            },
            'race': {
                'question': 6000,
                'quote and question': 7000,
                'question and quotes': 10000,
                'quotes and question': 10000,
                'passage and question': 22000,
            },
        }[self.dataset]
        self.response_freq_reject_threshold = {
            3: .8,
            4: .7,
        }[opt['num_options']]

        if opt['num_options'] > 4:
            raise('Invalid task_config[\'num_options\'] = ' + str(opt['num_options']))
        self.options = ['A', 'B', 'C', 'D'][:opt['num_options']]
        self.debate_mode_to_option = {'Ⅰ': 'A', 'Ⅱ': 'B', 'Ⅲ': 'C', 'Ⅳ': 'D', 'ⅰ': 'A', 'ⅱ': 'B', 'ⅲ': 'C', 'ⅳ': 'D', None: None}

        self.dream_speaker_to_name = {
            'M': 'Man',
            'M1': 'Man 1',
            'M2': 'Man 2',
            'F': 'Woman',
            'F1': 'Woman 1',
            'F2': 'Woman 2',
            'W': 'Woman',
            'W1': 'Woman 1',
            'W2': 'Woman 2',
            'A': 'Speaker A',
            'B': 'Speaker B',
        }
        if 'passage and question' in self.prompt_types:
            for option in self.options:  # Avoid replacing option name with speaker name
                if option in self.dream_speaker_to_name:
                    self.dream_speaker_to_name.pop(option)

        random.seed(0)
        self.sample_debate_modes = None
        if evaluation_data:
            self.num_changed_responses = 0
            self.num_debate_mode_responses = 0
            self.possible_debate_modes = list(evaluation_data.keys())
            self.possible_debate_modes.sort()
            if 'quote and question' in self.prompt_types:
                self.sample_debate_modes = [self.possible_debate_modes[random.randint(0, len(self.possible_debate_modes) - 1) - self.option_split_no]
                                            for _ in range(self.max_collected)]
                print(self.mturk_agent.worker_id, '| DEBATE MODES:', self.sample_debate_modes)

        self.num_test_turns = 0  # 2
        self.test_turns = []
        self.test_questions = {}
        if self.num_test_turns == 2:
            self.test_turns = [1, self.max_collected + self.num_test_turns]
            self.test_questions = {
                1: {
                    'text': '"We taught you this just last week!"\n\n' +
                            'Based on the passage, what does the student not know that the teacher expects him to know for the exam?\n\n' +
                            'A. 13 + 12 = 35\n' +
                            'B. 15 / 5 = 4\n' +
                            'C. 41 - 22 = 18\n' +
                            'D. 6 x 4 = 24',
                    'answer': 'D'
                },
                self.max_collected + self.num_test_turns: {
                    'text': '"Wow, I never knew a banana could be that color."\n\n' +
                            'When Fred opens his pantry, he is surprised the banana is not colored _.\n\n' +
                            'A. Gray-ish blue\n' +
                            'B. Purple and pink\n' +
                            'C. Green or yellow\n' +
                            'D. Plain white',
                    'answer': 'C'
                },
            }
        assert self.num_test_turns == len(self.test_turns), 'self.num_test_turns != len(self.test_turns)'

        print(self.mturk_agent.worker_id,
              "| question_split:", self.question_split_no,
              "| option_split:", self.option_split_no,
              '| assignment_id:', self.mturk_agent.assignment_id)

    def parley(self):
        self.cur_example_no = self.num_collected + self.num_tested + 1

        # Verify work quality with a test question
        if self.cur_example_no in self.test_questions.keys():
            test_question = self.test_questions[self.cur_example_no]
            response = self.prompt_and_receive_response(test_question['text'], 'quote and question', None)
            if test_question['answer'] != response:
                reason = 'Test failed: Example ' + str(self.cur_example_no) + ' - Answered ' + response + ' not ' + \
                         (test_question['answer'] if test_question['answer'] is not None else test_question['text'])
                # self.reject_reasons.append(reason)
            self.num_tested += 1
            return
        elif self.cur_example_no > (self.max_collected + self.num_test_turns):
            if any(['quote' in prompt_type for prompt_type in self.prompt_types]):
                # Get quote rating
                self.quote_rating, quote_rating_duration = self.get_response_and_duration({
                    'episode_done': False,
                    'id': 'System',
                    'text': 'All done! How useful were the provided passage quotes in answering questions?',
                    'task_data': {'respond_with_form': [{
                        'type': 'choices',
                        'question': 'On a scale of 0-10',
                        'choices': [i for i in range(0, 11)]
                    }]}
                })
                if self.quote_rating is None:
                    return

                # Get quote description
                self.quote_description, quote_description_duration = self.get_response_and_duration({
                    'episode_done': False,
                    'id': 'System',
                    'text': 'How would you describe the provided passage quotes?',
                    'task_data': {"respond_with_form": None},
                })
                if self.quote_description is None:
                    return

                print(self.mturk_agent.worker_id,
                      '| quote_rating:', self.quote_rating,
                      '| quote_description:', self.quote_description)

            outcome_text = 'Thanks!'
            for prompt_type, num_correct_for_prompt_type in self.num_correct_on_labeled.items():
                # Show accuracy
                prompt_type_accuracy = int(round((100. * num_correct_for_prompt_type) /
                                                 self.num_collected_on_labeled[prompt_type]))
                outcome_text += ' You got ' + str(prompt_type_accuracy) + '% right with ' + self.prompt_type_description[prompt_type] + '.'
            self.mturk_agent.observe({
                'episode_done': False,
                'id': 'System',
                'text': outcome_text,
                'task_data': {"respond_with_form": None}
            })

            # Net Promoter Score
            self.task_rating, task_rating_duration = self.get_response_and_duration({
                'episode_done': False,
                'id': 'System',
                'text': 'How likely are you to recommend this task to a colleague?',
                'task_data': {'respond_with_form': [{
                    'type': 'choices',
                    'question': 'On a scale of 0-10',
                    'choices': [i for i in range(0, 11)]
                }]}
            })
            if self.task_rating is None:
                return

            # Solicit free-form text feedback
            self.feedback, feedback_duration = self.get_response_and_duration({
                'episode_done': False,
                'id': 'System',
                'text': 'How can we improve this task?',
                'task_data': {"respond_with_form": None},
            })
            if self.feedback is None:
                return

            print(self.mturk_agent.worker_id,
                  '| task_rating:', self.task_rating,
                  '| feedback:', self.feedback)

            # Conclude HIT and send final message
            self.hit_done = True
            self.episodeDone = True
            self.mturk_agent.observe({
                'episode_done': True,
                'id': 'System',
                'text': 'Thanks for your help!',
            })
            return
        else:
            # Get prompt text from dataset teacher agent
            sample = self.task.act()
            sample['debate_mode'] = self.sample_debate_modes[self.num_collected] if (self.sample_debate_modes is not None) else None

            for prompt_type in self.prompt_types:
                if prompt_type == 'question':
                    prompt_text = '\n'.join([sample['question'] + '\n'] + sample['options'])
                    question_response = self.prompt_and_receive_response(prompt_text, prompt_type, sample)
                    if question_response is None:
                        return
                elif prompt_type == 'question and quotes':
                    prompt_text = sample['question']
                    sample['sentences_chosen'] = []
                    for i, debate_mode in enumerate(self.possible_debate_modes):
                        evaluation_sample = self.evaluation_data[debate_mode][sample['qid']]
                        sentences_chosen = [evaluation_sample['sentences_chosen'][0]]  # NB: Always first sentence only
                        self._format_sentences(sentences_chosen)
                        sentences_chosen = '\n'.join(sentences_chosen)
                        prompt_text += '\n'
                        prompt_text += '\nQuote: “' + sentences_chosen + '”'
                        prompt_text += '\n' + sample['options'][i]
                        sample['sentences_chosen'].append(sentences_chosen)
                    sample['sentences_chosen'] = '\n'.join(sample['sentences_chosen'])

                    question_and_quotes_response = self.prompt_and_receive_response(
                        prompt_text, prompt_type, sample)
                    if question_and_quotes_response is None:
                        return
                elif prompt_type == 'quotes and question':
                    no_spaces_passage = sample['passage'].replace(' ', '').replace('\n', '')
                    sentence_chosen_to_passage_index = {}
                    for i, debate_mode in enumerate(self.possible_debate_modes):
                        evaluation_sample = self.evaluation_data[debate_mode][sample['qid']]
                        sentences_chosen = [evaluation_sample['sentences_chosen'][0]]  # NB: Always first sentence only
                        for sentence_chosen in sentences_chosen:
                            no_spaces_sentence_chosen = sentence_chosen.replace(' ', '').replace('\n', '')
                            if no_spaces_sentence_chosen in no_spaces_passage:
                                passage_index = no_spaces_passage.index(no_spaces_sentence_chosen)
                            else:
                                passage_index = float('inf')
                                print('Couldn\'t find sentence:', sentence_chosen, '\nin passage:', sample['passage'])
                            sentence_chosen_to_passage_index[self._format_sentences([sentence_chosen])[0]] = passage_index
                    sorted_sentences_chosen = sorted(sentence_chosen_to_passage_index,
                                                     key=sentence_chosen_to_passage_index.get)
                    sample['sentences_chosen'] = ' ... '.join(sorted_sentences_chosen)

                    prompt_text = '“' + sample['sentences_chosen'] + '”\n\n' + '\n'.join([sample['question'] + '\n'] + sample['options'])
                    quotes_and_question_response = self.prompt_and_receive_response(
                        prompt_text, prompt_type, sample)
                    if quotes_and_question_response is None:
                        return
                elif prompt_type == 'quote and question':
                    # Get sentences chosen
                    evaluation_sample = self.evaluation_data[sample['debate_mode']][sample['qid']]
                    sentences_chosen = [evaluation_sample['sentences_chosen'][0]]  # NB: Always first agent only
                    self._format_sentences(sentences_chosen)

                    # Format prompt
                    sentences_chosen = '\n'.join(sentences_chosen)
                    prompt_text = '\n'.join([sample['question'] + '\n'] + sample['options'])
                    prompt_text = sentences_chosen + '\n\n' + prompt_text
                    sample['sentences_chosen'] = sentences_chosen

                    quote_and_question_response = self.prompt_and_receive_response(prompt_text, prompt_type, sample)
                    if quote_and_question_response is None:
                        return
                    if 'question' in self.prompt_types:
                        self.num_changed_responses += (quote_and_question_response != question_response)
                    if sample['debate_mode'] is not None:
                        self.num_debate_mode_responses += (quote_and_question_response ==
                                                           self.debate_mode_to_option[sample['debate_mode']])
                elif prompt_type == 'passage and question':
                    passage_text = sample['text'].split('\n')
                    prompt_text = '\n' + '\n'.join(self._format_sentences(passage_text))
                    passage_and_question_response = self.prompt_and_receive_response(
                        prompt_text, prompt_type, sample)
                    if passage_and_question_response is None:
                        return
                else:
                    raise NotImplementedError('Prompt Type:', prompt_type)

            # Terminate episode (if applicable)
            self.num_collected += 1
            return

    def prompt_and_receive_response(self, prompt_text, prompt_type, sample=None):
        # Clear previous answer from form. Emphasize questions are unrelated.
        self.mturk_agent.observe({
            'episode_done': False,
            'id': 'New ' + prompt_type,
            'text': None,
            'task_data': {"respond_with_form": None},
        })

        # Data collection prompt
        response, duration = self.get_response_and_duration({
            'episode_done': False,
            'id': '(#' + str(self.cur_example_no) + ')',
            'text': prompt_text,
            'task_data': {"respond_with_form": [{
                "type": "choices",
                "question": "Which option is most likely correct?",
                "choices": self.options
            }]}
        })
        if response is None:
            return

        if sample is not None:
            # Evaluate work on non-qualifying questions
            if 'eval_labels' in sample:
                is_correct = (response == sample['eval_labels'][0])
                if prompt_type in {'question and quotes', 'quotes and question', 'passage and question'}:
                    if is_correct:
                        self.mturk_agent.observe({
                            'episode_done': False,
                            'id': 'System',
                            'text': 'Correct!',
                        })
                    else:
                        correct_answer_text = 'The correct answer was ' + sample['eval_labels'][0] + '.'
                        if any(['quote' in prompt_type for prompt_type in self.prompt_types]):
                            correct_answer_text += ' However, it may have been tough to know from our quotes.'
                        correct_answer_text += ' Feel free to review the previous question and continue when you are ready.'
                        answer_feedback_response, answer_feedback_duration = self.get_response_and_duration({
                            'episode_done': False,
                            'id': 'System',
                            'text': correct_answer_text,
                            'task_data': {'respond_with_form': [{
                                'type': 'choices',
                                'question': 'Ready to continue?',
                                'choices': ['Yes']
                            }]}
                        })
                        if answer_feedback_response is None:
                            return
                self.num_correct_on_labeled[prompt_type] = self.num_correct_on_labeled.get(prompt_type, 0)
                self.num_correct_on_labeled[prompt_type] += is_correct
                self.num_collected_on_labeled[prompt_type] = self.num_collected_on_labeled.get(prompt_type, 0)
                self.num_collected_on_labeled[prompt_type] += 1
                self.accuracy[prompt_type] = self.num_correct_on_labeled[prompt_type] / self.num_collected_on_labeled[prompt_type]

            # Update answer stats and return
            self.durations[prompt_type] = self.durations.get(prompt_type, [])
            self.durations[prompt_type].append(duration)
            self.answer_to_count_by_prompt[prompt_type] = self.answer_to_count_by_prompt.get(prompt_type, {option: 0 for option in self.options})
            self.answer_to_count_by_prompt[prompt_type][response] += 1
            self.data.append({
                'sample': sample,
                'context': prompt_text,
                'response': response,
                'duration': duration,
            })

        print(self.mturk_agent.worker_id,
              '| prompt_type:', prompt_type,
              '| response:', response,
              '| debate_mode:', self.debate_mode_to_option[sample['debate_mode']] if sample is not None else 'TEST',
              '| answer:', sample['eval_labels'][0] if ((sample is not None) and ('eval_labels' in sample)) else 'TEST',
              '| duration:', round(duration / 1000., 1),
              '| qid:', sample['qid'] if sample is not None else 'TEST',
              '' if (sample is None) or ('eval_labels' not in sample) else
              '| accuracy: ' + str(self.num_correct_on_labeled[prompt_type]) + '/' + str(self.num_collected_on_labeled[prompt_type]))
        return response

    def get_response_and_duration(self, ad):
        # Check for required ad data
        assert 'task_data' in ad, "Ill-formed ad: 'task_data' not in ad"
        assert 'respond_with_form' in ad['task_data'], "Ill-formed ad: 'respond_with_form' not in ad['task_data']"

        # Serve ad and receive response
        self.mturk_agent.observe(ad)
        response_message = self.mturk_agent.act()

        # Check for disconnect, return, etc.
        if 'task_data' not in response_message:
            print(self.mturk_agent.worker_id, '| DISCONNECT:', response_message)
            self.episodeDone = True
            return None, None

        # Return string response
        if ad['task_data']['respond_with_form'] is None:  # Text field response
            return response_message['text'], response_message['duration']
        else:  # Form response
            return response_message['task_data']['form_responses'][0]['response'], response_message['duration']

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.task.shutdown()
        self.mturk_agent.shutdown()

    def review_work(self):
        if not self.hit_done:  # Don't review work if agent disconnected
            print(self.mturk_agent.worker_id, 'Done! (Disconnected) | num_debate_mode_responses:',
                  self.num_debate_mode_responses, '/', self.num_collected)
            return

        # Can review the work here to accept or reject it
        if (('question' in self.prompt_types) and ('quote and question' in self.prompt_types) and
                (self.num_changed_responses is not None)):
            self.freq_changed_responses = (self.num_changed_responses / self.num_collected)
            if self.freq_changed_responses <= .2:  # Not reading closely
                reason = 'freq_changed_responses = ' + str(self.freq_changed_responses)
                self.reject_reasons.append(reason)
                if self.freq_changed_responses <= .1:  # Spamming
                    self.block_reasons.append(reason)

        # Turker should be spending a minimum amount of time on each question
        median_durations = []
        for prompt_type, durations in self.durations.items():
            durations.sort()
            median_duration = durations[len(durations) // 2]
            median_durations.append(median_duration)
            if median_duration <= self.median_sample_ms_reject_threshold[prompt_type]:
                reason = 'median_duration = ' + str(median_duration)
                self.reject_reasons.append(reason)
                if median_duration <= (self.median_sample_ms_reject_threshold[prompt_type] / 2.):
                    self.block_reasons.append(reason)

        # Turker answer distribution shouldn't be too peaky
        for answer_to_count in self.answer_to_count_by_prompt.values():
            for answer, count in answer_to_count.items():
                freq = count / self.num_collected
                reason = answer + ' freq = ' + str(freq)
                if freq >= self.response_freq_reject_threshold:
                    self.reject_reasons.append(reason)
                    if freq >= ((1 + self.response_freq_reject_threshold) / 2.):
                        self.block_reasons.append(reason)

        # Bonus for above-average accuracy
        for prompt_type, prompt_type_acc in self.accuracy.items():
            if prompt_type_acc >= self.accuracy_bonus_threshold[prompt_type]:
                self.bonus_reasons.append(prompt_type + ' accuracy = ' + str(prompt_type_acc))

        # Bonus for changing your answer based on context
        if (('question' in self.prompt_types) and ('quote and question' in self.prompt_types)
                and (self.num_changed_responses is not None) and (self.freq_changed_responses >= .5)):
            self.bonus_reasons.append('freq_changed_responses = ' + str(self.freq_changed_responses))

        print(self.mturk_agent.worker_id, 'Done! | num_debate_mode_responses:', self.num_debate_mode_responses, '/', self.num_collected,
              '| block_reasons:', self.block_reasons,
              '| reject_reasons:', self.reject_reasons,
              '| bonus_reasons:', self.bonus_reasons)

        if len(self.bonus_reasons) > 0:  # Meeting bonus condition overrides rejection/blocking
            self.mturk_agent.approve_work()
            bonus_amount = round(self.accuracy_bonus_multiplier * self.reward, 2)
            self.mturk_agent.pay_bonus(bonus_amount, 'Great accuracy!')
            print(self.mturk_agent.worker_id, '| BONUS AWARDED')
        elif len(self.block_reasons) > 0:
            self.mturk_agent.reject_work('effort')
            if not self.is_sandbox:
                self.mturk_agent.mturk_manager.soft_block_worker(self.mturk_agent.worker_id)
        elif len(self.reject_reasons) > 0:
            self.mturk_agent.reject_work('effort')
        else:
            self.mturk_agent.approve_work()

    def get_custom_task_data(self):
        # brings important data together for the task, to later be used for
        # creating the dataset. If data requires pickling, put it in a field
        # called 'needs-pickle'.
        return {
            'data': self.data,
            'worker_id': self.mturk_agent.worker_id,
            'assignment_id': self.mturk_agent.assignment_id,
            'quote_rating': self.quote_rating,
            'quote_description': self.quote_description,
            'task_rating': self.task_rating,
            'feedback': self.feedback,
            'reject_reasons': self.reject_reasons,
            'block_reasons': self.block_reasons,
            'bonus_reasons': self.bonus_reasons,
            'hit_done': self.hit_done,
            'accuracy': self.accuracy,
            'question_split_no': self.question_split_no,
            'option_split_no': self.option_split_no,
            'freq_changed_responses': self.freq_changed_responses,
        }

    def _format_sentences(self, sentence_list):
        """
        Format and preprocess selected sentences before showing to workers. Modifies original sentence list.
        """
        for i in range(len(sentence_list)):
            if self.dataset == 'dream':
                for speaker, name in self.dream_speaker_to_name.items():
                    if (sentence_list[i].startswith(speaker + ': ')) or (sentence_list[i].startswith(speaker + ' : ')):
                        sentence_list[i] = sentence_list[i].replace(speaker, name, 1)
                        break
            for punct in {'.', '?', '!', ';', ',', '\'', ':', 'n\'t', '%', ')'}:
                sentence_list[i] = sentence_list[i].replace(' ' + punct, punct)
            for punct in {'$', '('}:
                sentence_list[i] = sentence_list[i].replace(punct + ' ', punct)
        return sentence_list
