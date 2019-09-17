#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld
import math
import time


class ContextEvaluationOnboardWorld(MTurkOnboardWorld):
    """Example onboarding world. Sends a message from the world to the
    worker and then exits as complete
    """
    def __init__(self, opt, mturk_agent):
        # MTurkOnboardWorld init
        self.mturk_agent = mturk_agent
        print(self.mturk_agent.worker_id, '| INIT ONBOARD WORLD')
        self.episodeDone = False

        self.passed_test = None
        self.cur_example_no = 1
        self.num_incorrect = 0
        self.options = ['A', 'B', 'C', 'D']  # Always use all 4 answer-options for practice questions.
        self.prompt_types = [opt['prompt_type']]
        assert len(self.prompt_types) == 1, 'Using multiple prompt_types not yet supported.'

        self.wrong_threshold = {
            'question': .25,
            'quote and question': .25,
            'question and quotes': .25,
            'quotes and question': 0.,
            'passage and question': 0.,
        }[self.prompt_types[0]]
        self.test_questions = {
            'question': [
                {
                    'text': 'When Fred opens his pantry, he is surprised the banana is not colored _.\n\n' +
                            'A. Gray-ish blue\n' +
                            'B. Purple and pink\n' +
                            'C. Green or yellow\n' +
                            'D. Plain white',
                    'answer': 'C',
                },
                {
                    'text': 'He who considers himself to be better and more important than others is likely to _.\n\n' +
                            'A. have his head in the clouds\n' +
                            'B. be easy to deal with\n' +
                            'C. have "common sense"\n' +
                            'D. have a "big head"',
                    'answer': 'D',
                },
                {
                    'text': 'What does Alan\'s grandfather do every Sunday?\n\n' +
                            'A. He hosts crazy parties.\n' +
                            'B. He studies for the medical school entrance exam.\n' +
                            'C. He flies to Hawaii and back.\n' +
                            'D. He goes to church with his wife.',
                    'answer': 'D',
                },
            ],
            'quote and question': [
                {
                    'text': '"Wow, I never knew a banana could be that color."\n\n' +
                            'When Fred opens his pantry, he is surprised the banana is not colored _.\n\n' +
                            'A. Gray-ish blue\n' +
                            'B. Purple and pink\n' +
                            'C. Green or yellow\n' +
                            'D. Plain white',
                    'answer': 'C',
                },
                {
                    'text': 'The film Schindler\'s List also takes place during World War Two.\n\n' +
                            'What\'s the similarity between Saving Private Ryan and Schindler\'s List?\n\n' +
                            'A. They are both humorous.\n' +
                            'B. They were released at the same time.\n' +
                            'C. They are both American movies.\n' +
                            'D. They both happen during World War Two.',
                    'answer': 'D',
                },
                {
                    'text': 'They are like sheep being led to the slaughterhouse.\n\n'
                            'The main idea of this passage is that _ .\n\n' +
                            'A. Farm animals suffer gruesome deaths.\n' +
                            'B. In every school there is a "top" crowd that sets the pace.\n' +
                            'C. At one time or another you probably did something you knew to be wrong.\n'
                            'D. It is a mistake to follow the "top" crowd blindly.',
                    'answer': 'D',
                },
            ],
            'question and quotes': [
                {
                    'text': """
Which is TRUE about LIFE WATERR?

Quote: “* If you are taking any special medication or have stomach problems, please check with the doctor before buying LIFE WATERR.”
A: It can't be sold without a doctor.

Quote: “You need LIFE WATERR when you feel thirsty after working in the office a long time.”
B: It's also good for stomach problems.

Quote: “For only a little money, you will feel great!”
C: It's not expensive.

Quote: “It\'s purified H2O straight from the Pacific Ocean.”
D: It's made from spring water in the mountains.""",
                    'answer': 'C',
                    'explanation': """
The quote for C provides direct evidence supporting it\'s answer option. Other quotes either only provide indirect support or contradict their associated answer.

Just to give you an idea, the quotes were from this passage:

You need LIFE WATERR when you feel thirsty after working in the office a long time.\nIt\'s purified H2O straight from the Pacific Ocean.\nFor only a little money, you will feel great again!\nGet LIFE WATERR at the stores near your house NOW!\n* If you are taking any special medication or have stomach problems, please check with the doctor before buying LIFE WATERR.
                    """,
                },
                {
                    'text': """
What does the man prefer to work for?

Quote: “Woman: How do you like your new job?”
A: A company of his own.

Quote: “Woman: You see, small businesses have a common problem: only the two or three people who run it can make decisions, and the employees may not be very happy because they can't make decisions.”
B: A small company.

Quote: “Man:   But large companies also have a common problem, so many people are making decisions that sometimes it is a waste of time and money.”
C: A large company.

Quote: “Man:   I like a small company because it's more exciting.”
D: He prefers not to work.""",
                    'answer': 'B',
                    'explanation': """
Here, no quotes support their associated answer. However, answer B's quote supports that the man doesn't like large companies, and answer D's quote supports that the man does like small companies. From this, we can infer that B is correct.   

Just to give you an idea, the quotes were from this passage:

"Woman: How do you like your new job?",
"Man:      I like it very much. This is a nice company to work for.",
"Woman: You worked for a large company before, didn't you?",
"Man:      Yes, I did. But I prefer a small company.",
"Woman: Is it really different?",
"Man:      Oh, yes. It's much different. I like a small company because it's more exciting.",
"Woman: You mean a large company is boring to work for?",
"Man:      No, it's not boring. But a large company has too many people and because it is so big that two or three people couldn't possibly make all the important decisions.",
"Woman: You see, small businesses have a common problem: only the two or three people who run it can make decisions, and the employees may not be very happy because they can't make decisions.",
"Man:      But large companies also have a common problem, so many people are making decisions that sometimes it is a waste of time and money.",
"Woman: Well, I guess there are problems everywhere.",
"Man:      Yeah, but I still prefer working for a small company. It's more interesting and I'll keep more opportunities.\"""",
                },
                {
                    'text': """
Amy Smith is an _ dancer.

Quote: “Nothing is impossible in this world.”
A: Asian

Quote: “Nothing is impossible in this world.”
B: American

Quote: “.”
C: Argentinian

Quote: “Nothing is impossible in this world.”
D: Nigerian
                    """,
                    'answer': 'B',
                    'explanation': """
Since the quotes weren't very helpful, you'd have to guess based on her name only.

Just to give you an idea, the quotes were from this passage:

Amy Smith, a famous dancer from the U.S., had to have her right leg cut after a car accident. She was also cut off on her career road.\nThough the accident brought her bright career to a stop, she didn't give up. In the painful months that followed, Amy met a doctor who developed a man-made leg for her. So strongly, she wanted to go back to dancing. Amy believed in herself and she thought she could realize her dream.\n\nAfter every public recital, she would ask her dad about her performance. \"You still have a long way to go\" was the answer she used to get in return. In January 1984, Amy made a historic comeback by giving a public recital in Los Angeles. She performed in such a great manner that it moved everyone to tears. That evening when she asked her dad the usual question, he didn't say anything. He just touched her feet as a praise. Amy's comeback was so moving that a film producer decided to make the story into a hit film.\n\nWhen someone asked Amy how she had managed to dance again, she said quite simply, \"You don't need feet to dance!\"  Nothing is impossible in this world. If you have the will to win, you can achieve anything.""",
                },
                {
                    'text': """
According to the passage, the key responsibilities include _.

Quote: “* Participate in and contribute to the budget and business planning cycle.”
A: taking charge of production work

Quote: “* Good at day to day leading and coaching.”
B: working on training programs

Quote: “* Conduct market and product research; maintain data base by identifying and gathering marketing information.”
C: maintaining data base of marketing information.

Quote: “* Good at day to day leading and coaching.”
D: serving as a network technician""",
                    'answer': 'C',
                    'explanation': """
Answer C's quote provides direct evidence supporting C. Other quotes only indirectly support their associated answer.

Just to give you an idea, the quotes were from this passage:

Key responsibilities:\n* Manage the whole marketing activities, i.e. brand building, market research and integrated-marketing functions.\n*  Develop and evaluate brand activities including the development of promotional activities, advertising and merchandising.\n*  Obtain market share by developing marketing plans and programs for key brands.\n*  Conduct market and product research; maintain data base by identifying and gathering marketing information.\n*  Understand market/competitor intelligence and cooperate with the sales teams in developing the appropriate marketing strategies.\n*  Keep contacts and exchange of information with regional operations on marketing issues.\n*  Participate in and contribute to the budget and business planning cycle.\n*  Supervise the project to establish company websites.\n* Complete marketing department operational requirements by scheduling and assigning employees; develop, maintain, evaluate and lead the marketing team of pan-China.\n*  Serve as a member of the senior management team providing input and direction on the company's strategic and operational goals and objects.\nRequirements:\n*  University degree or above, MBA is a plus.\n*  At least Bi-lingual: Chinese and English, any other language is a plus.\n*  Strong wits and oral communication skills; analytic skill; active listening.\n*  Good at day to day leading and coaching.\n*  More than 10 years working experience in sales and marketing of _ industry, including at least 5 years management experience; professional in marketing function.\nEmployer introduction:\nSummergate was established in 1999 to import, distribute and market some of the world's best wines to the Chinese market. Today Summergate represents more than 60 wineries from 12 countries around the world.\nWith offices in Beijing, Shanghai; Shenzhen, Guangzhou, Macau and now Hong Kong, Summergate services the entire China market. We distribute and market our brands to all the major food and beverage operators in China, establishing solid business partnerships with national hotel groups as well as all China retail chains and fine dining western and Chinese restaurants.""",
                },
                {
                    'text': """
Why did the Prior complain about the delay?

Quote: “The creative artist needs time for contemplation; he may be busiest when his hands are idlest.”
A: Because he knew that genius might be busiest when seemingly idlest.

Quote: “Leonardo was slightly unhappy and explained to somebody else that there is a great difference between the work of the creative artist and the stonemason.”
B: Because he liked the work of a stonemason.

Quote: “But he would look no further; if none came his way, he would be satisfied to take Prior as a model for Judas.”
C: Because he was eager to be taken as a model for Judas.

Quote: “This inactivity aroused the anger of the fussy Prior, the head of the church, who belonged to the large group of those who believed that the busier a man seems, the more he accomplishes; and so he tried to find fault with the idle painter.”
D: Because he thought that the painter idled most of the hours.""",
                    'answer': 'D',
                    'explanation': """
Here, all quotes are related to their respective answers. However, answer D and its quote respond to the question in the most plausible manner. 

Just to give you an idea, the quotes were from this passage:

The Last Supper is regarded as one of the supreme masterpieces in the whole field of pictorial art. Tradition has it that Leonardo Da Vinci worked for ten years upon the painting, the monks in the church annoyed at the delay. It was said that Leonardo often painted continuously from dawn to night without eating his meals. But at other times he spent hours before the picture, lost in contemplation, examining, comparing, and measuring his figures.\n\nThis inactivity aroused the anger of the fussy Prior, the head of the church, who belonged to the large group of those who believed that the busier a man seems, the more he accomplishes; and so he tried to find fault with the idle  painter. Leonardo was slightly unhappy and explained to somebody else that there is a great difference between the work of the creative artist and the stonemason . The creative artist needs time for contemplation; he may be busiest when his hands are idlest. Just now he needed two heads to complete the picture: that of Christ, for which no model on earth could be found, for where was the man to be found whose face would express the strength, and beauty, and tenderness, and deep sorrow of the Christ; then he also needed a head of Judas, and that was hard to find as well, for where was the man whose face could express the meanness of that base traitor . But he would look no further; if none came his way, he would be satisfied to take Prior as a model for Judas. This threat silenced the angry Prior, who quite naturally had no desire to pass to descendants in such a fashion.""",
                },
            ],
            'quotes and question': [
                {
                    'text': """
“You need LIFE WATERR when you feel thirsty after working in the office a long time. ... It\'s purified H2O straight from the Pacific Ocean. ... For only a little money, you will feel great! ... * If you are taking any special medication or have stomach problems, please check with the doctor before buying LIFE WATERR.”

Which is TRUE about LIFE WATERR?

A: It can't be sold without a doctor.
B: It's also good for stomach problems.
C: It's not expensive.
D: It's made from spring water in the mountains.""",
                    'answer': 'C',
                    'explanation': """
The quoted passage states that "For only a little money, you will feel great again!", which directly supports C. The quotes only indirectly support other answers.

Just to give you an idea, the quotes were from this passage:

You need LIFE WATERR when you feel thirsty after working in the office a long time.\nIt\'s purified H2O straight from the Pacific Ocean.\nFor only a little money, you will feel great again!\nGet LIFE WATERR at the stores near your house NOW!\n* If you are taking any special medication or have stomach problems, please check with the doctor before buying LIFE WATERR.""",
                },
                {
                    'text': """
“Woman: How do you like your new job? ... Man:   I like a small company because it's more exciting. ... Woman: You see, small businesses have a common problem: only the two or three people who run it can make decisions, and the employees may not be very happy because they can't make decisions. ... Man:   But large companies also have a common problem, so many people are making decisions that sometimes it is a waste of time and money.”

What does the man prefer to work for?

A: A company of his own.
B: A small company.
C: A large company.
D: He prefers not to work.""",
                    'answer': 'B',
                    'explanation': """
B is correct, as the quoted sentences support that the man likes working at a small company and not at a large company.   

Just to give you an idea, the quotes were from this passage:

"Woman: How do you like your new job?",
"Man:      I like it very much. This is a nice company to work for.",
"Woman: You worked for a large company before, didn't you?",
"Man:      Yes, I did. But I prefer a small company.",
"Woman: Is it really different?",
"Man:      Oh, yes. It's much different. I like a small company because it's more exciting.",
"Woman: You mean a large company is boring to work for?",
"Man:      No, it's not boring. But a large company has too many people and because it is so big that two or three people couldn't possibly make all the important decisions.",
"Woman: You see, small businesses have a common problem: only the two or three people who run it can make decisions, and the employees may not be very happy because they can't make decisions.",
"Man:      But large companies also have a common problem, so many people are making decisions that sometimes it is a waste of time and money.",
"Woman: Well, I guess there are problems everywhere.",
"Man:      Yeah, but I still prefer working for a small company. It's more interesting and I'll keep more opportunities.\"""",
                },
                {
                    'text': """
“. ... Nothing is impossible in this world. ... ? ... .”

Amy Smith is an _ dancer.

A: Asian
B: American
C: Argentinian
D: Nigerian
                    """,
                    'answer': 'B',
                    'explanation': """
Since the quotes weren't very helpful, you'd have to guess based on her name only.

Just to give you an idea, the quotes were from this passage:

Amy Smith, a famous dancer from the U.S., had to have her right leg cut after a car accident. She was also cut off on her career road.\nThough the accident brought her bright career to a stop, she didn't give up. In the painful months that followed, Amy met a doctor who developed a man-made leg for her. So strongly, she wanted to go back to dancing. Amy believed in herself and she thought she could realize her dream.\n\nAfter every public recital, she would ask her dad about her performance. \"You still have a long way to go\" was the answer she used to get in return. In January 1984, Amy made a historic comeback by giving a public recital in Los Angeles. She performed in such a great manner that it moved everyone to tears. That evening when she asked her dad the usual question, he didn't say anything. He just touched her feet as a praise. Amy's comeback was so moving that a film producer decided to make the story into a hit film.\n\nWhen someone asked Amy how she had managed to dance again, she said quite simply, \"You don't need feet to dance!\"  Nothing is impossible in this world. If you have the will to win, you can achieve anything.""",
                },
                {
                    'text': """
“* Conduct market and product research; maintain data base by identifying and gathering marketing information. ... * Participate in and contribute to the budget and business planning cycle. ... * Good at day to day leading and coaching.”

According to the passage, the key responsibilities include _.

A: taking charge of production work
B: working on training programs
C: maintaining data base of marketing information.
D: serving as a network technician""",
                    'answer': 'C',
                    'explanation': """
The role lists a responsibility to "maintain data base by identifying and gathering marketing information," which directly supports answer C. The quotes only support other answers indirectly.

Just to give you an idea, the quotes were from this passage:

Key responsibilities:\n* Manage the whole marketing activities, i.e. brand building, market research and integrated-marketing functions.\n*  Develop and evaluate brand activities including the development of promotional activities, advertising and merchandising.\n*  Obtain market share by developing marketing plans and programs for key brands.\n*  Conduct market and product research; maintain data base by identifying and gathering marketing information.\n*  Understand market/competitor intelligence and cooperate with the sales teams in developing the appropriate marketing strategies.\n*  Keep contacts and exchange of information with regional operations on marketing issues.\n*  Participate in and contribute to the budget and business planning cycle.\n*  Supervise the project to establish company websites.\n* Complete marketing department operational requirements by scheduling and assigning employees; develop, maintain, evaluate and lead the marketing team of pan-China.\n*  Serve as a member of the senior management team providing input and direction on the company's strategic and operational goals and objects.\nRequirements:\n*  University degree or above, MBA is a plus.\n*  At least Bi-lingual: Chinese and English, any other language is a plus.\n*  Strong wits and oral communication skills; analytic skill; active listening.\n*  Good at day to day leading and coaching.\n*  More than 10 years working experience in sales and marketing of _ industry, including at least 5 years management experience; professional in marketing function.\nEmployer introduction:\nSummergate was established in 1999 to import, distribute and market some of the world's best wines to the Chinese market. Today Summergate represents more than 60 wineries from 12 countries around the world.\nWith offices in Beijing, Shanghai; Shenzhen, Guangzhou, Macau and now Hong Kong, Summergate services the entire China market. We distribute and market our brands to all the major food and beverage operators in China, establishing solid business partnerships with national hotel groups as well as all China retail chains and fine dining western and Chinese restaurants.""",
                },
                {
                    'text': """
“This inactivity aroused the anger of the fussy Prior, the head of the church, who belonged to the large group of those who believed that the busier a man seems, the more he accomplishes; and so he tried to find fault with the idle painter. ... Leonardo was slightly unhappy and explained to somebody else that there is a great difference between the work of the creative artist and the stonemason. ... The creative artist needs time for contemplation; he may be busiest when his hands are idlest. ... But he would look no further; if none came his way, he would be satisfied to take Prior as a model for Judas.”

Why did the Prior complain about the delay?

A: Because he knew that genius might be busiest when seemingly idlest.
B: Because he liked the work of a stonemason.
C: Because he was eager to be taken as a model for Judas.
D: Because he thought that the painter idled most of the hours.""",
                    'answer': 'D',
                    'explanation': """
The first quoted sentence most directly answers the question, and it supports answer D.

Just to give you an idea, the quotes were from this passage:

The Last Supper is regarded as one of the supreme masterpieces in the whole field of pictorial art. Tradition has it that Leonardo Da Vinci worked for ten years upon the painting, the monks in the church annoyed at the delay. It was said that Leonardo often painted continuously from dawn to night without eating his meals. But at other times he spent hours before the picture, lost in contemplation, examining, comparing, and measuring his figures.\n\nThis inactivity aroused the anger of the fussy Prior, the head of the church, who belonged to the large group of those who believed that the busier a man seems, the more he accomplishes; and so he tried to find fault with the idle  painter. Leonardo was slightly unhappy and explained to somebody else that there is a great difference between the work of the creative artist and the stonemason . The creative artist needs time for contemplation; he may be busiest when his hands are idlest. Just now he needed two heads to complete the picture: that of Christ, for which no model on earth could be found, for where was the man to be found whose face would express the strength, and beauty, and tenderness, and deep sorrow of the Christ; then he also needed a head of Judas, and that was hard to find as well, for where was the man whose face could express the meanness of that base traitor . But he would look no further; if none came his way, he would be satisfied to take Prior as a model for Judas. This threat silenced the angry Prior, who quite naturally had no desire to pass to descendants in such a fashion.""",
                },
            ],
            'passage and question': [
                {
                    'text': """
You need LIFE WATERR when you feel thirsty after working in the office a long time.\nIt\'s purified H2O straight from the Pacific Ocean.\nFor only a little money, you will feel great again!\nGet LIFE WATERR at the stores near your house NOW!\n* If you are taking any special medication or have stomach problems, please check with the doctor before buying LIFE WATERR.

Which is TRUE about LIFE WATERR?

A: It can't be sold without a doctor.
B: It's also good for stomach problems.
C: It's not expensive.
D: It's made from spring water in the mountains.""",
                    'answer': 'C',
                    'explanation': """The passage states that "For only a little money, you will feel great again!", which directly supports C. The passage only indirectly supports other answers.""",
                },
                {
                    'text': """
"Woman: How do you like your new job?",
"Man:      I like it very much. This is a nice company to work for.",
"Woman: You worked for a large company before, didn't you?",
"Man:      Yes, I did. But I prefer a small company.",
"Woman: Is it really different?",
"Man:      Oh, yes. It's much different. I like a small company because it's more exciting.",
"Woman: You mean a large company is boring to work for?",
"Man:      No, it's not boring. But a large company has too many people and because it is so big that two or three people couldn't possibly make all the important decisions.",
"Woman: You see, small businesses have a common problem: only the two or three people who run it can make decisions, and the employees may not be very happy because they can't make decisions.",
"Man:      But large companies also have a common problem, so many people are making decisions that sometimes it is a waste of time and money.",
"Woman: Well, I guess there are problems everywhere.",
"Man:      Yeah, but I still prefer working for a small company. It's more interesting and I'll keep more opportunities."

What does the man prefer to work for?

A: A company of his own.
B: A small company.
C: A large company.
D: He prefers not to work.""",
                    'answer': 'B',
                    'explanation': """The man states he finds a small company more exciting and that he is frustrated that large companies have too many people.""",
                },
                {
                    'text': """
Key responsibilities:\n* Manage the whole marketing activities, i.e. brand building, market research and integrated-marketing functions.\n*  Develop and evaluate brand activities including the development of promotional activities, advertising and merchandising.\n*  Obtain market share by developing marketing plans and programs for key brands.\n*  Conduct market and product research; maintain data base by identifying and gathering marketing information.\n*  Understand market/competitor intelligence and cooperate with the sales teams in developing the appropriate marketing strategies.\n*  Keep contacts and exchange of information with regional operations on marketing issues.\n*  Participate in and contribute to the budget and business planning cycle.\n*  Supervise the project to establish company websites.\n* Complete marketing department operational requirements by scheduling and assigning employees; develop, maintain, evaluate and lead the marketing team of pan-China.\n*  Serve as a member of the senior management team providing input and direction on the company's strategic and operational goals and objects.\nRequirements:\n*  University degree or above, MBA is a plus.\n*  At least Bi-lingual: Chinese and English, any other language is a plus.\n*  Strong wits and oral communication skills; analytic skill; active listening.\n*  Good at day to day leading and coaching.\n*  More than 10 years working experience in sales and marketing of _ industry, including at least 5 years management experience; professional in marketing function.\nEmployer introduction:\nSummergate was established in 1999 to import, distribute and market some of the world's best wines to the Chinese market. Today Summergate represents more than 60 wineries from 12 countries around the world.\nWith offices in Beijing, Shanghai; Shenzhen, Guangzhou, Macau and now Hong Kong, Summergate services the entire China market. We distribute and market our brands to all the major food and beverage operators in China, establishing solid business partnerships with national hotel groups as well as all China retail chains and fine dining western and Chinese restaurants.

According to the passage, the key responsibilities include _.

A: taking charge of production work
B: working on training programs
C: maintaining data base of marketing information.
D: serving as a network technician""",
                    'answer': 'C',
                    'explanation': """The role lists a responsibility to "maintain data base by identifying and gathering marketing information," which directly supports answer C. Other answers only have indirect support.""",
                },
                {
                    'text': """
The Last Supper is regarded as one of the supreme masterpieces in the whole field of pictorial art. Tradition has it that Leonardo Da Vinci worked for ten years upon the painting, the monks in the church annoyed at the delay. It was said that Leonardo often painted continuously from dawn to night without eating his meals. But at other times he spent hours before the picture, lost in contemplation, examining, comparing, and measuring his figures.\n\nThis inactivity aroused the anger of the fussy Prior, the head of the church, who belonged to the large group of those who believed that the busier a man seems, the more he accomplishes; and so he tried to find fault with the idle  painter. Leonardo was slightly unhappy and explained to somebody else that there is a great difference between the work of the creative artist and the stonemason . The creative artist needs time for contemplation; he may be busiest when his hands are idlest. Just now he needed two heads to complete the picture: that of Christ, for which no model on earth could be found, for where was the man to be found whose face would express the strength, and beauty, and tenderness, and deep sorrow of the Christ; then he also needed a head of Judas, and that was hard to find as well, for where was the man whose face could express the meanness of that base traitor . But he would look no further; if none came his way, he would be satisfied to take Prior as a model for Judas. This threat silenced the angry Prior, who quite naturally had no desire to pass to descendants in such a fashion.

Why did the Prior complain about the delay?

A: Because he knew that genius might be busiest when seemingly idlest.
B: Because he liked the work of a stonemason.
C: Because he was eager to be taken as a model for Judas.
D: Because he thought that the painter idled most of the hours.""",
                    'answer': 'D',
                    'explanation': """The answer is contained in the sentence \"This inactivity aroused the anger of the fussy Prior... who believed that the busier a man seems, the more he accomplishes.\"""",
                },
            ],
        }

        for prompt_type in self.test_questions.keys():
            for i in range(len(self.test_questions[prompt_type])):
                self.test_questions[prompt_type][i]['qid'] = prompt_type + '/' + str(i)
            # random.shuffle(self.test_questions[prompt_type])

    def parley(self):
        prompt_type = self.prompt_types[0]
        num_test = len(self.test_questions[prompt_type])
        max_incorrect = int(math.floor(num_test * self.wrong_threshold))
        self.mturk_agent.observe({
            'episode_done': False,
            'id': 'System',
            'text': 'Welcome onboard! We\'ll first give you ' + str(num_test) +
                    ' practice examples to help you understand the task. ' +
                    'To qualify for the HIT, you\'ll have to answer ' + str(num_test - max_incorrect) + ' correct.',
        })

        if prompt_type in {'question and quotes'}:
            initial_understand_response, initial_understand_duration = self.get_response_and_duration({
                'episode_done': False,
                'id': 'System',
                'text': 'Note: Answer-supporting quotes won\'t always be helpful; sometimes, you\'ll have to guess from the question and answers only. ' +
                        'Other times, the quote may contradict its answer or support a different answer.',
                'task_data': {"respond_with_form": [{
                    "type": "choices",
                    "question": "Does this make sense?",
                    "choices": ['Yes, let\'s see a few examples.', 'Not really, but let\'s see a few examples.']
                }]}
            })
            if initial_understand_response is None:
                return
            print(self.mturk_agent.worker_id, '| Makes sense?', initial_understand_response)

        for test_question in self.test_questions[prompt_type]:
            response = self.prompt_and_receive_response(test_question, prompt_type)
            if response is None:
                return
            elif response != test_question['answer']:
                print(self.mturk_agent.worker_id,
                      '| WRONG', test_question['qid'],
                      '| Answered', response, 'not', test_question['answer'])
                self.num_incorrect += 1
                # Soft block
                if self.num_incorrect > max_incorrect:
                    print(self.mturk_agent.worker_id, '| SOFT BLOCK')
                    self.mturk_agent.mturk_manager.soft_block_worker(self.mturk_agent.worker_id)
                    self.passed_test = False
                # Give feedback
                self.mturk_agent.observe({
                    'episode_done': False,
                    'id': 'System',
                    'text': 'The correct answer was ' + test_question['answer'] + '.',
                })
                # Terminate qualifier if necessary
                if self.num_incorrect > max_incorrect:
                    self.episodeDone = True
                    self.mturk_agent.observe({
                        'episode_done': True,
                        'id': 'System',
                        'text': 'Unfortunately, you did not qualify for our task at this time, but we hope to see you again soon!',
                    })
                    self.mturk_agent.set_hit_is_abandoned()  # NB: May not be the right thing to do
                    return
            else:
                self.mturk_agent.observe({
                    'episode_done': False,
                    'id': 'System',
                    'text': 'Correct!',
                })
            if 'explanation' in test_question:
                explanation_response, explanation_duration = self.get_response_and_duration({
                    'episode_done': False,
                    'id': 'System',
                    'text': test_question['explanation'],
                    'task_data': {"respond_with_form": [{
                        "type": "choices",
                        "question": "Ready to continue?",
                        "choices": ['Yes']
                    }]}
                })
                if explanation_response is None:
                    return

            self.cur_example_no += 1

        continue_choices = ['No, so I\'d rather not do the task.',
                            'Yes, but I\'d rather not do the task.',
                            'Yes, let\'s do the real task!']
        continue_response, continue_duration = self.get_response_and_duration({
            'episode_done': False,
            'id': 'System',
            'text': 'Does the task make sense?',
            'task_data': {"respond_with_form": [{
                "type": "choices",
                "question": "Does the task make sense?",
                "choices": continue_choices
            }]}
        })
        if continue_response is None:
            return
        print(self.mturk_agent.worker_id, '| Continue?', continue_response)
        if continue_response != continue_choices[-1]:
            self.episodeDone = True
            self.mturk_agent.observe({
                'episode_done': True,
                'id': 'System',
                'text': 'No problem. Thanks for considering our HIT, and we hope to see you again soon!',
            })
            self.mturk_agent.set_hit_is_abandoned()  # NB: May not be the right thing to do
            return

        self.passed_test = True
        self.episodeDone = True
        self.mturk_agent.observe({
            'episode_done': True,
            'id': 'System',
            'text': 'Great! Advancing to the real task...',
        })
        time.sleep(3)

    def prompt_and_receive_response(self, test_question, prompt_type):
        prompt_text = test_question['text']
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

        print(self.mturk_agent.worker_id,
              '| prompt_type:', prompt_type,
              '| response:', response,
              '| answer:', test_question['answer'],
              '| duration:', round(duration / 1000., 1),
              '| qid:', test_question['qid'])
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
