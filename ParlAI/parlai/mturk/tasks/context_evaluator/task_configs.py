#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

task_configs = {
    'general': {
        'block_qualification': 'poor performance',
        'count_complete': True,
        'max_time': 5400,
        'max_time_qual': 'max time',
        'frontend_version': 1,
        'hit_title': 'Guess the answer!',  # Passage comprehension [with just quotes / without the passage]
        'hit_keywords': 'reading,question,answer',
        'mturk_agent_id': 'Guesser',
        'question_splits': 5,  # max Q's/passage in 'datatype' field directory
    },
    'race': {
        'datatype': 'test.human_eval',
        'num_conversations': 100,
        'num_options': 4,
        'unique_qual_name': 'race_raw/test.human_eval',
    },
    'dream': {
        'datatype': 'test.num_questions=100',
        'num_conversations': 75,
        'num_options': 3,
        'unique_qual_name': 'dream',
    },
    'live': {
        'allowed_conversations': 1,
        'hobby': True,
        'max_hits_per_worker': 1,
        'unique_worker': True,
    },
    'sandbox': {
        'allowed_conversations': 100,
        'hobby': False,
        'max_hits_per_worker': 100,
    },
    'question': {
        'evaluation_data_dir': None,
        'num_conversations': 25,
        'reward': 1.0,  # RACE: 1.0, 7m, .5 bonus. DREAM: .76, 5m, .37 bonus
        'assignment_duration_in_seconds': 1800,
        'hit_description': 'Can you answer passage comprehension questions without the passage?',
        'task_description': """
            <b>Your Goal</b>: See how well you can answer passage-comprehension exam questions, without the passage - just the question and answer options.
            You\'ll get a bonus if you do well!<br><br>
            
            <b>Our Goal</b>: We\'re trying to evaluate how well people can do on reading comprehension exams without reading the passage. Options can often be eliminated by common sense, general knowledge, or the question/option phrasing; if you read closely, you should do notably better than random guessing.<br><br>
            
            <font color="blue"><b>IMPORTANT</b></font>: Our setup inherently makes some questions nonsensical or impossible to answer. For these questions, just give your best guess! The task is meant to be fun.<br><br>
            
            <b>Questions in HIT</b>: 20<br>
            <b>Time</b>: 7 minutes<br>
            <b>Bonus</b>: $0.50 for exceeding average worker accuracy<br>
            <b>Payout</b>: Immediate<br>
            <b>Qualifying</b>: Must pass 3 trial questions first. We have the right to reject work for workers who pass the qualifier but provide spam on the real HIT.<br><br>
            
            <b>------------------- EXAMPLE -------------------</b> <br><br>
            <b>Question</b>:<br>
            What does the doctor think of Heelys?<br><br>
            
            A: They are too expensive to buy.<br>
            B: They are too fast to go.<br>
            C: They are bad for kids' health.<br>
            D: They are good for training.<br><br>
            
            Which option is most likely correct?<br>
            <b>Guesser</b>: D
        """
    },
    'passage and question': {
        'evaluation_data_dir': None,
        'num_conversations': 25,
        'reward': 4.5,  # DREAM: 1.5, 11m, 1.5, RACE: 4.5, 23-34m, 4.5
        'assignment_duration_in_seconds': 5400,
        'hit_description': 'How well can you answer passage comprehension questions?',
        'task_description': """
            <b>Your Goal</b>: See how well you can answer reading comprehension exam, multiple-choice questions.
            You\'ll be paid double if you do well!<br><br>
            
            <b>Questions in HIT</b>: 20<br>
            <b>Time</b>: 23-34 minutes<br>
            <b>Bonus</b>: $4.5 for exceeding average worker accuracy<br>
            <b>Payout</b>: Immediate<br>
            <b>Qualifying</b>: Must pass 4 trial questions first. We have the right to reject work for workers who pass the qualifier but provide spam on the real HIT.<br><br>
            
            <b>------------------- EXAMPLE -------------------</b> <br><br>
            <b>Passage and Question</b>:<br>
            What's the coolest kind of transportation for middle school students back from winter holidays? A racing bike? A car? No, it's a special kind of shoes called Heelys. Heelys look just like common sports shoes, but they have a wheel hidden in the heel. So instead of walking, kids can \"fly\" around in them. \"Wearing Heelys is fun and cool! \" said Wu Peng, a boy who wore them on his first day back at No. 6 Middle School in Beijing. Wu Peng said he loves the shoes so much that he wears them to go here and there. Sometimes he even follows his parents' car to the supermarket in his Heelys! Other students also think they are very cool, but some aren't so lucky with their Heelys. It's said that some children fell down and got hurt while wearing these shoes. \"Heelys wheels are in the heels of the shoes, so it's easy to fall,\" said Liu Rui, a doctor at the Hong Kong International Medical Clinic, Beijing. Even worse, Liu said, \"Wearing Heelys for a long time could stop young people from developing their legs.\"<br><br>
            
            What does the doctor think of Heelys?<br><br>
            
            A: They are too expensive to buy.<br>
            B: They are too fast to go.<br>
            C: They are bad for kids' health.<br>
            D: They are good for training.<br><br>
            
            <b>Which option is most likely correct?</b><br>
            <b>Guesser</b>: C
        """
    },
    'question and quotes': {
        'evaluation_data_dir': '../allennlp/eval/dream/sl/test',
        'num_conversations': 25,
        'reward': 1.5,  # RACE: 2.0, 16m, 2.0 bonus. DREAM: 1.5, 11m, 1.5 bonus
        'assignment_duration_in_seconds': 5400,
        'hit_description': 'Can you answer passage comprehension questions using just a few quotes?',
        'task_description': """
            <b>Your Goal</b>: See how well you can guess the answers to passage-comprehension exam questions, given just passage quotes. For each possible multiple-choice answer, you\'ll receive one sentence quoted from the passage in defense of that answer.
            You\'ll be paid double if you do well!<br><br>
            
            <b>Our Goal</b>: We\'re evaluating a tool for helping people quickly answer questions about lots of text.<br><br>
            
            <font color="blue"><b>IMPORTANT</b></font>: Our setup inherently makes many questions challenging to answer. For these questions, just give your best guess! The task is meant to be fun.<br><br>
            
            <b>Questions in HIT</b>: 20<br>
            <b>Time</b>: 11 minutes<br>
            <b>Bonus</b>: $1.5 for exceeding average worker accuracy<br>
            <b>Payout</b>: Immediate<br>
            <b>Qualifying</b>: Must pass 5 trial questions first. We have the right to reject work for workers who pass the qualifier but provide spam on the real HIT.<br><br>
            
            <b>------------------- EXAMPLE -------------------</b> <br><br>
            <b>Question and Answer-Supporting Passage Quotes</b>:<br>
            What does the doctor think of Heelys?<br><br>
            
            Quote: “No, it's a special kind of shoes called Heelys.”<br>
            A: They are too expensive to buy.<br><br>
            
            Quote: “It's said that some children fell down and got hurt while wearing these shoes.”<br>
            B: They are too fast to go.<br><br>
            
            Quote: “'Wearing Heelys for a long time could stop young people from developing their legs.'”<br>
            C: They are bad for kids' health.<br><br>
            
            Quote: “Sometimes he even follows his parents' car to the supermarket in his Heelys!”<br>
            D: They are good for training.<br><br>
            
            <b>Which option is most likely correct?</b><br>
            <b>Guesser</b>: C
        """
    },
    'quote and question': {
        'evaluation_data_dir': '../allennlp/eval/dream/human.3/test',
        'reward': 1.12,  # RACE: 1.5, 11m, .75 bonus. DREAM: 1.12, 8m, .56 bonus
        'assignment_duration_in_seconds': 2700,
        'hit_description': 'Can you answer passage comprehension questions using just a quote?',
        'task_description': """
            <b>Your Goal</b>: See how well you can guess the answers to passage-comprehension exam questions, given just a quote from the passage.
            You\'ll get a bonus if you do well!<br><br>
            
            <b>Our Goal</b>: We\'re trying to evaluate how important various passage sentences are for answering each question.<br><br>
            
            <font color="blue"><b>IMPORTANT</b></font>: Our setup inherently makes many questions nonsensical or impossible to answer. For these questions, just give your best guess! The task is meant to be fun.<br><br>
            
            <b>Questions in HIT</b>: 20<br>
            <b>Time</b>: 8 minutes<br>
            <b>Bonus</b>: $0.56 for exceeding average worker accuracy<br>
            <b>Payout</b>: Immediate<br>
            <b>Qualifying</b>: Must pass 3 trial questions first. We have the right to reject work for workers who pass the qualifier but provide spam on the real HIT.<br><br>
            
            <b>------------------- EXAMPLE -------------------</b> <br><br>
            <b>Passage quote and question</b>:<br>
            "Wearing Heelys for a long time could stop young people from developing their legs."<br><br>
            
            What does the doctor think of Heelys?<br><br>
            
            A: They are too expensive to buy.<br>
            B: They are too fast to go.<br>
            C: They are bad for kids' health.<br>
            D: They are good for training.<br><br>
            
            Which option is most likely correct?<br>
            <b>Guesser</b>: C
        """
    },
    'quotes and question': {
        'evaluation_data_dir': '../allennlp/eval/dream/human.3/test',
        'num_conversations': 25,
        'reward': 1.5,  # RACE: 3.1, 21-33m, 3.1 bonus. DREAM: 1.5, 11m, 1.5 bonus
        'assignment_duration_in_seconds': 5400,
        'hit_description': 'Can you answer passage comprehension questions using just a few quotes?',
        'task_description': """
            <b>Your Goal</b>: See how well you can guess the answers to passage-comprehension exam questions, given just passage quotes. For each possible multiple-choice answer, you\'ll receive one sentence quoted from the passage in defense of that answer.
            You\'ll be paid double if you do well!<br><br>
            
            <b>Our Goal</b>: We\'re evaluating a tool for helping people quickly answer questions about lots of text.<br><br>
            
            <font color="blue"><b>IMPORTANT</b></font>: Our setup inherently makes many questions challenging to answer. For these questions, just give your best guess! The task is meant to be fun.<br><br>
            
            <b>Questions in HIT</b>: 20<br>
            <b>Time</b>: 11 minutes<br>
            <b>Bonus</b>: $1.5 for exceeding average worker accuracy<br>
            <b>Payout</b>: Immediate<br>
            <b>Qualifying</b>: Must pass 5 trial questions first. We have the right to reject work for workers who pass the qualifier but provide spam on the real HIT.<br><br>
            
            <b>------------------- EXAMPLE -------------------</b> <br><br>
            <b>Passage Quotes and Question</b>:<br>
            No, it's a special kind of shoes called Heelys. ... Sometimes he even follows his parents' car to the supermarket in his Heelys! ... It's said that some children fell down and got hurt while wearing these shoes. ... 'Wearing Heelys for a long time could stop young people from developing their legs.'<br><br>
            
            What does the doctor think of Heelys?<br><br>
            
            A: They are too expensive to buy.<br>
            B: They are too fast to go.<br>
            C: They are bad for kids' health.<br>
            D: They are good for training.<br><br>
            
            <b>Which option is most likely correct?</b><br>
            <b>Guesser</b>: C
        """
    },
}
