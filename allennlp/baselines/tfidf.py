"""
tfidf.py

TF-IDF Baseline (running as judge - takes debate logs as input, returns persuasiveness accuracy)
"""
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

import argparse
import json
import os
import numpy as np
import tqdm


ANS2IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
DEBATE2IDX = {'Ⅰ': 0, 'Ⅱ': 1, 'Ⅲ': 2, 'Ⅳ': 3}


def parse_args():
    p = argparse.ArgumentParser(description='TF-IDF Judge')
    p.add_argument("-m", "--mode", default='cross-model', help='Mode to run in < judge | cross-model >')
    p.add_argument("-d", "--dataset", default='dream', help='Dataset to run on < race | dream >')

    p.add_argument("-t", "--train", required=True, help='Path to raw train data to compute TF-IDF')
    p.add_argument("-v", "--val", nargs='+', required=True, help='Paths to debate logs for each agent.')
    p.add_argument("-q", "--with_question", default=False, action='store_true', help='TF-IDF with Question + Answer')

    p.add_argument("-p", "--pretrained", default='datasets/bert/uncased_L-12_H-768_A-12/vocab.txt')

    return p.parse_args()


def compute_tf(p_a):
    """Given tensor of [ndoc, words], compute Term Frequence (BoW) Representation [ndoc, voc_sz]"""

    # Compute Vocabulary
    vocab = set()
    print('\nCreating Vocabulary...')
    for doc in tqdm.tqdm(p_a):
        vocab |= set(doc)
    vocab = {w: i for i, w in enumerate(list(vocab))}

    # Compute Count TF Matrix
    tf = np.zeros((len(p_a), len(vocab)), dtype=int)
    print('\nComputing TF Matrix...')
    for i, doc in tqdm.tqdm(enumerate(p_a)):
        for w in doc:
            tf[i][vocab[w]] += 1

    return tf


def race_judge(args, idf, keys):
    """Run and Compute Accuracy on Baseline QA Model"""
    if args.mode == 'judge':
        levels = [os.path.join(args.val[0], x) for x in os.listdir(args.val[0])]
        correct, total = 0, 0
        for level in levels:
            passages = [os.path.join(level, x) for x in os.listdir(level)]
            print('\nRunning Debates for %s...' % level)
            for p in tqdm.tqdm(passages):
                # Get Key Stub
                k, cur_question = os.path.relpath(p, args.val[0]), 0
                while os.path.join(k, str(cur_question)) in keys:
                    key = os.path.join(k, str(cur_question))
                    d = keys[key]

                    # Compute Scores
                    passage_idx = d['passage_idx'][0]
                    opt_idxs = d['option_idx']

                    opt_scores = cosine_similarity(idf[opt_idxs], idf[passage_idx]).flatten()
                    best_opt = np.argmax(opt_scores)

                    # Score
                    if best_opt == d['answer']:
                        correct += 1

                    total += 1
                    cur_question += 1
        print("\nJudge Accuracy: %.5f out of %d Total Examples" % (correct / total, total))

    else:
        correct, total = 0, 0
        for key in keys:
            d = keys[key]

            # Compute Scores
            passage_idx = d['passage_idx'][0]
            opt_idxs = d['option_idx']

            opt_scores = cosine_similarity(idf[opt_idxs], idf[passage_idx]).flatten()
            best_opt = np.argmax(opt_scores)

            # Score
            if best_opt == d['answer']:
                correct += 1

            total += 1

        print("\nPersuasion Accuracy: %.5f out of %d Total Examples" % (correct / total, total))


def dream_judge(args, idf, keys):
    """Run and Compute Accuracy on Baseline QA Model"""
    if args.mode == 'judge':
        # Load and Iterate through Data
        with open(args.val[0], 'rb') as f:
            data = json.load(f)

        correct, total = 0, 0
        for i, article in enumerate(data):
            for idx in range(len(article[1])):
                # Get Key
                key = os.path.join(article[2], str(idx))

                d = keys[key]

                # Compute Scores
                passage_idx = d['passage_idx'][0]
                opt_idxs = d['option_idx']

                opt_scores = cosine_similarity(idf[opt_idxs], idf[passage_idx]).flatten()
                best_opt = np.argmax(opt_scores)

                # Score
                if best_opt == d['answer']:
                    correct += 1

                total += 1

        print("\nJudge Accuracy: %.5f out of %d Total Examples" % (correct / total, total))
    else:
        correct, total = 0, 0
        for key in keys:
            d = keys[key]

            # Compute Scores
            passage_idx = d['passage_idx'][0]
            opt_idxs = d['option_idx']

            opt_scores = cosine_similarity(idf[opt_idxs], idf[passage_idx]).flatten()
            best_opt = np.argmax(opt_scores)

            # Score
            if best_opt == d['answer']:
                correct += 1
            total += 1

        print("\nPersuasion Accuracy: %.5f out of %d Total Examples" % (correct / total, total))


def parse_race_data(args, tokenizer):
    # Create Tracking Variables
    keys, p_a = {}, []

    if args.mode == 'judge':
        # Iterate through Data
        for dtype in [args.train, args.val[0]]:
            levels = [os.path.join(dtype, x) for x in os.listdir(dtype)]
            for level in levels:
                passages = [os.path.join(level, x) for x in os.listdir(level)]

                print('\nProcessing %s...' % level)
                for p in tqdm.tqdm(passages):
                    # Get Key Stub
                    k = os.path.relpath(p, dtype)

                    # Read File
                    with open(p, 'rb') as f:
                        data = json.load(f)

                    # Create State Variables
                    passage_idx = []

                    # Tokenize Passage => Split into Sentences, then Tokenize each Sentence
                    context = data['article']

                    # Tokenize and Add to P_A
                    tokens = tokenizer.tokenize(context)
                    passage_idx.append(len(p_a))
                    p_a.append(tokens)

                    # Iterate through each Question
                    for idx in range(len(data['questions'])):
                        # Create Specific Example Key
                        key = os.path.join(k, str(idx))

                        # Fetch
                        q, ans, options = data['questions'][idx], ANS2IDX[data['answers'][idx]], data['options'][idx]

                        # Create State Variables
                        option_idx = []

                        # Tokenize Options (Q + Option if specified) and Add to P_A
                        for o_idx in range(len(options)):
                            if args.with_question:
                                option = q + " " + options[o_idx]
                            else:
                                option = options[o_idx]

                            option_tokens = tokenizer.tokenize(option)
                            option_idx.append(len(p_a))
                            p_a.append(option_tokens)

                        # Create Dictionary Entry
                        keys[key] = {'passage_idx': passage_idx, 'question': q, 'answer': ans, 'options': options,
                                     'option_idx': option_idx}
        return keys, p_a
    else:
        # Iterate through Train Data First!
        levels = [os.path.join(args.train, x) for x in os.listdir(args.train)]
        for level in levels:
            passages = [os.path.join(level, x) for x in os.listdir(level)]

            print('\nProcessing %s...' % level)
            for p in tqdm.tqdm(passages):
                # Get Key Stub
                k = os.path.relpath(p, args.train)

                # Read File
                with open(p, 'rb') as f:
                    data = json.load(f)

                # Create State Variables
                passage_idx = []

                # Tokenize Passage
                context = data['article']

                # Tokenize and Add to P_A
                tokens = tokenizer.tokenize(context)
                passage_idx.append(len(p_a))
                p_a.append(tokens)

                # Iterate through each Question
                for idx in range(len(data['questions'])):
                    # Fetch
                    q, ans, options = data['questions'][idx], ANS2IDX[data['answers'][idx]], data['options'][idx]

                    # Create State Variables
                    option_idx = []

                    # Tokenize Options (Q + Option if specified) and Add to P_A
                    for o_idx in range(len(options)):
                        if args.with_question:
                            option = q + " " + options[o_idx]
                        else:
                            option = options[o_idx]

                        option_tokens = tokenizer.tokenize(option)
                        option_idx.append(len(p_a))
                        p_a.append(option_tokens)

        # Iterate through all Validation Debate Logs
        for deb_mode, val in enumerate(args.val):
            with open(val, 'rb') as f:
                logs = json.load(f)

            for key in logs:
                # Fetch Data
                data = logs[key]

                # Create State Variables
                passage_idx = []

                # Tokenize Passage
                context = data['sentences_chosen'][0]

                # Tokenize and Add to P_A
                tokens = tokenizer.tokenize(context)
                passage_idx.append(len(p_a))
                p_a.append(tokens)

                # Create Question/Answer State Variables
                q, ans, options = data['question'], deb_mode, data['options']
                option_idx = []

                for o_idx in range(len(options)):
                    if args.with_question:
                        option = q + " " + options[o_idx]
                    else:
                        option = options[o_idx]

                    option_tokens = tokenizer.tokenize(option)
                    option_idx.append(len(p_a))
                    p_a.append(option_tokens)

                # Create Dictionary Entry
                keys[key + "_%d_mode" % deb_mode] = {'passage_idx': passage_idx, 'question': q, 'answer': ans,
                                                     'options': options, 'option_idx': option_idx}

        return keys, p_a


def parse_dream_data(args, tokenizer):
    # Create Tracking Variables
    keys, p_a = {}, []

    if args.mode == 'judge':
        # Iterate through Data
        for dfile in [args.train, args.val[0]]:
            # Load, Iterate through Data
            with open(dfile, 'rb') as f:
                data = json.load(f)

            for i, article in enumerate(data):
                passage_idx = []

                # Tokenize Passage
                context = " ".join(article[0])

                # Tokenize and Add to P_A
                tokens = tokenizer.tokenize(context)
                passage_idx.append(len(p_a))
                p_a.append(tokens)

                # Iterate through each Question
                for idx in range(len(article[1])):
                    # Create Specific Example Key
                    key = os.path.join(article[2], str(idx))

                    # Fetch
                    q, options = article[1][idx]['question'], article[1][idx]['choice']
                    ans = options.index(article[1][idx]['answer'])

                    # Create State Variables
                    option_idx = []

                    # Tokenize Options (Q + Option if specified) and add to dict
                    for o_idx in range(len(options)):
                        if args.with_question:
                            option = q + " " + options[o_idx]
                        else:
                            option = options[o_idx]

                        option_tokens = tokenizer.tokenize(option)
                        option_idx.append(len(p_a))
                        p_a.append(option_tokens)

                    # Create Dictionary Entry
                    keys[key] = {'passage_idx': passage_idx, 'question': q, 'answer': ans, 'options': options,
                                 'option_idx': option_idx}

        return keys, p_a
    else:
        # Iterate through Train Data First!
        with open(args.train, 'rb') as f:
            data = json.load(f)

        for i, article in enumerate(data):
            passage_idx = []

            # Tokenize Passage
            context = " ".join(article[0])

            # Tokenize and Add to P_A
            tokens = tokenizer.tokenize(context)
            passage_idx.append(len(p_a))
            p_a.append(tokens)

            # Iterate through each Question
            for idx in range(len(article[1])):
                # Fetch
                q, options = article[1][idx]['question'], article[1][idx]['choice']

                # Create State Variables
                option_idx = []

                # Tokenize Options (Q + Option if specified) and add to dict
                for o_idx in range(len(options)):
                    if args.with_question:
                        option = q + " " + options[o_idx]
                    else:
                        option = options[o_idx]

                    option_tokens = tokenizer.tokenize(option)
                    option_idx.append(len(p_a))
                    p_a.append(option_tokens)

        # Iterate through all Validation Debate Logs
        for deb_mode, val in enumerate(args.val):
            with open(val, 'rb') as f:
                logs = json.load(f)

            for key in logs:
                # Fetch Data
                data = logs[key]

                # Create State Variables
                passage_idx = []

                # Tokenize Passage
                context = data['sentences_chosen'][0]

                # Tokenize and Add to P_A
                tokens = tokenizer.tokenize(context)
                passage_idx.append(len(p_a))
                p_a.append(tokens)

                # Create Question/Answer State Variables
                q, ans, options = data['question'], deb_mode, data['options']
                option_idx = []

                for o_idx in range(len(options)):
                    if args.with_question:
                        option = q + " " + options[o_idx]
                    else:
                        option = options[o_idx]

                    option_tokens = tokenizer.tokenize(option)
                    option_idx.append(len(p_a))
                    p_a.append(option_tokens)

                # Create Dictionary Entry
                keys[key + "_%d_mode" % deb_mode] = {'passage_idx': passage_idx, 'question': q, 'answer': ans,
                                                     'options': options, 'option_idx': option_idx}

        return keys, p_a


if __name__ == "__main__":
    # Parse Args
    arguments = parse_args()

    # Load BERT Tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(arguments.pretrained, do_lower_case=True)

    # Create Dataset
    if arguments.dataset == 'race':
        D, PA = parse_race_data(arguments, bert_tokenizer)

        # Compute TF Matrix
        TF = compute_tf(PA)

        # Compute TF-IDF Matrix
        print('\nComputing TF-IDF Matrix...')
        transformer = TfidfTransformer()
        TF_IDF = transformer.fit_transform(TF)
        assert (TF_IDF.shape[0] == len(PA) == len(TF))

        # Run Appropriate Accuracy Scorer
        race_judge(arguments, TF_IDF, D)

    elif arguments.dataset == 'dream':
        D, PA = parse_dream_data(arguments, bert_tokenizer)

        # Compute TF Matrix
        TF = compute_tf(PA)

        # Compute TF-IDF Matrix
        print('\nComputing TF-IDF Matrix...')
        transformer = TfidfTransformer()
        TF_IDF = transformer.fit_transform(TF)
        assert (TF_IDF.shape[0] == len(PA) == len(TF))

        # Run Appropriate Accuracy Scorer
        dream_judge(arguments, TF_IDF, D)
