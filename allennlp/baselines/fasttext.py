"""
fasttext.py

FastText Baseline (running as judge - takes debate logs as input, returns persuasiveness accuracy)
"""
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language
from tqdm import tqdm

import argparse
import json
import numpy as np
import os

ANS2IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
DEBATE2IDX = {'Ⅰ': 0, 'Ⅱ': 1, 'Ⅲ': 2, 'Ⅳ': 3}


def parse_args():
    p = argparse.ArgumentParser(description='FastText Judge')
    p.add_argument("-m", "--mode", default='cross-model', help='Mode to run in < judge | cross-model >')
    p.add_argument("-d", "--dataset", default='dream', help='Dataset to run on < race | dream >')

    p.add_argument("-v", "--val", nargs='+', required=True, help='Paths to debate logs for each agent.')

    p.add_argument("-p", "--pretrained", default='datasets/fasttext')
    return p.parse_args()


def race_judge(args, keys):
    """Run and Compute Accuracy on Baseline QA Model"""
    if args.mode == 'judge':
        levels = [os.path.join(args.val[0], x) for x in os.listdir(args.val[0])]
        correct, total = 0, 0
        for level in levels:
            passages = [os.path.join(level, x) for x in os.listdir(level)]
            print('\nRunning Debates for %s...' % level)
            for p in tqdm(passages):
                # Get Key Stub
                k, cur_question = os.path.relpath(p, args.val[0]), 0
                while os.path.join(k, str(cur_question)) in keys:
                    key = os.path.join(k, str(cur_question))
                    d = keys[key]

                    # Compute Scores
                    passage_vec = np.array([d['passage'].vector])
                    option_vecs = np.array([x.vector for x in d['option_vecs']])

                    opt_scores = cosine_similarity(option_vecs, passage_vec).flatten()
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
            passage_vec = np.array([d['passage'].vector])
            option_vecs = np.array([x.vector for x in d['option_vecs']])

            opt_scores = cosine_similarity(option_vecs, passage_vec).flatten()
            best_opt = np.argmax(opt_scores)

            # Score
            if best_opt == d['answer']:
                correct += 1

            total += 1

        print("\nPersuasion Accuracy: %.5f out of %d Total Examples" % (correct / total, total))


def dream_judge(args, keys):
    """Run and Compute Accuracy on Baseline QA Model"""
    if args.mode == 'judge':
        with open(args.val[0], 'rb') as f:
            data = json.load(f)

        correct, total = 0, 0
        for i, article in enumerate(data):
            for idx in range(len(article[1])):
                # Get Key
                key = os.path.join(article[2], str(idx))
                d = keys[key]

                # Compute Scores
                passage_vec = np.array([d['passage'].vector])
                option_vecs = np.array([x.vector for x in d['option_vecs']])

                opt_scores = cosine_similarity(option_vecs, passage_vec).flatten()
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
            passage_vec = np.array([d['passage'].vector])
            option_vecs = np.array([x.vector for x in d['option_vecs']])

            opt_scores = cosine_similarity(option_vecs, passage_vec).flatten()
            best_opt = np.argmax(opt_scores)

            # Score
            if best_opt == d['answer']:
                correct += 1

            total += 1

        print("\nPersuasion Accuracy: %.5f out of %d Total Examples" % (correct / total, total))


def parse_race_data(args, spcy):
    # Create Tracking Variables
    keys = {}

    if args.mode == 'judge':
        # Iterate through Data
        for dtype in [args.val[0]]:
            levels = [os.path.join(dtype, x) for x in os.listdir(dtype)]
            for level in levels:
                passages = [os.path.join(level, x) for x in os.listdir(level)]

                print('\nProcessing %s...' % level)
                for p in tqdm(passages):
                    # Get Key Stub
                    k = os.path.relpath(p, dtype)

                    # Read File
                    with open(p, 'rb') as f:
                        data = json.load(f)

                    # Tokenize Passage => Tokenize Passage, then Perform Sentence Split
                    context = data['article']

                    # Get Context Vector
                    tok_context = spcy(context)
                    if not tok_context.has_vector:
                        import IPython
                        IPython.embed()

                    # Iterate through each Question
                    for idx in range(len(data['questions'])):
                        # Create Specific Example Key
                        key = os.path.join(k, str(idx))

                        # Fetch
                        q, ans, options = data['questions'][idx], ANS2IDX[data['answers'][idx]], data['options'][idx]

                        # Create State Variables
                        option_vecs = []

                        # Tokenize Options (Q + Option if specified) and Add to P_A
                        for o_idx in range(len(options)):
                            option = options[o_idx]

                            option_tokens = spcy(option)
                            if option_tokens.has_vector:
                                option_vecs.append(option_tokens)
                            else:
                                import IPython
                                IPython.embed()

                        # Create Dictionary Entry
                        keys[key] = {'passage': tok_context, 'question': q, 'answer': ans, 'options': options,
                                     'option_vecs': option_vecs}

        return keys

    else:
        # Iterate through all Validation Debate Logs
        for deb_mode, val in enumerate(args.val):
            with open(val, 'rb') as f:
                logs = json.load(f)

            for key in logs:
                # Fetch Data
                data = logs[key]

                # Tokenize Passage
                context = data['sentences_chosen'][0]

                # Get Context Vector
                tok_context = spcy(context)
                if not tok_context.has_vector:
                    import IPython
                    IPython.embed()

                # Create Question/Answer State Variables
                q, ans, options = data['question'], deb_mode, data['options']
                option_vecs = []

                for o_idx in range(len(options)):
                    option = options[o_idx]

                    option_tokens = spcy(option)
                    if not option_tokens.has_vector:
                        import IPython
                        IPython.embed()
                    option_vecs.append(option_tokens)

                # Create Dictionary Entry
                keys[key + "_%d_mode" % deb_mode] = {'passage': tok_context, 'question': q, 'answer': ans,
                                                     'options': options, 'option_vecs': option_vecs}

        return keys


def parse_dream_data(args, spcy):
    # Create Tracking Variables
    keys = {}

    if args.mode == 'judge':
        # Iterate through Data
        with open(args.val[0], 'rb') as f:
            data = json.load(f)

        for i, article in enumerate(data):
            context = " ".join(article[0])

            # Tokenize Passage
            tok_context = spcy(context)
            if not tok_context.has_vector:
                import IPython
                IPython.embed()

            # Iterate through each Question
            for idx in range(len(article[1])):
                # Create Specific Example Key
                key = os.path.join(article[2], str(idx))

                # Fetch
                q, options = article[1][idx]['question'], article[1][idx]['choice']
                ans = options.index(article[1][idx]['answer'])

                option_vecs = []

                # Tokenize Options
                for o_idx in range(len(options)):
                    option = options[o_idx]

                    option_tokens = spcy(option)
                    if not option_tokens.has_vector:
                        import IPython
                        IPython.embed()
                    option_vecs.append(option_tokens)

                # Create Dictionary Entry
                keys[key] = {'passage': tok_context, 'question': q, 'answer': ans, 'options': options,
                             'option_vecs': option_vecs}

        return keys

    else:
        # Iterate through all Validation Debate Logs
        for deb_mode, val in enumerate(args.val):
            with open(val, 'rb') as f:
                logs = json.load(f)

            for key in logs:
                # Fetch Data
                data = logs[key]

                # Tokenize Passage
                context = data['sentences_chosen'][0]

                # Get Context Vector
                tok_context = spcy(context)
                if not tok_context.has_vector:
                    import IPython
                    IPython.embed()

                # Create Question/Answer State Variables
                q, ans, options = data['question'], deb_mode, data['options']
                option_vecs = []

                for o_idx in range(len(options)):
                    option = options[o_idx]

                    option_tokens = spcy(option)
                    if not option_tokens.has_vector:
                        import IPython
                        IPython.embed()
                    option_vecs.append(option_tokens)

                # Create Dictionary Entry
                keys[key + "_%d_mode" % deb_mode] = {'passage': tok_context, 'question': q, 'answer': ans,
                                                     'options': options, 'option_vecs': option_vecs}

        return keys


if __name__ == "__main__":
    # Parse Args
    arguments = parse_args()

    # Get FastText Data if it doesn't exist
    if not os.path.exists(os.path.join(arguments.pretrained, 'crawl-300d-2M.vec')):
        os.system('wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip')
        os.system('unzip crawl-300d-2M.vec.zip')

        if not os.path.exists(arguments.pretrained):
            os.makedirs(arguments.pretrained)

        os.system('rm crawl-300d-2M.vec.zip')
        os.system('mv crawl-300d-2M.vec %s' % arguments.pretrained)

    # Use Spacy to load Vectors
    nlp = Language()
    print('[*] Loading Vectors with Spacy...')
    with open(os.path.join(arguments.pretrained, 'crawl-300d-2M.vec'), "rb") as f:
        header = f.readline()
        nr_row, nr_dim = header.split()
        nlp.vocab.reset_vectors(width=int(nr_dim))
        for line in tqdm(f):
            line = line.rstrip().decode("utf8")
            pieces = line.rsplit(" ", int(nr_dim))
            word = pieces[0]
            vector = np.asarray([float(v) for v in pieces[1:]], dtype="f")
            nlp.vocab.set_vector(word, vector)

    # Create Dataset
    if arguments.dataset == 'race':
        D = parse_race_data(arguments, nlp)

        # Run Appropriate Accuracy Scorer
        race_judge(arguments, D)

    elif arguments.dataset == 'dream':
        D = parse_dream_data(arguments, nlp)

        # Run Appropriate Accuracy Scorer
        dream_judge(arguments, D)

