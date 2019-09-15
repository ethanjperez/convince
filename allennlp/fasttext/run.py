"""
run.py

Run FastText Debater and generate debater data for the given debate option.
"""
from pytorch_pretrained_bert.tokenization import BasicTokenizer
from spacy.language import Language
from tqdm import tqdm

import argparse
import json
import numpy as np
import os


EOS_TOKENS = "(\.|\!|\?)"

ANS2IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
DEBATE2STR = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']


def parse_args():
    p = argparse.ArgumentParser(description='FastText Runner')

    p.add_argument("-v", "--val", required=True, help='Path to raw valid data to compute Fastext')
    p.add_argument("-q", "--with_question", default=False, action='store_true', help='Fastext with question + option')

    p.add_argument("-d", "--dataset", default="race", help='Which dataset to run on - dream/race.')
    p.add_argument("-p", "--pretrained", default='datasets/fasttext')
    p.add_argument("-x", "--sort", default=False, help='Return all sentences in sorted order.')
    p.add_argument("-r", "--prefix", required=True, help='Prefix for file')

    return p.parse_args()


def parse_dream_data(args, spcy, basic):
    # Create Tracking Variables
    keys = {}

    # Load, Iterate through Data
    with open(args.val, 'rb') as f:
        data = json.load(f)

    for i, article in enumerate(data):
        context = " ".join(article[0])

        # Tokenize Passage, then Perform Sentence Split
        ctx_tokens = basic.tokenize(context)

        # Iterate through tokens and create new sentence every EOS token
        ctx_sentence_tokens = [[]]
        for t in ctx_tokens:
            if t in EOS_TOKENS:
                ctx_sentence_tokens[-1].append(t)
                ctx_sentence_tokens.append([])
            else:
                ctx_sentence_tokens[-1].append(t)

        # Pop off last empty sentence if necessary
        if len(ctx_sentence_tokens[-1]) == 0:
            ctx_sentence_tokens = ctx_sentence_tokens[:-1]

        # Create Context Sentences by joining each sentence
        ctx_sentences = [" ".join(x) for x in ctx_sentence_tokens]

        # Tokenize and Vectorize each Sentence
        vec_sentences = []
        for c in ctx_sentences:
            tok_sent = spcy(c)
            if tok_sent.has_vector:
                vec_sentences.append(tok_sent)
            else:
                print('Oops')
                import IPython
                IPython.embed()

        # Iterate through each question
        for idx in range(len(article[1])):
            # Create Specific Example Key
            key = os.path.join(article[2], str(idx))

            # Fetch
            q, ans, options = article[1][idx]['question'], article[1][idx]['answer'], article[1][idx]['choice']

            # Create State Variables
            option_vecs = []

            # Tokenize Options (Q + Option if specified) and add to dict
            for o_idx in range(len(options)):
                if args.with_question:
                    option = q + " " + options[o_idx]
                else:
                    option = options[o_idx]

                option_tokens = spcy(option)
                if option_tokens.has_vector:
                    option_vecs.append(option_tokens)
                else:
                    print('Oops')
                    import IPython
                    IPython.embed()

            # Create Dictionary Entry
            keys[key] = {'passage': ctx_sentences, 'passage_vecs': vec_sentences, 'question': q, 'answer': ans,
                         'options': options, 'option_vecs': option_vecs}

    return keys


def parse_race_data(args, spcy, basic):
    # Create Tracking Variables
    keys = {}

    # Iterate through Data
    for dtype in [args.val]:
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
                ctx_tokens = basic.tokenize(context)

                # Iterate through tokens and create new sentence every EOS token
                ctx_sentence_tokens = [[]]
                for t in ctx_tokens:
                    if t in EOS_TOKENS:
                        ctx_sentence_tokens[-1].append(t)
                        ctx_sentence_tokens.append([])
                    else:
                        ctx_sentence_tokens[-1].append(t)

                # Pop off last empty sentence if necessary
                if len(ctx_sentence_tokens[-1]) == 0:
                    ctx_sentence_tokens = ctx_sentence_tokens[:-1]

                # Create Context Sentences by joining each sentence
                ctx_sentences = [" ".join(x) for x in ctx_sentence_tokens]

                # Tokenize and Vectorize each Sentence
                vec_sentences = []
                for c in ctx_sentences:
                    tok_sent = spcy(c)
                    if tok_sent.has_vector:
                        vec_sentences.append(tok_sent)
                    else:
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

                    # Tokenize Options (Q + Option if specified) and add to dict
                    for o_idx in range(len(options)):
                        if args.with_question:
                            option = q + " " + options[o_idx]
                        else:
                            option = options[o_idx]

                        option_tokens = spcy(option)
                        if option_tokens.has_vector:
                            option_vecs.append(option_tokens)
                        else:
                            import IPython
                            IPython.embed()

                    # Create Dictionary Entry
                    keys[key] = {'passage': ctx_sentences, 'passage_vecs': vec_sentences, 'question': q, 'answer': ans,
                                 'options': options, 'option_vecs': option_vecs}

    return keys


def dump_race_debates(args, keys):
    """Run Single-Turn Debates on validation set, dump to files"""
    levels = [os.path.join(args.val, x) for x in os.listdir(args.val)]
    dump_dicts = [{} for _ in range(len(DEBATE2STR))]
    for level in levels:
        passages = [os.path.join(level, x) for x in os.listdir(level)]
        print('\nRunning Debates for %s...' % level)
        for p in tqdm(passages):
            # Get Key Stub
            k, cur_question = os.path.relpath(p, args.val), 0
            while os.path.join(k, str(cur_question)) in keys:
                key = os.path.join(k, str(cur_question))
                d = keys[key]

                if not args.sort:
                    # Iterate over different option indices
                    for oidx in range(len(DEBATE2STR)):
                        option_vec = d['option_vecs'][oidx]

                        # Compute Scores
                        sent_scores = [option_vec.similarity(sent_vec) for sent_vec in d['passage_vecs']]
                        best_sent, best_score = np.argmax(sent_scores), max(sent_scores)

                        # Assemble Example Dict
                        example_dict = {"passage": " ".join(d['passage']), "question": d['question'], "advantage": 0,
                                        "debate_mode": [DEBATE2STR[oidx]], "stances": [], "em": 0,
                                        "sentences_chosen": [d['passage'][best_sent]], "answer_index": d['answer'],
                                        "prob": best_score, "options": d['options']}

                        dump_dicts[oidx][os.path.join('test', key)] = example_dict

                    cur_question += 1

                else:
                    # Return all sentence choices sorted
                    for oidx in range(len(DEBATE2STR)):
                        option_vec = d['option_vecs'][oidx]

                        # Compute Scores
                        sent_scores = np.array([option_vec.similarity(sent_vec) for sent_vec in d['passage_vecs']])
                        k_max_ind = sent_scores.argsort()[::-1]
                        chosen = [d['passage'][k] for k in k_max_ind]

                        # Assemble Example Dict
                        example_dict = {"passage": " ".join(d['passage']), "question": d['question'], "advantage": 0,
                                        "debate_mode": [DEBATE2STR[oidx]], "stances": [], "em": 0,
                                        "sentences_chosen": chosen, "answer_index": d['answer'],
                                        "prob": 0.0, "options": d['options']}

                        dump_dicts[oidx][os.path.join('test', key)] = example_dict

                    cur_question += 1

    # Dump to Files
    for i, mode in enumerate(DEBATE2STR):
        file_stub = 'fasttext/%s_race_test_fasttext_%s' % (args.prefix, mode)
        if args.with_question:
            file_stub += '_wq'

        with open(file_stub + '.json', 'w') as f:
            json.dump(dump_dicts[i], f)


def dump_dream_debates(args, keys):
    dump_dicts = [{} for _ in range(3)]

    with open(args.val, 'rb') as f:
        data = json.load(f)

    for i, article in enumerate(data):
        for idx in range(len(article[1])):
            # Get Key
            key = os.path.join(article[2], str(idx))
            d = keys[key]

            if not args.sort:
                # Iterate over different option indices
                for oidx in range(3):
                    option_vec = d['option_vecs'][oidx]

                    # Compute Scores
                    sent_scores = [option_vec.similarity(sent_vec) for sent_vec in d['passage_vecs']]
                    best_sent, best_score = np.argmax(sent_scores), max(sent_scores)

                    # Assemble Example Dict
                    example_dict = {"passage": " ".join(d['passage']), "question": d['question'], "advantage": 0,
                                    "debate_mode": [DEBATE2STR[oidx]], "stances": [], "em": 0,
                                    "sentences_chosen": [d['passage'][best_sent]], "answer_index": d['answer'],
                                    "prob": best_score, "options": d['options']}
                    dump_dicts[oidx][os.path.join('test', key)] = example_dict

            else:
                # Return all sentence choices sorted
                for oidx in range(3):
                    option_vec = d['option_vecs'][oidx]

                    # Compute Scores
                    sent_scores = np.array([option_vec.similarity(sent_vec) for sent_vec in d['passage_vecs']])
                    k_max_ind = sent_scores.argsort()[::-1]
                    chosen = [d['passage'][k] for k in k_max_ind]

                    # Assemble Example Dict
                    example_dict = {"passage": " ".join(d['passage']), "question": d['question'], "advantage": 0,
                                    "debate_mode": [DEBATE2STR[oidx]], "stances": [], "em": 0,
                                    "sentences_chosen": chosen, "answer_index": d['answer'],
                                    "prob": 0.0, "options": d['options']}
                    dump_dicts[oidx][os.path.join('test', key)] = example_dict

    # Dump to Files
    for i, mode in enumerate(DEBATE2STR[:3]):
        file_stub = 'fasttext/%s_dream_test_fasttext_%s' % (args.prefix, mode)
        if args.with_question:
            file_stub += '_wq'

        with open(file_stub + '.json', 'w') as f:
            json.dump(dump_dicts[i], f)


if __name__ == "__main__":
    # Parse Args
    arguments = parse_args()

    # Load Basic Tokenizer
    basic_tokenizer = BasicTokenizer(do_lower_case=False)

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

    # Parse Data - parses passage, question, answers and formulates vector representation
    if arguments.dataset == 'race':
        D = parse_race_data(arguments, nlp, basic_tokenizer)
        dump_race_debates(arguments, D)
    else:
        D = parse_dream_data(arguments, nlp, basic_tokenizer)
        dump_dream_debates(arguments, D)
