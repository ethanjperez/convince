"""
run.py

Run TF-IDF Debater and generate debater data for the given debate option.
"""
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

import argparse
import json
import os
import numpy as np
import tqdm

EOS_TOKENS = ['.', '!', '?']

ANS2IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
DEBATE2STR = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']


def parse_args():
    p = argparse.ArgumentParser(description='TF-IDF Runner')

    p.add_argument("-m", "--mode", required=True, help='Mode in n-sents, q-only')
    p.add_argument("-t", "--train", required=True, help='Path to raw train data to compute TF-IDF')
    p.add_argument("-v", "--val", required=True, help='Path to raw valid data to compute TF-IDF')
    p.add_argument("-d", "--debate_option", default=0, type=int, help='Which MC option to support (I, II, III, IV)')

    p.add_argument("-s", "--dataset", default='race', help='Dataset to run on')
    p.add_argument("-p", "--pretrained", default='datasets/bert/uncased_L-12_H-768_A-12/vocab.txt')
    p.add_argument("-x", "--sort", default=False, help='Whether or not to return all sentences sorted by choice.')
    p.add_argument("-r", "--prefix", required=True, help='Prefix for file')

    return p.parse_args()


def parse_data(args, tokenizer, basic):
    # Create Tracking Variables
    keys, p_a = {}, []

    # Iterate through Data
    for dtype in [args.train, args.val]:
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

                # Create BERT CTX Sentence Tokens
                ctx_sentence_tokens = [tokenizer.tokenize(x) for x in ctx_sentences]
                for sent in ctx_sentence_tokens:
                    passage_idx.append(len(p_a))
                    p_a.append(sent)

                # Iterate through each Question
                for idx in range(len(data['questions'])):
                    # Create Specific Example Key
                    key = os.path.join(k, str(idx))

                    # Fetch
                    q, ans, options = data['questions'][idx], ANS2IDX[data['answers'][idx]], data['options'][idx]

                    # Tokenize Question
                    question_tokens = tokenizer.tokenize(q)
                    q_idx = len(p_a)
                    p_a.append(question_tokens)

                    # Create Dictionary Entry
                    keys[key] = {'passage': ctx_sentences, 'passage_idx': passage_idx, 'question': q, 'answer': ans,
                                 'options': options, 'question_idx': q_idx}

    return keys, p_a


def parse_dream_data(args, tokenizer, basic):
    # Create Tracking Variables
    keys, p_a = {}, []

    # Iterate through Data
    for dfile in [args.train, args.val]:
        # Load, Iterate through Data
        with open(dfile, 'rb') as f:
            data = json.load(f)

        for i, article in enumerate(data):
            passage_idx = []
            context = " ".join(article[0])
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

            # Create BERT CTX Sentence Tokens
            ctx_sentence_tokens = [tokenizer.tokenize(x) for x in ctx_sentences]

            for sent in ctx_sentence_tokens:
                passage_idx.append(len(p_a))
                p_a.append(sent)

            # Iterate through each Question
            for idx in range(len(article[1])):
                # Create Specific Example Key
                key = os.path.join(article[2], str(idx))

                # Fetch
                q, ans, options = article[1][idx]['question'], article[1][idx]['answer'], article[1][idx]['choice']

                # Tokenize Question
                question_tokens = tokenizer.tokenize(q)
                q_idx = len(p_a)
                p_a.append(question_tokens)

                # Create Dictionary Entry
                keys[key] = {'passage': ctx_sentences, 'passage_idx': passage_idx, 'question': q, 'answer': ans,
                             'options': options, 'question_idx': q_idx}

    return keys, p_a


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


def dump_debates(args, idf, keys):
    """Run Single-Turn Debates on validation set, dump to file"""
    levels = [os.path.join(args.val, x) for x in os.listdir(args.val)]
    dump_dicts = [{} for _ in range(len(DEBATE2STR))]
    for level in levels:
        passages = [os.path.join(level, x) for x in os.listdir(level)]
        print('\nRunning Debates for %s...' % level)
        for p in tqdm.tqdm(passages):
            # Get Key Stub
            k, cur_question = os.path.relpath(p, args.val), 0
            while os.path.join(k, str(cur_question)) in keys:
                key = os.path.join(k, str(cur_question))
                d = keys[key]

                # Mode in num-sent
                if args.mode == 'n-sents':
                    for oidx in range(len(DEBATE2STR)):
                        # Get Chosen Sentence
                        if oidx < len(d['passage']):
                            chosen = d['passage'][oidx]
                        else:
                            chosen = ""

                        # Assemble Example Dict
                        example_dict = {"passage": " ".join(d['passage']), "question": d['question'], "advantage": 0,
                                        "debate_mode": [DEBATE2STR[args.debate_option]], "stances": [], "em": 0,
                                        "sentences_chosen": [chosen], "answer_index": d['answer'],
                                        "prob": 0.0, "options": d['options']}
                        dump_dicts[oidx][os.path.join('test', key)] = example_dict

                    cur_question += 1

                # Mode in q-only:
                elif args.mode == 'q-only':
                    # Get Q-IDX, Passage-IDX
                    q_idx, passage_idx = d['question_idx'], d['passage_idx']

                    if (idf[passage_idx].shape[0] == 0) or (idf[q_idx].shape[0] == 0):
                        import IPython
                        IPython.embed()

                    # Compute Scores
                    sent_scores = cosine_similarity(idf[passage_idx], idf[q_idx]).flatten()
                    k_max_ind = sent_scores.argsort()[::-1]

                    if not args.sort:
                        for oidx in range(len(DEBATE2STR)):
                            # Get Chosen Sentence
                            if oidx < len(k_max_ind):
                                chosen = d['passage'][k_max_ind[oidx]]
                            else:
                                chosen = ""

                            # Assemble Example Dict
                            example_dict = {"passage": " ".join(d['passage']), "question": d['question'],
                                            "advantage": 0, "debate_mode": [DEBATE2STR[args.debate_option]],
                                            "stances": [], "em": 0, "sentences_chosen": [chosen],
                                            "answer_index": d['answer'], "prob": 0.0, "options": d['options']}
                            dump_dicts[oidx][os.path.join('test', key)] = example_dict

                        cur_question += 1

                    else:
                        # Return all sentence choices sorted
                        chosen = [d['passage'][k] for k in k_max_ind]

                        # Assemble Example Dict
                        example_dict = {"passage": " ".join(d['passage']), "question": d['question'],
                                        "advantage": 0, "debate_mode": [DEBATE2STR[0]],
                                        "stances": [], "em": 0, "sentences_chosen": chosen,
                                        "answer_index": d['answer'], "prob": 0.0, "options": d['options']}
                        dump_dicts[0][os.path.join('test', key)] = example_dict
                        cur_question += 1

    # Dump to file
    if not args.sort:
        for i, mode in enumerate(DEBATE2STR):
            file_stub = 'tf_idf/race_test_tfidf_%s_%s' % (args.mode, mode)

            with open(file_stub + ".json", 'w') as f:
                json.dump(dump_dicts[i], f)
    else:
        file_stub = 'tf_idf/%s_race_test_tfidf_agnostic_all_sorted' % args.prefix
        with open(file_stub + '.json', 'w') as f:
            json.dump(dump_dicts[0], f)


def dump_dream_debates(args, idf, keys):
    dump_dicts = [{} for _ in range(3)]

    with open(args.val, 'rb') as f:
        data = json.load(f)

    for i, article in enumerate(data):
        for idx in range(len(article[1])):
            # Get Key
            key = os.path.join(article[2], str(idx))
            d = keys[key]

            # Mode in num-sent
            if args.mode == 'n-sents':
                for oidx in range(3):
                    # Get Chosen Sentence
                    if oidx < len(d['passage']):
                        chosen = d['passage'][oidx]
                    else:
                        chosen = ""

                    # Assemble Example Dict
                    example_dict = {"passage": " ".join(d['passage']), "question": d['question'], "advantage": 0,
                                    "debate_mode": [DEBATE2STR[oidx]], "stances": [], "em": 0,
                                    "sentences_chosen": [chosen], "answer_index": d['answer'],
                                    "prob": 0.0, "options": d['options']}
                    dump_dicts[oidx][os.path.join('test', key)] = example_dict

            elif args.mode == 'q-only':
                # Get Q-IDX, Passage-IDX
                q_idx, passage_idx = d['question_idx'], d['passage_idx']

                if (idf[passage_idx].shape[0] == 0) or (idf[q_idx].shape[0] == 0):
                    import IPython
                    IPython.embed()

                # Compute Scores
                sent_scores = cosine_similarity(idf[passage_idx], idf[q_idx]).flatten()
                k_max_ind = sent_scores.argsort()[::-1]

                if not args.sort:
                    for oidx in range(3):
                        # Get Chosen Sentence
                        if oidx < len(k_max_ind):
                            chosen = d['passage'][k_max_ind[oidx]]
                        else:
                            chosen = ""

                        # Assemble Example Dict
                        example_dict = {"passage": " ".join(d['passage']), "question": d['question'], "advantage": 0,
                                        "debate_mode": [DEBATE2STR[args.debate_option]], "stances": [], "em": 0,
                                        "sentences_chosen": [chosen], "answer_index": d['answer'],
                                        "prob": 0.0, "options": d['options']}
                        dump_dicts[oidx][os.path.join('test', key)] = example_dict

                else:
                    # Return all sentence choices sorted
                    chosen = [d['passage'][k] for k in k_max_ind]

                    # Assemble Example Dict
                    example_dict = {"passage": " ".join(d['passage']), "question": d['question'], "advantage": 0,
                                    "debate_mode": [DEBATE2STR[args.debate_option]], "stances": [], "em": 0,
                                    "sentences_chosen": chosen, "answer_index": d['answer'],
                                    "prob": 0.0, "options": d['options']}
                    dump_dicts[0][os.path.join('test', key)] = example_dict

    if not args.sort:
        for i, mode in enumerate(DEBATE2STR[:3]):
            file_stub = 'tf_idf/dream_test_tfidf_%s_%s' % (args.mode, mode)

            with open(file_stub + '.json', 'w') as f:
                json.dump(dump_dicts[i], f)

    else:
        file_stub = 'tf_idf/%s_dream_test_tfidf_agnostic_all_sorted' % args.prefix
        with open(file_stub + '.json', 'w') as f:
            json.dump(dump_dicts[0], f)


if __name__ == '__main__':
    # Parse Args
    arguments = parse_args()

    # Load BERT Tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(arguments.pretrained, do_lower_case=True)
    basic_tokenizer = BasicTokenizer(do_lower_case=False)

    # Create Dataset
    if arguments.dataset == 'race':
        D, PA = parse_data(arguments, bert_tokenizer, basic_tokenizer)

        # Compute TF Matrix
        TF = compute_tf(PA)

        # Compute TF-IDF Matrix
        print('\nComputing TF-IDF Matrix...')
        transformer = TfidfTransformer()
        TF_IDF = transformer.fit_transform(TF)
        assert(TF_IDF.shape[0] == len(PA) == len(TF))

        # Compute Scoring Matrix
        print('\nScoring Matrix...')

        # Dump Debates
        dump_debates(arguments, TF_IDF, D)
    else:
        D, PA = parse_dream_data(arguments, bert_tokenizer, basic_tokenizer)

        # Compute TF Matrix
        TF = compute_tf(PA)

        # Compute TF-IDF Matrix
        print('\nComputing TF-IDF Matrix...')
        transformer = TfidfTransformer()
        TF_IDF = transformer.fit_transform(TF)
        assert (TF_IDF.shape[0] == len(PA) == len(TF))

        # Compute Scoring Matrix
        print('\nScoring Matrix...')

        # Dump Debates
        dump_dream_debates(arguments, TF_IDF, D)
