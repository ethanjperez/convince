"""
create_annotateable

Creates debate logs from raw files, with prepended sentence ID tokens.
"""
from pytorch_pretrained_bert.tokenization import BasicTokenizer

import argparse
import json
import os
import re
import tqdm


ANS2IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
DEBATE2STR = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']
EOS_TOKENS = "(\.|\!|\?)"


def parse_args():
    p = argparse.ArgumentParser(description='Annotateable Runner')

    p.add_argument("-t", "--train", required=True, help='Path to raw test data to annotate.')
    p.add_argument("-s", "--dataset", default='race', help='Dataset to run on')

    return p.parse_args()


def parse_race_data(args):
    # Create Tracking Variables
    keys = {}

    # Iterate through Data
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

            # Tokenize Passage => Split into Sentences, then Tokenize each Sentence
            context = data['article']

            # Split on ./!/?
            ctx_split = re.split(EOS_TOKENS, context)[:-1]
            ctx_sentences = [(ctx_split[i] + ctx_split[i + 1]).strip() for i in range(0, len(ctx_split), 2)]

            # Iterate through each Question
            for idx in range(len(data['questions'])):
                # Create Specific Example Key
                key = os.path.join(k, str(idx))

                # Fetch
                q, ans, options = data['questions'][idx], ANS2IDX[data['answers'][idx]], data['options'][idx]

                # Create Dictionary Entry
                keys[key] = {'passage': ctx_sentences, 'question': q, 'answer': ans, 'options': options}

    return keys


def parse_dream_data(args, basic):
    # Create Tracking Variables
    keys = {}

    # Iterate through Data
    with open(args.train, 'rb') as f:
        data = json.load(f)

    for i, article in enumerate(data):
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


        # Iterate through each Question
        for idx in range(len(article[1])):
            # Create Specific Example Key
            key = os.path.join(article[2], str(idx))

            # Fetch
            q, ans, options = article[1][idx]['question'],  article[1][idx]['answer'], article[1][idx]['choice']

            # Create Dictionary Entry
            keys[key] = {'passage': ctx_sentences, 'question': q, 'answer': ans, 'options': options}

    return keys


def dump_race_debates(args, keys):
    levels = [os.path.join(args.train, x) for x in os.listdir(args.train)]
    dump_dicts = [{} for _ in range(len(DEBATE2STR))]
    for level in levels:
        passages = [os.path.join(level, x) for x in os.listdir(level)]
        for p in tqdm.tqdm(passages):
            # Get Key Stub
            k, cur_question = os.path.relpath(p, args.train), 0
            while os.path.join(k, str(cur_question)) in keys:
                key = os.path.join(k, str(cur_question))
                d = keys[key]

                # Create Annotated Passage
                annotated_passage = [("%d: " % i) + x for i, x in enumerate(d['passage'])]

                for oidx in range(len(DEBATE2STR)):
                    # Assemble Example Dict
                    example_dict = {"passage": " ".join(d['passage']), "annotated_passage": "\n".join(annotated_passage),
                                    "passage_sentences": d['passage'], "question": d['question'], "options": d['options'],
                                    "debate_mode": [DEBATE2STR[oidx]]}

                    dump_dicts[oidx][os.path.join('test', key)] = example_dict

                cur_question += 1

    for i, mode in enumerate(DEBATE2STR):
        file_stub = 'tf_idf/race_annotated_%s' % mode
        with open(file_stub + '.json', 'w') as f:
            json.dump(dump_dicts[i], f)


def dump_dream_debates(args, keys):
    dump_dicts = [{} for _ in range(3)]
    for key in keys:
        d = keys[key]

        # Create Annotated Passage
        annotated_passage = [("%d: " % i) + x for i, x in enumerate(d['passage'])]

        for oidx in range(3):
            example_dict = {"passage": " ".join(d['passage']), "annotated_passage": "\n".join(annotated_passage),
                            "passage_sentences": d['passage'], "question": d['question'], "options": d['options'],
                            "debate_mode": [DEBATE2STR[oidx]]}

            dump_dicts[oidx][os.path.join('test', key)] = example_dict

    for i, mode in enumerate(DEBATE2STR[:3]):
        file_stub = 'tf_idf/dream_annotated_%s' % mode
        with open(file_stub + '.json', 'w') as f:
            json.dump(dump_dicts[i], f)


if __name__ == '__main__':
    # Parse Args
    arguments = parse_args()

    basic_tokenizer = BasicTokenizer(do_lower_case=False)

    # Create Dataset
    if arguments.dataset == 'race':
        D = parse_race_data(arguments)
        dump_race_debates(arguments, D)

    elif arguments.dataset == 'dream':
        D = parse_dream_data(arguments, basic_tokenizer)
        dump_dream_debates(arguments, D)
