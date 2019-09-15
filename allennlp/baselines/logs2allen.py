"""
logs2allen.py

Translates Debate Logs to the Allennlp-Expected RACE Format.
"""
from tqdm import tqdm
import argparse
import json
import os


DUMP_DIR = 'datasets/race_raw/test_%s'

ANSWERS = ['A', 'B', 'C', 'D']


def parse_args():
    p = argparse.ArgumentParser(description='Debate Logs -> Allennlp Translator')
    p.add_argument("-m", "--mode", required=True, help='String ID for Debater Mode (tfidf, fasttext, bert, etc.)')

    p.add_argument("-v", "--val", nargs='+', required=True, help='Path to debate logs for mode agent')
    return p.parse_args()


def translate(d, log, stance_idx):
    with open(log, 'rb') as f:
        data = json.load(f)

    print("Processing Stance %d..." % stance_idx)
    for key in tqdm(data):
        # Split Key Up
        dtype, lvl, text_id, q_num = key.split('/')
        assert (dtype == 'test')

        # Create file path
        file_path = os.path.join(d, lvl, 'd%d-%s-q%s' % (stance_idx, text_id, q_num))

        # Create Answers list
        answers = [ANSWERS[stance_idx]]

        # Create options list
        options = [data[key]['options']]

        assert(len(options[0]) == 4)

        # Create questions list
        questions = [data[key]['question']]

        # Create article
        article = data[key]['sentences_chosen'][0]

        # Create id
        id_str = 'd%d-q%s-%s' % (stance_idx, q_num, text_id)

        # Write file
        with open(file_path, 'w') as f:
            json.dump({"answers": answers, "options": options, "questions": questions, "article": article,
                       "id": id_str}, f)


if __name__ == "__main__":
    # Parse Arguments
    args = parse_args()

    # Create Dump Dir
    dump_dir = DUMP_DIR % args.mode

    # Create High-Level Directories
    high_dir = os.path.join(dump_dir, "high")
    mid_dir = os.path.join(dump_dir, "middle")

    os.makedirs(high_dir)
    os.makedirs(mid_dir)

    # Iterate through Debate Logs and Dump
    for i, val in enumerate(args.val):
        translate(dump_dir, val, i)
