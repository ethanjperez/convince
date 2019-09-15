"""
dream2allen.py

Converts DREAM Debate Logs to the Allennlp-Expected DREAM Format.
"""
from tqdm import tqdm
import argparse
import json

DUMP_DIR = 'datasets/dream/test_%s.json'


def parse_args():
    p = argparse.ArgumentParser(description='Debate Logs -> Allennlp')
    p.add_argument("-m", "--mode", required=True, help='String ID for Debater Mode (tfidf, fasttext, bert, etc.)')

    p.add_argument("-v", "--val", nargs='+', required=True, help='Path to debate logs for mode agent')
    return p.parse_args()


def translate(d, logs):
    dump_data = []
    for stance_idx, val in enumerate(logs):
        print("Processing Stance %d" % stance_idx)
        with open(val, 'rb') as f:
            data = json.load(f)

        # Start Iterating through
        for key in tqdm(data):
            # Get Passage
            passage = data[key]['sentences_chosen']

            # Build Question/Choice/Answer Dict
            qca = {
                "question": data[key]['question'],
                "choice": data[key]['options'],
                "answer": data[key]['options'][stance_idx]
            }

            # Create Key
            key_id = "%s_d%d" % (key, stance_idx)

            # Add to Dump Data
            dump_data.append([passage, [qca], key_id])

    # Dump JSON File
    with open(d, 'w') as f:
        json.dump(dump_data, f)


if __name__ == "__main__":
    # Parse Arguments
    args = parse_args()

    dump_file = DUMP_DIR % args.mode

    # Iterate through Debate Logs and Dump
    translate(dump_file, args.val)
