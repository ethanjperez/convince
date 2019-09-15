import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prefix",
                    default='/checkpoint/siddk/debate/runs/dream/dream.bert_mc_gpt.bsz=32.lr=2.0e-05.f/oracle_outputs.d=I',
                    type=str,
                    help="The prefix for files to load.")
args = parser.parse_args()

postfixes = ['train.pkl', 'dev.pkl', 'test.pkl']

files = [args.prefix + '.' + postfix for postfix in postfixes]
save_file = args.prefix + '.all.pkl'
assert not os.path.exists(save_file), 'Save file already exists! Not overriding: ' + save_file


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def fix_sample_id(sample_id: str) -> str:
    if 'datasets' not in sample_id:
        return sample_id
    file_parts = sample_id.split('/')[2:]
    for split in ['train', 'dev', 'test']:
        if split in file_parts[0]:
            file_parts[0] = split
            break
    return '/'.join(file_parts)


oracle_outputs = []
for file in files:
    print('Reading', file, '...')
    with open(file, 'rb') as f:
        oracle_outputs.append(pickle.load(f))

print('Merging dictionaries...')
all_oracle_outputs = merge_dicts(*oracle_outputs)

print('Fixing sample_ids...')
fixed_all_oracle_outputs = {fix_sample_id(sample_id): v for sample_id, v in all_oracle_outputs.items()}

print('Saving to file:', save_file, '...')
with open(save_file, 'wb') as f:
    pickle.dump(fixed_all_oracle_outputs, f, pickle.HIGHEST_PROTOCOL)

example_key = list(fixed_all_oracle_outputs.keys())[-1]
print('Example key:', example_key)
example_prev_turns_strs = list(fixed_all_oracle_outputs[example_key].keys())
print('Example prev_turns_strs:', example_prev_turns_strs)
print('Example oracle_output dict:', fixed_all_oracle_outputs[example_key][example_prev_turns_strs[0]])
print('Done!')
