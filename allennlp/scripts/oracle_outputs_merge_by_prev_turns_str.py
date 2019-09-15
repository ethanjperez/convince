import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--max_turns",
                    required=True,
                    type=int,
                    help="The maximum number of Oracle turns to save.")
parser.add_argument("-p", "--prefix",
                    default='tmp/race.best.f/oracle_outputs.c=concat.d=',
                    type=str,
                    help="The prefix for files to load.")
args = parser.parse_args()

# postfixes = ['Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ.all.pkl', 'Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ_Ⅱ_Ⅰ.all.pkl']
postfixes = ['ABABAB.all.pkl', 'BABABA.all.pkl']

save_file_postfix = str(args.max_turns) + '_ⅠⅡ_turns.all.pkl'
files = [args.prefix + postfix for postfix in postfixes]
save_file = args.prefix + save_file_postfix
assert not os.path.exists(save_file), 'Save file already exists! Not overriding: ' + save_file
print('Saving to:', save_file)


def merge_dicts_by_key_and_value(*dict_args, max_turns=None):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = dict_args[0]
    for dictionary in dict_args[1:]:
        for sample_id, oracle_outputs_per_turn in dictionary.items():
            result[sample_id].update(oracle_outputs_per_turn)
    if max_turns is not None:
        for sample_id, oracle_outputs_per_turn in result.items():
            filtered_results = {}
            for turn, oracle_outputs in result[sample_id].items():
                if (len(turn.replace('_', '')) + 1) <= max_turns:
                    filtered_results[turn] = oracle_outputs
            result[sample_id] = filtered_results
    return result


oracle_outputs = []
for file in files:
    print('Reading', file, '...')
    with open(file, 'rb') as f:
        oracle_outputs.append(pickle.load(f))

print('Merging dictionaries...')
fixed_all_oracle_outputs = merge_dicts_by_key_and_value(*oracle_outputs, max_turns=args.max_turns)

print('Saving to file:', save_file, '...')
with open(save_file, 'wb') as f:
    pickle.dump(fixed_all_oracle_outputs, f, pickle.HIGHEST_PROTOCOL)

example_key = list(fixed_all_oracle_outputs.keys())[-1]
print('Example key:', example_key)
example_prev_turns_strs = list(fixed_all_oracle_outputs[example_key].keys())
print('Example prev_turns_strs:', example_prev_turns_strs)
print('Example oracle_output dict:', fixed_all_oracle_outputs[example_key][example_prev_turns_strs[0]])
print('Done!')
