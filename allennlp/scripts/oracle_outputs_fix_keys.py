import os
import pickle
import torch

file = 'tmp/race.best.f/oracle_outputs.c=concat.d=A_B_A_B_A_B_A_B.all.pkl'

save_file = file[:-3] + 'fixed.pkl'
assert not os.path.exists(save_file), 'Save file already exists! Not overriding: ' + save_file

print('Reading', file, '...')
with open(file, 'rb') as f:
    oracle_output = pickle.load(f)

print('Fixing keys: Replacing cum_turn_str with prev_turns_str...')
fixed_all_oracle_outputs = {}
for sample_id, sample_oracle_outputs in oracle_output.items():
    fixed_all_oracle_outputs[sample_id] = {}
    for cum_turn_str, oracle_dict in sample_oracle_outputs.items():
        prev_turns_str = '_'.join(list(cum_turn_str[:-1]))
        fixed_all_oracle_outputs[sample_id][prev_turns_str] = oracle_dict

print('Saving to file:', save_file, '...')
with open(save_file, 'wb') as f:
    pickle.dump(fixed_all_oracle_outputs, f, pickle.HIGHEST_PROTOCOL)

example_key = list(fixed_all_oracle_outputs.keys())[-1]
print('Example key:', example_key)
example_prev_turns_strs = list(fixed_all_oracle_outputs[example_key].keys())
print('Example prev_turns_strs:', example_prev_turns_strs)
print('Example oracle_output dict:', fixed_all_oracle_outputs[example_key][example_prev_turns_strs[0]])
print('Done!')
