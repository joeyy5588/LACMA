import json
import os
import re
from tqdm import tqdm
import numpy as np
import pickle
import torch
import sys
# import matplotlib.pyplot as plt

root = sys.argv[1]
dataset = 'lmdb_human/'
# action set: move, turn left/right/around, sidestep, stepback, face left/right, u turn
'''
    sidestep: move *2, turn left/right, move *2, turn right/left, move
    stepback: turn around, move * 2, turn around
    face left/right: turn left/right, move *2, interact
    u turn: turn left/right, move, turn left/right
'''
subpolicy_count = {
    'move': 0,
    'turn left': 0,
    'turn right': 0,
    'turn around': 0,
    'step back': 0,
    'step left': 0,
    'step right': 0,
    'face left': 0,
    'face right': 0,
    'look up': 0,
    'look down': 0,
    'interaction': 0,
}
action_to_alpha = {
    'MoveAhead': 'm',
    'RotateRight': 'r',
    'RotateLeft': 'l',
    'LookUp': 'u',
    'LookDown': 'd',
}
subpolicy_to_re = {
    'move': "m{1,}",
    'turn left': "l{1}",
    'turn right': "r{1}",
    'turn around': "(lm?l)|(rm?r)",
    'step left': "(lm{,3}r)",
    'step right': "(rm{,3}l)",
    'step back': "(ll|rr)m+(ll|rr)",
    'face left': "(lm{0,2}$)|(l$)",
    'face right': "(rm{0,2}$)|(r$)",
    'look up': "u{1,}",
    'look down': "d{1,}",
    'interaction': "i{1,}",
}
num_to_subpolicy = {
    0: 'move forward',
    1: 'turn left',
    2: 'turn right',
    3: 'turn around',
    4: 'step left',
    5: 'step right',
    6: 'step back',
    7: 'face left',
    8: 'face right',
    9: 'look up',
    10: 'look down',
    11: 'interaction',
}

subpolicy_length = {}
total_traj = 0
location_argument_not_equal = 0
total_subpolicy_length = 0
max_subpolicy_length = 0
subpolicy_num = len(subpolicy_to_re)
total_ll_actions = 0
inst_to_subpolicy = {}
inst_to_boundary = {}
final_dict = {}

all_traj_data = pickle.load(open(dataset + root + '/jsons.pkl', 'rb'))
vocab = torch.load(dataset + 'data.vocab')['word']
for idx in range(len(all_traj_data)):
    key = '{:06}'.format(idx).encode('ascii')
    task_jsons = all_traj_data[key]
    for traj in task_jsons:
        ll_actions = traj['plan']['low_actions']
        total_ll_actions += len(ll_actions)
        high_idx_list = [x['high_idx'] for x in ll_actions]
        ll_actseq = [x['api_action']['action'] for x in ll_actions]
        high_pddl = traj['plan']['high_pddl']
        high_pddl = [x['discrete_action'] for x in high_pddl]

        prev_idx = 0
        curr_idx = 0
        partitioned_act = []
        for i in range(1,max(high_idx_list)+1):
            curr_idx = high_idx_list.index(i)
            partitioned_act.append(ll_actseq[prev_idx:curr_idx])
            prev_idx = curr_idx
        partitioned_act.append(ll_actseq[prev_idx:])

        nav_set = set(action_to_alpha.keys())
        nav_inst = []
        nav_pddl = []
        string_traj = []
        for i in range(len(partitioned_act)):
            subtask = (partitioned_act[i])
            nav_pddl.append(high_pddl[i])
            subtask = [action_to_alpha[x] if x in nav_set else 'i' for x in subtask]
            subtask = ''.join(subtask)
            string_traj.append(subtask)


        subpolicies = []
        total_traj += 1
        for stri in range(len(string_traj)):
            subtask = string_traj[stri]

            sub_pddl = nav_pddl[stri]['action'] + ' ' + nav_pddl[stri]['args'][0]
            if nav_pddl[stri]['args'][0].lower() == 'countertop':
                loc_arg = 'counter'
            elif nav_pddl[stri]['args'][0].lower() == 'sinkbasin':
                loc_arg = 'sink'
            elif 'table' in nav_pddl[stri]['args'][0].lower():
                loc_arg = 'table'
            elif 'lamp' in nav_pddl[stri]['args'][0].lower():
                loc_arg = 'lamp'
            else:
                loc_arg = nav_pddl[stri]['args'][0].lower()

            subpolicy = []
            dp1 = np.zeros((subpolicy_num, len(subtask), len(subtask)))
            for i, (k, v) in enumerate(subpolicy_to_re.items()):
                find_pos = [(m.start(0), m.end(0)) for m in re.finditer(v, subtask)]
                for pos in find_pos:
                    if k == 'move':
                        dp1[i][pos[0]][pos[0]:pos[1]] = 1    
                    else:
                        dp1[i][pos[0]][pos[1]-1] = 1


            dp2 = np.full((len(subtask)+1,), 100)
            traj_log = [[-1] for i in range(len(subtask)+1)]
            start_log = [[0] for i in range(len(subtask)+1)]
            dp2[0] = 0
            for i in range(len(subtask)):
                for j in range(len(subtask)):
                    for k in range(subpolicy_num):
                        if dp1[k][i][j] == 1:
                            if dp2[i] + 1 <= dp2[j+1]:
                                dp2[j+1] = dp2[i] + 1
                                
                                traj_log[j+1] = traj_log[i].copy()
                                traj_log[j+1].append(k)

                                start_log[j+1] = start_log[i].copy()
                                start_log[j+1].append(i)

            subpolicy_seq = traj_log[-1][1:]
            total_subpolicy_length += len(subpolicy_seq)
            if len(subpolicy_seq) > max_subpolicy_length:
                max_subpolicy_length = len(subpolicy_seq)
            if len(subpolicy_seq) not in subpolicy_length:
                subpolicy_length[len(subpolicy_seq)] = 1
            else:
                subpolicy_length[len(subpolicy_seq)] += 1
            for i, (k, v) in enumerate(subpolicy_to_re.items()):
                subpolicy_count[k] += subpolicy_seq.count(i)

            dict_key = traj['task_id'] + '_' + str(traj['repeat_idx'])
            dict_value = [num_to_subpolicy[x] for x in subpolicy_seq]

            ll_subpolicy = []
            boundary = start_log[-1][1:]
            subpolicy_pointer = -1
            subpolicy_boundary = []
            for i in range(len(subtask)):
                if i in boundary:
                    subpolicy_pointer += 1
                curr_sub = dict_value[subpolicy_pointer]
                ll_subpolicy.append(curr_sub)
            for i in range(len(ll_subpolicy)):
                if i in boundary:
                    subpolicy_boundary.append(1)
                else:
                    subpolicy_boundary.append(0)

            if dict_key in inst_to_subpolicy:
                inst_to_subpolicy[dict_key].append(ll_subpolicy)
                inst_to_boundary[dict_key].append(subpolicy_boundary)
            else:
                inst_to_subpolicy[dict_key] = [ll_subpolicy]
                inst_to_boundary[dict_key] = [subpolicy_boundary]

        # if ll_actseq[1] not in nav_set:
        final_subpolicy = []
        final_boundary = []
        final_sentnum = []
        for subp in inst_to_subpolicy[dict_key]:
            final_subpolicy += subp
        for sent_num, subp in enumerate(inst_to_boundary[dict_key]):
            final_boundary += subp
            final_sentnum += [sent_num] * len(subp)
        assert len(final_subpolicy) == len(ll_actions)
        final_subpolicy.append('stop')
        final_boundary.append(1)
        final_sentnum.append(0)

        final_dict[dict_key] = final_subpolicy
        traj['meta_action'] = final_subpolicy
        traj['meta_boundary'] = final_boundary
        traj['meta_sentnum'] = final_sentnum

with open(dataset + root + '/jsons.pkl', 'wb') as outfile:
    pickle.dump(all_traj_data, outfile)
print('Avg ll_actions: %f, Avg ll_subpolicy: %f' % (total_ll_actions / total_traj, total_subpolicy_length / total_traj))
print('Total traj: %d, Max subpolicy length: %d, Avg subpolicy length: %f' % (total_traj, max_subpolicy_length, total_subpolicy_length / total_traj))
print('Total traj: %d, Instruction not include location argument : %d, Percentage: %f' % (total_traj, location_argument_not_equal, location_argument_not_equal/total_traj))

dis_dict = dict(sorted(subpolicy_count.items(), key=lambda item: item[1]))
print(dis_dict)
# plt.figure(figsize=(10,10))
# plt.bar(list(dis_dict.keys()), list(dis_dict.values()), 0.1)
# plt.show()

dis_dict = dict(sorted(subpolicy_length.items()))
print(dis_dict)
# plt.figure(figsize=(10,10))
# plt.bar(list(dis_dict.keys()), list(dis_dict.values()), 0.1)
# plt.show()