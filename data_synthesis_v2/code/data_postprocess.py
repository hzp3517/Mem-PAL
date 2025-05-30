'''
对生成的log, dialogue以及对话相关的需求及方案做后处理，以留下的每个对话作为一个sample的粒度，将对话、需求及方案，以及对应时间段内的日志统一组织起来。
日志按时间戳排序
'''
import os
import json
import random
from tqdm import tqdm
from copy import copy, deepcopy
from datetime import datetime

data_dir = '/data5/hzp/MemPAL/data_synthesis_v2/data'
situation_file = os.path.join(data_dir, 'situation', 'situation.json')
with open(situation_file, 'r', encoding='utf-8') as f:
    situation_dict = json.load(f)
dialogue_situation_file = os.path.join(data_dir, 'situation', 'dialogue_situation_ids.json')
with open(dialogue_situation_file, 'r', encoding='utf-8') as f:
    dialogue_situation_dict = json.load(f)
log_file = os.path.join(data_dir, 'log', 'log.json')
with open(log_file, 'r', encoding='utf-8') as f:
    log_dict = json.load(f)
dialogue_file = os.path.join(data_dir, 'dialogue', 'dialogue.json')
with open(dialogue_file, 'r', encoding='utf-8') as f:
    dialogue_dict = json.load(f)
candidate_solution_file = os.path.join(data_dir, 'dialogue_framework', 'candidate_solution', 'candidate_solution.json')
with open(candidate_solution_file, 'r', encoding='utf-8') as f:
    candidate_solution_dict = json.load(f)
requirement_framework_file = os.path.join(data_dir, 'dialogue_framework', 'requirement', 'requirement_framework.json')
with open(requirement_framework_file, 'r', encoding='utf-8') as f:
    requirement_framework_dict = json.load(f)
solution_preference_file = os.path.join(data_dir, 'dialogue_framework', 'solution_preference', 'solution_preference.json')
with open(solution_preference_file, 'r', encoding='utf-8') as f:
    solution_preference_dict = json.load(f)


save_path = os.path.join(data_dir, 'input.json')

save_dict = {}

for user_id in dialogue_situation_dict.keys():
    user_dict = {}
    user_dict['history'] = {}
    user_dict['query'] = {}

    # log重新组织并排序
    user_log_list = []
    for date in log_dict[user_id].keys():
        for log_item in log_dict[user_id][date]:
            new_log_item = copy(log_item)
            del new_log_item['event']
            del new_log_item['type']
            user_log_list.append(new_log_item)
    user_log_list = sorted(user_log_list, key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M"))
    
    user_log_idx = 0
    for i, date in enumerate(dialogue_situation_dict[user_id]['history'] + dialogue_situation_dict[user_id]['query']):
        if date in dialogue_situation_dict[user_id]['history']:
            set_name = 'history'
        else:
            set_name = 'query'
        user_dict[set_name][date] = {}
        user_dict[set_name][date]['sample_id'] = "{}_sample{}".format(user_id, str(i))
        end_timestamp = situation_dict[user_id][date]['end_timestamp']
        user_dict[set_name][date]['dialogue_timestamp'] = end_timestamp
        user_dict[set_name][date]['logs'] = []
        while user_log_idx < len(user_log_list) and datetime.strptime(user_log_list[user_log_idx]["timestamp"], "%Y-%m-%d %H:%M") <= datetime.strptime(end_timestamp, "%Y-%m-%d %H:%M:%S").replace(second=0, microsecond=0):
            user_dict[set_name][date]['logs'].append(user_log_list[user_log_idx])
            user_log_idx += 1
        assert len(user_dict[set_name][date]['logs']) != 0
        user_dict[set_name][date]['dialogue'] = dialogue_dict[user_id][set_name][date]
        for turn_id in user_dict[set_name][date]['dialogue'].keys():
            for role in ['user', 'assistant']:
                user_dict[set_name][date]['dialogue'][turn_id][role].pop('reference', None)

        # user_dict[set_name][date]['user_query'] = requirement_framework_dict[user_id][date]
        user_dict[set_name][date]['topics'] = {}
        for topic_id in requirement_framework_dict[user_id][date].keys():
            user_dict[set_name][date]['topics'][topic_id] = {}
            user_dict[set_name][date]['topics'][topic_id]['user_query'] = requirement_framework_dict[user_id][date][topic_id]['user_query']
            user_dict[set_name][date]['topics'][topic_id]['implicit_needs'] = [i["need"] for i in requirement_framework_dict[user_id][date][topic_id]['implicit_needs']]
            user_dict[set_name][date]['topics'][topic_id]['requirement'] = requirement_framework_dict[user_id][date][topic_id]['requirement']

            user_dict[set_name][date]['topics'][topic_id]['solution'] = {'pos': [], 'neg': []}
            for polarity in ['pos', 'neg']:
                for solution_item in solution_preference_dict[user_id][date][topic_id]['{}_list'.format(polarity)]:
                    user_dict[set_name][date]['topics'][topic_id]['solution'][polarity].append(solution_item['solution'])
            
            user_dict[set_name][date]['topics'][topic_id]['candidate_solutions'] = []
            for solution in candidate_solution_dict[user_id][date][topic_id]:
                if solution in user_dict[set_name][date]['topics'][topic_id]['solution']['pos']:
                    user_dict[set_name][date]['topics'][topic_id]['candidate_solutions'].append({'solution': solution, 'feedback': 'pos'})
                elif solution in user_dict[set_name][date]['topics'][topic_id]['solution']['neg']:
                    user_dict[set_name][date]['topics'][topic_id]['candidate_solutions'].append({'solution': solution, 'feedback': 'neg'})
                else:
                    user_dict[set_name][date]['topics'][topic_id]['candidate_solutions'].append({'solution': solution, 'feedback': 'neu'})

            # del user_dict[set_name][date]['topics'][topic_id]['solution']

            # check
            pos_num = 0
            neg_num = 0
            for i in user_dict[set_name][date]['topics'][topic_id]['candidate_solutions']:
                if i['feedback'] == 'pos':
                    pos_num += 1
                elif i['feedback'] == 'neg':
                    neg_num += 1
            assert pos_num == 2 and neg_num == 2

    save_dict[user_id] = user_dict
    
# with open(save_path, "w") as f:
#     json.dump(save_dict, f, ensure_ascii=False, indent=4)
new_save_dict = {}
for user_id in save_dict.keys():
    new_save_dict[user_id] = {}
    for set_name in ['history', 'query']:
        new_save_dict[user_id][set_name] = []
        for sample_id in save_dict[user_id][set_name].keys():
            new_save_dict[user_id][set_name].append(save_dict[user_id][set_name][sample_id])
with open(save_path, "w") as f:
    json.dump(new_save_dict, f, ensure_ascii=False, indent=4)    



