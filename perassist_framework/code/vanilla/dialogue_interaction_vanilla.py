import os
import json
import ast
import csv
import random
import torch
import numpy as np
from tqdm import tqdm
from copy import copy, deepcopy

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')

from dialogue_evaluation.user_assistant_dialogue import User_Assistant_Dialogue


class User_Assistant_Dialogue_Vanilla(User_Assistant_Dialogue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_assistant_prompt(self, sample_id):
        assistant_prompt_template_dir = os.path.join(self.framework_root, 'prompt_template', 'vanilla', 'dialogue_interaction')

        user_id = sample_id.split('_')[0]
        topic_sample_id = '_'.join(sample_id.split('_')[:-1])
        cur_turn_id = 'turn_{}'.format(sample_id.split('_')[-1])
        cur_action = self.dialogue_template_dict[topic_sample_id][cur_turn_id]['assistant']
        assert cur_action in ['<需求推测>', '<方案提议>', '<方案讨论>', '<反馈回应>']
        
        with open(os.path.join(assistant_prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            system_prompt_template = ''.join(lines)
        system_prompt = copy(system_prompt_template)

        with open(os.path.join(assistant_prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            user_prompt_template = ''.join(lines)

        topic_item = self.dataset_dict[user_id][topic_sample_id]
        logs = ["- [{}] {}".format(log_item['timestamp'], log_item['content']) for log_item in topic_item['logs']]
        logs_str = "\n".join(logs)

        dialogue_context_list = []
        current_turn_list = []
        for i in range(1, int(cur_turn_id.split('_')[-1])):
            turn_id = "turn_{}".format(i)
            for role in ['user', 'assistant']:
                dialogue_context_list.append("- \"{}\": \"{}\"".format(role, self.interaction_context_dict[topic_sample_id][turn_id][role]))
        dialogue_context_list.append("- \"{}\": \"{}\"".format('user', self.interaction_context_dict[topic_sample_id][cur_turn_id]['user']))
        current_turn_list.append("- \"{}\": \"{}\"".format('user', self.interaction_context_dict[topic_sample_id][cur_turn_id]['user']))
        dialogue_context_list.append("- \"{}\": {}".format('assistant', cur_action))
        current_turn_list.append("- \"{}\": {}".format('assistant', cur_action))
        dialogue_context_str = '\n'.join(dialogue_context_list)
        current_turn_str = '\n'.join(current_turn_list)

        user_prompt = user_prompt_template.replace('<logs>', logs_str).replace('<dialogue_context>', dialogue_context_str).replace('<current_turn>', current_turn_str).replace('<action>', "{}".format(cur_action)).replace('<action_description>', self.assistant_action_description_dict[cur_action])
        return system_prompt, user_prompt



if __name__ == '__main__':
    user_model = 'qwen2.5-max'
    assistant_model = 'qwen_max'
    model_dir_name = assistant_model


    # -----------------
    framework_root = '/data5/hzp/MemPAL/perassist_framework'
    data_dir = assistant_model
    vanilla_save_dir = os.path.join(data_dir, 'vanilla', 'dialogue_interaction')

    dialogue_save_path = 'dialogue.json'
    dialogue_generator = User_Assistant_Dialogue_Vanilla(user_model_name=user_model, assistant_model_name=assistant_model, save_dir=vanilla_save_dir)

    dialogue_generator.interaction_generate()
    # dialogue_generator.interaction_generate(record_prompt=True)
    dialogue_generator.extract_and_save(dialogue_save_path)
    # -----------------