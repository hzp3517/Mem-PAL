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

from llm_generation import LLM_Sequential_Generation, User_Assistant_Interaction_Generation


class User_Assistant_Dialogue(User_Assistant_Interaction_Generation):
    def __init__(self, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_path = '/data5/hzp/MemPAL/data_synthesis_v2/data/input.json'
        self.data_synthesis_root = '/data5/hzp/MemPAL/data_synthesis_v2'
        self.framework_root = '/data5/hzp/MemPAL/perassist_framework'
        self.model_dir_name = self.assistant_model_name
        self.dialogue_evaluation_dir = os.path.join(self.framework_root, 'code', 'dialogue_evaluation')
        dialogue_template_save_path = os.path.join(self.dialogue_evaluation_dir, 'dialogue_template.json')

        self.dataset_dict = self.preprocess_dataset(start_user_id, end_user_id)
        self.dialogue_template_dict, self.sample_id_list = self.get_dialogue_template(dialogue_template_save_path)

        self.user_action_description_dict = {
            "<话题询问>": "用户向助手提出自己关于当前话题的初始询问，应该根据输入中“用户当前需求”部分\"user_query\"的内容生成。可以适当对该内容做调整以适应对话场景，但不要改变其主要语义；且应保持较为简短且模糊化的初始询问形式，不要包含隐式需求以及相关背景经历细节。",
            "<需求确认>": "用户对助手当前轮次对于用户需求的推测进行有效回应，确认自己的真实需求。应结合输入中“用户当前需求”部分列举的隐式需求以及相关背景经历，判断助手推测的准确性。如果助手对于用户当前需求的推测正确，则给出简单的正面回应；如果助手的推测存在错误，则给出负面回应，并针对助手推测错误的内容进行简短的反馈和澄清，但不要主动向助手交代其他任何未涉及到的隐式需求和相关背景内容。",
            "<方案讨论>": "针对助手给出的方案提议，用户暂不对于方案整体给出明确的正面或负面评价，仅针对其中的细节与助手进行讨论。但不要主动向助手提及自身对于方案的偏好。",
            "<方案反馈>": "用户针对助手当前提出或讨论的方案，表达出明显的正面认同态度或负面否定态度。应该根据输入中“用户当前需求与相关偏好”部分提及的用户偏好描述判断用户对于当前方案的态度，但在生成的语句中不要主动交代自身的详细偏好，而是仅针对当前方案进行反馈。"
        }
        self.assistant_action_description_dict = {
            "<需求推测>": "助手在回复中结合了解到的用户近期情境经历或总体背景，主动推测用户询问背后的隐式需求，并向用户寻求对于其实际需求的确认。注意：每次回复只推测一方面隐式需求并向用户寻求确认即可，不需要面面俱到。另外，在当前回复中不要涉及对于解决方案的讨论，仅关注于用户实际需求的推测。",
            "<方案提议>": "助手针对当前用户需求，给出一种符合用户个性化偏好的解决方案作为建议。如果之前已经讨论过某些方案，则继续提出一种新的方案。注意：在回复中直接给出解决方案内容即可，不需要显式提及自己对于用户需求或偏好的理解。",
            "<方案讨论>": "助手回应用户关于当前方案的意见或询问，结合了解到的用户背景或相关经历与用户进一步讨论该方案，并尝试说服用户接受该方案。",
            "<反馈回应>": "助手对于用户表达出的反馈给出简短的回应。注意：在这里不需要尝试给出其他解决方案，尤其是在用户表达出负面态度的情况下，仅需对用户表示歉意或遗憾即可，不需要推荐其他方案。"
        }


    def preprocess_dataset(self, start_user_id, end_user_id):
        synthesis_data_dir = os.path.join(self.data_synthesis_root, 'data')

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset_dict = json.load(f)
        situation_path = os.path.join(synthesis_data_dir, 'situation', 'situation.json')
        with open(situation_path, 'r', encoding='utf-8') as f:
            situation_dict = json.load(f)
        preference_path = os.path.join(synthesis_data_dir, 'preference', 'preference.json')
        with open(preference_path, 'r', encoding='utf-8') as f:
            preference_dict = json.load(f)
        requirement_framework_path = os.path.join(synthesis_data_dir, 'dialogue_framework', 'requirement', 'requirement_framework.json')
        with open(requirement_framework_path, 'r', encoding='utf-8') as f:
            requirement_framework_dict = json.load(f)
        solution_preference_path = os.path.join(synthesis_data_dir, 'dialogue_framework', 'solution_preference', 'solution_preference.json')
        with open(solution_preference_path, 'r', encoding='utf-8') as f:
            solution_preference_dict = json.load(f)
        
        processed_dataset_dict = {}

        all_user_ids = list(dataset_dict.keys())
        if start_user_id:
            start_user_idx = all_user_ids.index(start_user_id)
        else:
            start_user_idx = 0
        if end_user_id:
            end_user_idx = all_user_ids.index(end_user_id)
        else:
            end_user_idx = len(all_user_ids) - 1
        process_user_ids = all_user_ids[start_user_idx: end_user_idx+1]

        for user_id in process_user_ids:
            processed_dataset_dict[user_id] = {}
            for dialogue_item in dataset_dict[user_id]['query']:
                dialogue_id = dialogue_item['sample_id']
                for topic_id in dialogue_item['topics'].keys():
                    topic_idx = int(topic_id.split('-')[-1]) - 1
                    topic_sample_id = "{}_{}".format(dialogue_id, topic_id) # e.g. 0000_sample0_topic-1
                    # sample_id_list.append(sample_id)
                    processed_dataset_dict[user_id][topic_sample_id] = {}
                    date = dialogue_item['dialogue_timestamp'].split(' ')[0]
                    general_requirement_id = situation_dict[user_id][date]['requirement_ids'][topic_idx]
                    requirement_type = general_requirement_id.split('-')[0]

                    processed_dataset_dict[user_id][topic_sample_id]['situation'] = situation_dict[user_id][date]['situation']
                    processed_dataset_dict[user_id][topic_sample_id]['user_query'] = requirement_framework_dict[user_id][date][topic_id]['user_query']
                    processed_dataset_dict[user_id][topic_sample_id]['implicit_needs'] = requirement_framework_dict[user_id][date][topic_id]['implicit_needs']
                    processed_dataset_dict[user_id][topic_sample_id]['requirement'] = requirement_framework_dict[user_id][date][topic_id]['requirement']
                    processed_dataset_dict[user_id][topic_sample_id]['general_preference'] = preference_dict[user_id][requirement_type][general_requirement_id]['preference']
                    processed_dataset_dict[user_id][topic_sample_id]['candidate_solutions'] = {}
                    processed_dataset_dict[user_id][topic_sample_id]['candidate_solutions']['pos_list'] = solution_preference_dict[user_id][date][topic_id]['pos_list']
                    processed_dataset_dict[user_id][topic_sample_id]['candidate_solutions']['neg_list'] = solution_preference_dict[user_id][date][topic_id]['neg_list']

                    processed_dataset_dict[user_id][topic_sample_id]['logs'] = dialogue_item['logs']

        return processed_dataset_dict


    def get_dialogue_template(self, dialogue_template_save_path):
        if os.path.exists(dialogue_template_save_path):
            print('Load dialogue template from exist files.')
            with open(dialogue_template_save_path, 'r', encoding='utf-8') as f:
                dialogue_template_dict = json.load(f)
        else:
            print('Sampling dialogue template now...')
            dialogue_template_dict = {}
            for user_id in self.dataset_dict.keys():
                for topic_sample_id in self.dataset_dict[user_id].keys():
                    dialogue_template_dict[topic_sample_id] = {}
                    user_template_list = []
                    assistant_template_list = []
                    user_template_list.append('<话题询问>')
                    for i in range(2):
                        assistant_template_list.append('<需求推测>')
                        user_template_list.append('<需求确认>')
                    for i in range(2):
                        assistant_template_list.append('<方案提议>')
                        solution_disscus_turn_num = random.choices([0, 1], weights=[0.5, 0.5], k=1)[0] # 0~1轮“方案讨论”（50%概率0轮，50%概率1轮）
                        for i in range(solution_disscus_turn_num):
                            user_template_list.append('<方案讨论>')
                            assistant_template_list.append('<方案讨论>')
                        user_template_list.append('<方案反馈>')
                    assistant_template_list.append('<反馈回应>')
                    assert len(user_template_list) == len(assistant_template_list)
                    for turn_idx in range(len(user_template_list)):
                        turn_id = "turn_{}".format(str(turn_idx+1))
                        dialogue_template_dict[topic_sample_id][turn_id] = {"user": user_template_list[turn_idx], "assistant": assistant_template_list[turn_idx]}
            with open(dialogue_template_save_path, 'w', encoding='utf-8') as f:
                json.dump(dialogue_template_dict, f, indent=4, ensure_ascii=False)

        sample_id_list = []
        for topic_sample_id in dialogue_template_dict.keys():
            user_id = topic_sample_id.split('_')[0]
            if user_id not in self.dataset_dict.keys():
                continue
            for turn_id in dialogue_template_dict[topic_sample_id].keys():
                turn_idx = turn_id.split('_')[-1]
                turn_sample_id = "{}_{}".format(topic_sample_id, turn_idx)
                sample_id_list.append(turn_sample_id)
        
        return dialogue_template_dict, sample_id_list


    def load_user_input(self):
        background_path = os.path.join(self.data_synthesis_root, 'data', 'background', 'background.json')
        with open(background_path, 'r', encoding='utf-8') as f:
            background_dict = json.load(f)
        return background_dict


    def get_user_prompt(self, sample_id):
        user_id = sample_id.split('_')[0]
        topic_sample_id = '_'.join(sample_id.split('_')[:-1])
        cur_turn_id = 'turn_{}'.format(sample_id.split('_')[-1])
        cur_action = self.dialogue_template_dict[topic_sample_id][cur_turn_id]['user']
        assert cur_action in ['<话题询问>', '<需求确认>', '<方案讨论>', '<方案反馈>']

        background_dict = self.load_user_input()

        user_prompt_template_dir = os.path.join(self.dialogue_evaluation_dir, 'user_llm_prompt')
        with open(os.path.join(user_prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            system_prompt_template = ''.join(lines)
        system_prompt = copy(system_prompt_template)

        with open(os.path.join(user_prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            user_prompt_template = ''.join(lines)

        sample_personality_dict = copy(background_dict[user_id]['personality'])
        sample_background_dict = copy(background_dict[user_id])
        topic_data_dict = self.dataset_dict[user_id][topic_sample_id]
        del sample_background_dict['personality']
        background_str = json.dumps(sample_background_dict, ensure_ascii=False)
        personality_str = json.dumps(sample_personality_dict, ensure_ascii=False)
        situation_str = json.dumps(topic_data_dict['situation'], ensure_ascii=False)

        common_persona_str = "## 用户背景\n\"\"\"\n<background>\n\"\"\"\n\n## 用户性格\n\"\"\"\n<personality>\n\"\"\"\n\n## 用户近期情境\n\"\"\"\n<situation>\n\"\"\"\n\n".replace('<background>', background_str).replace('<personality>', personality_str).replace('<situation>', situation_str)
        if cur_action in ['<话题询问>', '<需求确认>']: # 需求讨论阶段
            persona_template = common_persona_str + "## 用户当前需求\n以下是用户当前需求的相关信息，其中\"user_query\"是用户的初始询问，\"implicit_needs\"是用户当前的隐式需求以及相关的用户背景或经历，\"requirement\"是对上述两部分的总结，即用户当前的实际需求。\n\"\"\"\n<requirement>\n\"\"\""
            requirement_dict = {
                'user_query': topic_data_dict['user_query'],
                'implicit_needs': ['<need_dict_{}>'.format(str(i)) for i in range(1, len(topic_data_dict['implicit_needs'])+1)],
                'requirement': topic_data_dict['requirement']
            }
            requirement_str = json.dumps(requirement_dict, indent=4, ensure_ascii=False)
            for i in range(1, len(topic_data_dict['implicit_needs'])+1):
                requirement_str = requirement_str.replace('<need_dict_{}>'.format(str(i)), json.dumps(topic_data_dict['implicit_needs'][i-1], ensure_ascii=False))
            persona_str = persona_template.replace('<requirement>', requirement_str)
        else: # 方案讨论阶段
            persona_template = common_persona_str + "## 用户当前需求与相关偏好\n以下信息中，\"requirement\"是当前用户的具体需求描述；\"general_preference\"为相关需求类型下用户的总体偏好描述，分为\"pos\"和\"neg\"两部分，分别概括了该用户喜欢和不喜欢的方案类型；\"candidate_solutions\"中的\"pos_list\"和\"neg_list\"则分别列举了当前需求下符合和不符合用户偏好的各2条具体方案。\n\"\"\"\n<solutions>\n\"\"\""
            solutions_dict = {
                'requirement': topic_data_dict['requirement'],
                'general_preference': topic_data_dict['general_preference'],
                'candidate_solutions': {
                    'pos_list': ['<pos_solution_dict_{}>'.format(str(i)) for i in range(1, len(topic_data_dict['candidate_solutions']['pos_list'])+1)],
                    'neg_list': ['<neg_solution_dict_{}>'.format(str(i)) for i in range(1, len(topic_data_dict['candidate_solutions']['neg_list'])+1)]
                }
            }
            solutions_str = json.dumps(solutions_dict, indent=4, ensure_ascii=False)
            for polarity in ['pos', 'neg']:
                for i in range(1, len(topic_data_dict['candidate_solutions']['{}_list'.format(polarity)])+1):
                    solutions_str = solutions_str.replace('<{}_solution_dict_{}>'.format(polarity, str(i)), json.dumps(topic_data_dict['candidate_solutions']['{}_list'.format(polarity)][i-1], ensure_ascii=False))
            persona_str = persona_template.replace('<solutions>', solutions_str)
        
        dialogue_context_list = []
        current_turn_list = []
        for i in range(1, int(cur_turn_id.split('_')[-1])):
            turn_id = "turn_{}".format(str(i))
            for role in ['user', 'assistant']:
                dialogue_context_list.append("- \"{}\": \"{}\"".format(role, self.interaction_context_dict[topic_sample_id][turn_id][role]))
        if cur_turn_id != 'turn_1':
            last_turn_id = "turn_{}".format(str(int(cur_turn_id.split('_')[-1]) - 1))
            current_turn_list.append("- \"{}\": \"{}\"".format('assistant', self.interaction_context_dict[topic_sample_id][last_turn_id]['assistant']))
        dialogue_context_list.append("- \"{}\": {}".format('user', cur_action))
        current_turn_list.append("- \"{}\": {}".format('user', cur_action))
        dialogue_context_str = '\n'.join(dialogue_context_list)
        current_turn_str = '\n'.join(current_turn_list)

        user_prompt = user_prompt_template.replace('<persona>', persona_str).replace('<dialogue_context>', dialogue_context_str).replace('<current_turn>', current_turn_str).replace('<action>', cur_action).replace('<action_description>', self.user_action_description_dict[cur_action])

        return system_prompt, user_prompt


    def load_assistant_input(self):
        data_dir = os.path.join(self.framework_root, 'data', self.model_dir_name)


    def get_assistant_prompt(self, sample_id):
        '''
        对于不同待测模型，需要重写该函数
        '''
        user_id = sample_id.split('_')[0]
        topic_sample_id = '_'.join(sample_id.split('_')[:-1])
        cur_turn_id = 'turn_{}'.format(sample_id.split('_')[-1])
        cur_action = self.dialogue_template_dict[topic_sample_id][cur_turn_id]['assistant']
        assert cur_action in ['<需求推测>', '<方案提议>', '<方案讨论>', '<反馈回应>']

        # xxx = self.load_assistant_input()

        assistant_prompt_template_dir = os.path.join(self.dialogue_evaluation_dir, 'assistant_llm_prompt_vanilla_wolog')
        with open(os.path.join(assistant_prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            system_prompt_template = ''.join(lines)
        system_prompt = copy(system_prompt_template)

        with open(os.path.join(assistant_prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            user_prompt_template = ''.join(lines)

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

        user_prompt = user_prompt_template.replace('<dialogue_context>', dialogue_context_str).replace('<current_turn>', current_turn_str).replace('<action>', "{}".format(cur_action)).replace('<action_description>', self.assistant_action_description_dict[cur_action])
        return system_prompt, user_prompt


    def check_user_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['content']:
                return False, "key error"
            if generation_dict['content'] == "..." or generation_dict['content'].strip() == "":
                return False, "empty response"
        except:
            return False, "json format error"
        return True, None


    def check_assistant_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['content']:
                return False, "key error"
            if generation_dict['content'] == "..." or generation_dict['content'].strip() == "":
                return False, "empty response"
        except:
            return False, "json format error"
        return True, None


    def update_interaction_context(self, role, sample_id, success, generation):
        '''
        {
            "0000_sample30_topic-1": {
                "turn_1": {
                    "user": "...",
                    "assistant": "..."
                },
            }
        }
        '''
        user_id = sample_id.split('_')[0]
        topic_sample_id = '_'.join(sample_id.split('_')[:-1])
        cur_turn_id = 'turn_{}'.format(sample_id.split('_')[-1])
        cur_action = self.dialogue_template_dict[topic_sample_id][cur_turn_id][role]

        if success:
            generation = self.generation_postprocess(generation)
            generation_json = json.loads(generation)
            if topic_sample_id not in self.interaction_context_dict.keys():
                self.interaction_context_dict[topic_sample_id] = {}
            if cur_turn_id not in self.interaction_context_dict[topic_sample_id].keys():
                self.interaction_context_dict[topic_sample_id][cur_turn_id] = {}
            self.interaction_context_dict[topic_sample_id][cur_turn_id][role] = deepcopy(generation_json['content'])
        else:
            self.interaction_context_dict[topic_sample_id][cur_turn_id][role] = deepcopy(cur_action)


    def extract_and_save(self, save_path, check_turn_num=False):
        result_dict = {}
        for topic_sample_id in self.interaction_context_dict.keys():
            user_id = topic_sample_id.split('_')[0]
            if user_id not in result_dict.keys():
                result_dict[user_id] = {}
            result_dict[user_id][topic_sample_id] = deepcopy(self.interaction_context_dict[topic_sample_id])

        if check_turn_num:
            for user_id in self.dataset_dict.keys():
                assert user_id in result_dict.keys()
                for topic_sample_id in self.dataset_dict[user_id].keys():
                    assert topic_sample_id in result_dict[user_id].keys()
                    assert len(self.dialogue_template_dict[topic_sample_id]) == len(result_dict[user_id][topic_sample_id])
        
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

        


if __name__ == '__main__':
    user_model = 'qwen2.5-max'
    assistant_model = 'qwen_max'
    model_dir_name = 'gpt4o' if assistant_model == 'onechat_gpt4o' else assistant_model


    framework_root = '/data5/hzp/MemPAL/perassist_framework'
    data_dir = assistant_model
    vanilla_wolog_save_dir = os.path.join(data_dir, 'vanilla_wolog', 'dialogue_interaction')

    dialogue_save_path = 'dialogue.json'
    dialogue_generator = User_Assistant_Dialogue(user_model_name=user_model, assistant_model_name=assistant_model, save_dir=vanilla_wolog_save_dir)

    dialogue_generator.interaction_generate()
    # dialogue_generator.interaction_generate(record_prompt=True)
    dialogue_generator.extract_and_save(dialogue_save_path)