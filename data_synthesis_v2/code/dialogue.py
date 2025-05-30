"""
第6步：根据对话框架，生成user-assistant对话内容。
每个user的数据需要分成history和query两部分。query对应于最后一个月的情境，对话内容中仅包含正向的方案，且假定assistant已经充分了解用户个性化信息；
history可能同时包含正向和负向的方案，对应于最后一个月之前的情境，且假定assistant不了解用户的个性化信息。
"""
import os
import json
import requests
import csv
from tqdm import tqdm
import re
import ast
import random
from copy import copy, deepcopy
from llm_generation import LLM_Sequential_Generation


class DialogueGeneration(LLM_Sequential_Generation):
    def __init__(self, data_type, background_dict, situation_dict, dialogue_situation_ids_dict, requirement_framework_dict, solution_preference_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert data_type in ['history', 'query']
        self.data_type = data_type
        self.sample_id_list = self.preprocess_dataset(dialogue_situation_ids_dict, start_user_id, end_user_id)
        self.background_dict = background_dict
        self.situation_dict = situation_dict
        self.requirement_framework_dict = requirement_framework_dict
        self.solution_preference_dict = solution_preference_dict
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)
        self.all_dialogue_template_dict = {}


    def preprocess_dataset(self, dialogue_situation_ids_dict, start_user_id, end_user_id):
        all_user_ids = list(dialogue_situation_ids_dict.keys())
        if start_user_id:
            start_user_idx = all_user_ids.index(start_user_id)
        else:
            start_user_idx = 0
        if end_user_id:
            end_user_idx = all_user_ids.index(end_user_id)
        else:
            end_user_idx = len(all_user_ids) - 1
        process_user_ids = all_user_ids[start_user_idx: end_user_idx+1]
        sample_id_list = []
        for user_id in process_user_ids:
            for date in dialogue_situation_ids_dict[user_id][self.data_type]:
                sample_id_list.append("{}_{}".format(user_id, date))
        return sample_id_list


    def make_dialogue_framework(self, user_id, date, topic_id):
        """
        需求与方案采样准则：
        - 需求：
            - 50%概率采样1个隐式需求，50%概率采样2个。
        - 偏好：
            - history部分：每个topic下面：1~3个方案。
                - 20%的概率只讨论1个方案，40%的概率讨论2个方案，40%的概率讨论3个方案
            - query部分：每个topic下面：1~2个方案
                - 50%的概率推荐1个方案，50%的概率推荐2个方案。
                - 方案一定是pos的
        """
        topic_key = "topic-{}".format(topic_id)
        topic_requirement_framework_dict = self.requirement_framework_dict[user_id][date][topic_key]
        topic_solution_preference_list = self.solution_preference_dict[user_id][date][topic_key]

        topic_framework_dict = {}
        topic_framework_dict['topic'] = topic_requirement_framework_dict["requirement"]

        # 隐式需求采样
        topic_framework_dict['requirements'] = {}
        topic_framework_dict['requirements']['T{}_Q'.format(topic_id)] = topic_requirement_framework_dict["user_query"]
        requirements_sample_num = random.choices([1, 2], weights=[0.5, 0.5], k=1)[0] # 50%概率1轮需求推测，50%概率2轮
        needs_list = topic_requirement_framework_dict["implicit_needs"][:requirements_sample_num]
        for i in range(requirements_sample_num):
            topic_framework_dict['requirements']['T{}_N{}'.format(topic_id, i+1)] = "<T{}_N{}>".format(topic_id, i+1)

        # 方案采样
        topic_framework_dict['solutions'] = {}
        if self.data_type == 'history':
            solution_list = []
            for polarity in ['pos', 'neg']:
                for solution_item in topic_solution_preference_list["{}_list".format(polarity)]:
                    solution_list.append({"solution": solution_item["solution"], "feedback": polarity, "reason": solution_item["feedback_reason"]})
            solution_sample_num = random.choices([1, 2, 3], weights=[0.2, 0.4, 0.4], k=1)[0] # 20%的概率采样1个方案，40%的概率采样2个方案，40%的概率采样3个方案
            sampled_solution_list = random.sample(solution_list, solution_sample_num)
        else:
            pos_solution_list = []
            for solution_item in topic_solution_preference_list["pos_list"]:
                pos_solution_list.append({"solution": solution_item["solution"], "feedback": "pos", "reason": solution_item["feedback_reason"]})
            solution_sample_num = random.choices([1, 2], weights=[0.5, 0.5], k=1)[0] # 50%的概率采样1个方案，50%的概率采样2个方案。
            sampled_solution_list = random.sample(pos_solution_list, solution_sample_num)
        for i in range(len(sampled_solution_list)):
            topic_framework_dict['solutions']['T{}_S{}'.format(topic_id, i+1)] = "<T{}_S{}>".format(topic_id, i+1)

        return topic_framework_dict, needs_list, sampled_solution_list


    def get_prompt(self, sample_id):
        user_id, date = sample_id.split('_')

        personality_dict = copy(self.background_dict[user_id]['personality'])
        background_dict = copy(self.background_dict[user_id])
        del background_dict['personality']
        background_str = json.dumps(background_dict, ensure_ascii=False)
        personality_str = json.dumps(personality_dict, ensure_ascii=False)
        situation_str = self.situation_dict[user_id][date]['situation']

        user_prompt_template_list = []
        tmp_user_prompt_template = deepcopy(self.user_prompt_template)
        user_prompt_template_list.append(tmp_user_prompt_template.split('<if_history_set>\n')[0])
        if self.data_type == 'history':
            output_principle_str = tmp_user_prompt_template.split('<if_history_set>\n')[-1].split('</if_history_set>\n')[0]
        else:
            output_principle_str = tmp_user_prompt_template.split('<if_query_set>\n')[-1].split('</if_query_set>\n')[0]
        user_prompt_template_list.append(output_principle_str)
        user_prompt_template_list.append(tmp_user_prompt_template.split('</if_query_set>\n')[-1])
        user_prompt_template = ''.join(user_prompt_template_list)

        dialogue_framework_dict = {}
        needs_str_dict = {}
        solutions_str_dict = {}

        # 加载对话框架
        for topic_key in self.requirement_framework_dict[user_id][date].keys():
            topic_id = topic_key.split('-')[-1]
            topic_framework_dict, needs_list, solutions_list = self.make_dialogue_framework(user_id, date, topic_id)
            dialogue_framework_dict['T{}'.format(topic_id)] = topic_framework_dict
            needs_str_dict['T{}'.format(topic_id)] = [json.dumps(need, ensure_ascii=False) for need in needs_list]
            solutions_str_dict['T{}'.format(topic_id)] = [json.dumps(solution, ensure_ascii=False) for solution in solutions_list]
        dialogue_framework_str = json.dumps(dialogue_framework_dict, indent=4, ensure_ascii=False)
        for topic_key in self.requirement_framework_dict[user_id][date].keys():
            topic_id = topic_key.split('-')[-1]
            for i, need_str in enumerate(needs_str_dict['T{}'.format(topic_id)]):
                dialogue_framework_str = dialogue_framework_str.replace("<T{}_N{}>".format(topic_id, str(i+1)), need_str)
            for i, solution_str in enumerate(solutions_str_dict['T{}'.format(topic_id)]):
                dialogue_framework_str = dialogue_framework_str.replace("<T{}_S{}>".format(topic_id, str(i+1)), solution_str)

        # 构造对话模板
        user_template_list = []
        assistant_template_list = []
        for topic_key in self.requirement_framework_dict[user_id][date].keys():
            topic_id = topic_key.split('-')[-1]
            user_template_list.append({"action": "话题询问", "reference": "T{}_Q".format(topic_id)})
            assistant_template_list.append({"action": "需求推测", "reference": "T{}_N{}".format(topic_id, 1)})
            topic_needs_num = len(needs_str_dict['T{}'.format(topic_id)])
            if topic_needs_num > 1:
                user_template_list.append({"action": "需求确认", "reference": "T{}_N{}".format(topic_id, 1)})
                assistant_template_list.append({"action": "需求推测", "reference": "T{}_N{}".format(topic_id, 2)})
            user_template_list.append({"action": "需求确认", "reference": "T{}_N{}".format(topic_id, topic_needs_num)})
            cur_solution_id = 0
            topic_solutions_num = len(solutions_str_dict['T{}'.format(topic_id)])
            while cur_solution_id < topic_solutions_num:
                assistant_template_list.append({"action": "方案提议", "reference": "T{}_S{}".format(topic_id, cur_solution_id + 1)})
                solution_disscus_turn_num = random.choices([0, 1], weights=[0.5, 0.5], k=1)[0] # 0~1轮“方案讨论”（50%概率0轮，50%概率1轮）
                for i in range(solution_disscus_turn_num):
                    user_template_list.append({"action": "方案讨论", "reference": "T{}_S{}".format(topic_id, cur_solution_id + 1)})
                    assistant_template_list.append({"action": "方案讨论", "reference": "T{}_S{}".format(topic_id, cur_solution_id + 1)})
                user_template_list.append({"action": "方案反馈", "reference": "T{}_S{}".format(topic_id, cur_solution_id + 1)})
                cur_solution_id += 1
            assistant_template_list.append({"action": "反馈回应", "reference": "T{}_S{}".format(topic_id, cur_solution_id)})
        assert len(user_template_list) == len(assistant_template_list)

        sample_dialogue_template_dict = {}
        self.all_dialogue_template_dict[sample_id] = {}
        for turn_idx in range(len(user_template_list)):
            turn_id = "turn_{}".format(str(turn_idx+1))
            sample_dialogue_template_dict[turn_id] = {"user": "<user_{}>".format(turn_id), "assistant": "<assistant_{}>".format(turn_id)}
            self.all_dialogue_template_dict[sample_id][turn_id] = {"user": user_template_list[turn_idx], "assistant": assistant_template_list[turn_idx]}
        dialogue_template_str = json.dumps(sample_dialogue_template_dict, indent=4, ensure_ascii=False)
        for turn_idx in range(len(user_template_list)):
            turn_id = "turn_{}".format(str(turn_idx+1))
            dialogue_template_str = dialogue_template_str.replace("\"<user_{}>\"".format(turn_id), json.dumps(user_template_list[turn_idx], ensure_ascii=False)).replace("\"<assistant_{}>\"".format(turn_id), json.dumps(assistant_template_list[turn_idx], ensure_ascii=False))
        
        system_prompt = self.system_prompt_template
        user_prompt = user_prompt_template.replace('<background>', background_str).replace('<personality>', personality_str).replace('<situation>', situation_str).replace('<dialogue_framework>', dialogue_framework_str).replace('<dialogue_template>', dialogue_template_str)

        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        raw_generation = self.generation_postprocess(raw_generation)
        try: # 确保生成内容符合json格式
            generation_json = json.loads(raw_generation)
            # 1. 检查turn_id
            if list(generation_json.keys()) != list(self.all_dialogue_template_dict[sample_id].keys()):
                return False, "wrong turn_id"
            for turn_id in self.all_dialogue_template_dict[sample_id].keys():
                # 2. 确定每个turn中有且仅有"user"和"assistant"（且注意顺序）
                if list(generation_json[turn_id].keys()) != ['user', 'assistant']:
                    return False, "wrong role_id in {}".format(turn_id)
                # 3. 每个turn中的"action"和"reference"是否正确
                if generation_json[turn_id]['user']['action'] != self.all_dialogue_template_dict[sample_id][turn_id]['user']['action'] or generation_json[turn_id]['assistant']['action'] != self.all_dialogue_template_dict[sample_id][turn_id]['assistant']['action']:
                    return False, "action wrong in {}".format(turn_id)
                if generation_json[turn_id]['user']['reference'] != self.all_dialogue_template_dict[sample_id][turn_id]['user']['reference'] or generation_json[turn_id]['assistant']['reference'] != self.all_dialogue_template_dict[sample_id][turn_id]['assistant']['reference']:
                    return False, "reference wrong in {}".format(turn_id)
                # 4. 每个turn中的content不为空
                if generation_json[turn_id]['user']['content'].split() == '' or generation_json[turn_id]['user']['content'] == '...' or generation_json[turn_id]['assistant']['content'].split() == '' or generation_json[turn_id]['assistant']['content'] == '...':
                    return False, "empty content in {}".format(turn_id)
        except json.JSONDecodeError as e:
            return False, "JSON format error"
        return True, None


    def extract_and_save(self, save_path):
        result_dict = {}
        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)
                generation_json = json.loads(generation)
                user_id, date = sample_id.split('_')
                if user_id not in result_dict.keys():
                    result_dict[user_id] = {}
                result_dict[user_id][date] = generation_json
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)


def combine_jsons(history_file_path_list, query_file_path_list, save_path):
    all_data_dict = {}
    
    for history_file_path in history_file_path_list:
        with open(history_file_path, 'r', encoding='utf-8') as f:
            history_dict = json.load(f)
        for user_id in history_dict.keys():
            if user_id not in all_data_dict.keys():
                all_data_dict[user_id] = {}
                all_data_dict[user_id]['history'] = copy(history_dict[user_id])
        
    for query_file_path in query_file_path_list:
        with open(query_file_path, 'r', encoding='utf-8') as f:
            query_dict = json.load(f)
        for user_id in query_dict.keys():
            if user_id not in all_data_dict.keys():
                all_data_dict[user_id] = {}
            if 'query' not in all_data_dict[user_id].keys():
                all_data_dict[user_id]['query'] = copy(query_dict[user_id])

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_data_dict, f, indent=4, ensure_ascii=False)

    # 验证所有user都有history和query set
    for user_id in all_data_dict.keys():
        assert 'history' in all_data_dict[user_id].keys() and 'query' in all_data_dict[user_id].keys()



if __name__ == '__main__':
    model_name = 'qwen2.5-max'

    data_synthesis_root = '/data5/hzp/MemPAL/data_synthesis_v2/data'
    prompt_template_dir = '/data5/hzp/MemPAL/data_synthesis_v2/prompt_template/dialogue'
    save_root = os.path.join(data_synthesis_root, 'dialogue')
    background_data_dir = os.path.join(data_synthesis_root, 'background')
    background_file_path = os.path.join(background_data_dir, 'background.json')
    with open(background_file_path,'r', encoding='utf-8') as f:
        background_dict = json.load(f)
    situation_data_dir = os.path.join(data_synthesis_root, 'situation')
    situation_file_path = os.path.join(situation_data_dir, 'situation.json')
    dialogue_situation_ids_file_path = os.path.join(situation_data_dir, 'dialogue_situation_ids.json')
    with open(situation_file_path,'r', encoding='utf-8') as f:
        situation_dict = json.load(f)
    with open(dialogue_situation_ids_file_path,'r', encoding='utf-8') as f:
        dialogue_situation_ids_dict = json.load(f)
    requirement_framework_data_dir = os.path.join(data_synthesis_root, 'dialogue_framework', 'requirement')
    requirement_framework_file_path = os.path.join(requirement_framework_data_dir, 'requirement_framework.json')
    with open(requirement_framework_file_path,'r', encoding='utf-8') as f:
        requirement_framework_dict = json.load(f)
    solution_preference_data_dir = os.path.join(data_synthesis_root, 'dialogue_framework', 'solution_preference')
    solution_preference_file_path = os.path.join(solution_preference_data_dir, 'solution_preference.json')
    with open(solution_preference_file_path,'r', encoding='utf-8') as f:
        solution_preference_dict = json.load(f)


    ### ----- Step 1: history部分对话生成 -----
    data_type = 'history'
    save_dir = os.path.join(save_root, 'history')
    history_dialogue_file_path = 'history_dialogue.json'
    history_dialogue_generator = DialogueGeneration(data_type, background_dict, situation_dict, dialogue_situation_ids_dict, requirement_framework_dict, solution_preference_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = history_dialogue_generator.get_prompt('0000_2024-02-25')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    history_dialogue_generator.sequential_generate()
    history_dialogue_generator.extract_and_save(history_dialogue_file_path)
    # ### ---------------------------


    ### ----- Step 2: query部分对话生成 -----
    data_type = 'query'
    save_dir = os.path.join(save_root, 'query')
    query_dialogue_file_path = 'query_dialogue.json'
    query_dialogue_generator = DialogueGeneration(data_type, background_dict, situation_dict, dialogue_situation_ids_dict, requirement_framework_dict, solution_preference_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = query_dialogue_generator.get_prompt('0000_2024-12-07')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    query_dialogue_generator.sequential_generate()
    query_dialogue_generator.extract_and_save(query_dialogue_file_path)
    ### ---------------------------