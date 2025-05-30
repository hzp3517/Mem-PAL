"""
第5步：合成用户-助手对话的框架（包含需求部分和方案部分）
"""
import os
import json
import requests
import csv
from tqdm import tqdm
import re
import ast
import random
from copy import copy
from llm_generation import LLM_Sequential_Generation


class RequirementFrameworkGeneration(LLM_Sequential_Generation):
    def __init__(self, background_dict, requirement_dict, situation_dict, dialogue_situation_ids_dict, experience_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in background_dict.keys():
            background = background_dict[k]
            del background['personality']
        self.sample_id_list = self.preprocess_dataset(dialogue_situation_ids_dict, start_user_id, end_user_id)
        self.background_dict = background_dict
        self.requirement_dict = requirement_dict
        self.situation_dict = situation_dict
        self.experience_dict = experience_dict
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)


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
            for set_name in ['history', 'query']:
                for date in dialogue_situation_ids_dict[user_id][set_name]:
                    sample_id_list.append("{}_{}".format(user_id, date))
        return sample_id_list


    def get_prompt(self, sample_id):
        user_id, date = sample_id.split('_')
        requirement_ids = self.situation_dict[user_id][date]['requirement_ids']

        background_dict = self.background_dict[user_id]
        background_str = json.dumps(background_dict, ensure_ascii=False)
        situation_str = self.situation_dict[user_id][date]['situation']
        experience_str = self.experience_dict[user_id][date]['experience']

        requirement_framework = {}
        for i, requirement_id in enumerate(requirement_ids):
            topic_id = 'topic-{}'.format(str(i+1))
            requirement_framework[topic_id] = {}
            aspect = requirement_id.split('-')[0]
            requirement_framework[topic_id]['requirement_type'] = self.requirement_dict[user_id][aspect][requirement_id]
            requirement_framework[topic_id]['user_query'] = "..."
            requirement_framework[topic_id]['implicit_needs'] = ["<implicit_needs>"]
            requirement_framework[topic_id]['requirement'] = "..."
        requirement_framework_str = json.dumps(requirement_framework, indent=4, ensure_ascii=False)
        implicit_needs_str = (',\n' + " "*4*3).join(["{\"need\": \"...\", \"evidences\": [\"...\", ...]}"] * 2)
        requirement_framework_str = requirement_framework_str.replace("\"<implicit_needs>\"", implicit_needs_str)

        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.replace('<background>', background_str).replace('<situation>', situation_str).replace('<experience>', experience_str).replace('<output_template>', requirement_framework_str)
        
        return system_prompt, user_prompt
        

    def check_generation(self, sample_id, raw_generation):
        raw_generation = self.generation_postprocess(raw_generation)

        user_id, date = sample_id.split('_')
        requirement_ids = self.situation_dict[user_id][date]['requirement_ids']

        try:
            generation_json = json.loads(raw_generation)
            if list(generation_json.keys()) != ['topic-{}'.format(str(i+1)) for i in range(len(requirement_ids))]:
                return False, "key error"
            for i, requirement_id in enumerate(requirement_ids):
                aspect = requirement_id.split('-')[0]
                topic_id = 'topic-{}'.format(str(i+1))
                if list(generation_json[topic_id].keys()) != ["requirement_type", "user_query", "implicit_needs", "requirement"]:
                    return False, "key error"
                if generation_json[topic_id]['requirement_type'] != self.requirement_dict[user_id][aspect][requirement_id]:
                    return False, "requirement_type error"
                if generation_json[topic_id]['user_query'] == "..." or generation_json[topic_id]['user_query'].strip() == "":
                    return False, "empty user_query"
                if not isinstance(generation_json[topic_id]['implicit_needs'], list):
                    return False, "implicit_needs format error"
                for implicit_needs_item in generation_json[topic_id]['implicit_needs']:
                    if list(implicit_needs_item.keys()) != ['need', 'evidences']:
                        return False, "implicit_needs key error"
                    if implicit_needs_item['need'] == "..." or implicit_needs_item['need'].strip() == "":
                        return False, "empty need"
                    if not isinstance(implicit_needs_item['evidences'], list):
                        return False, "evidences format error"
                    if len(implicit_needs_item['evidences']) < 1 or len(implicit_needs_item['evidences']) > 5:
                        return False, "too much or too few evidences"
                if generation_json[topic_id]['requirement'] == "..." or generation_json[topic_id]['requirement'].strip() == "":
                    return False, "empty requirement"
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


class CandidateSolutionGeneration(LLM_Sequential_Generation):
    def __init__(self, dialogue_situation_ids_dict, requirement_framework_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_id_list = self.preprocess_dataset(dialogue_situation_ids_dict, start_user_id, end_user_id)
        self.requirement_framework_dict = requirement_framework_dict
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)

    
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
            for set_name in ['history', 'query']:
                for date in dialogue_situation_ids_dict[user_id][set_name]:
                    sample_id_list.append("{}_{}".format(user_id, date))
        return sample_id_list


    def get_prompt(self, sample_id):
        user_id, date = sample_id.split('_')

        requirement_dict = self.requirement_framework_dict[user_id][date]
        implicit_needs_dict = {}
        for i, topic_key in enumerate(list(requirement_dict.keys())):
            del requirement_dict[topic_key]["requirement_type"]
            implicit_needs_dict[topic_key] = copy(requirement_dict[topic_key]["implicit_needs"])
            requirement_dict[topic_key]["implicit_needs"] = ["<implicit_needs_of_{}>".format(topic_key)]
        output_template = {}
        for topic_key in requirement_dict.keys():
            output_template[topic_key] = "<solutions>"

        requirement_str = json.dumps(requirement_dict, indent=4, ensure_ascii=False)
        for topic_key in requirement_dict.keys():
            implicit_needs_str = (',\n' + " "*4*3).join([json.dumps(i, ensure_ascii=False) for i in implicit_needs_dict[topic_key]])
            requirement_str = requirement_str.replace("\"<implicit_needs_of_{}>\"".format(topic_key), implicit_needs_str)
        output_template_str = json.dumps(output_template, indent=4, ensure_ascii=False)
        output_template_str = output_template_str.replace("\"<solutions>\"", "[...]")

        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.replace('<requirement>', requirement_str).replace('<output_template>', output_template_str)
        
        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        raw_generation = self.generation_postprocess(raw_generation)
        user_id, date = sample_id.split('_')
        try:
            generation_json = json.loads(raw_generation)
            if list(generation_json.keys()) != list(self.requirement_framework_dict[user_id][date].keys()):
                return False, "key error"
            for topic_id in self.requirement_framework_dict[user_id][date].keys():
                if not isinstance(generation_json[topic_id], list):
                    return False, "format error"
                if len(generation_json[topic_id]) != 8:
                    return False, "too much or too few solutions"
                for solution_item in generation_json[topic_id]:
                    if solution_item == "..." or solution_item.strip() == "":
                        return False, "empty solution"
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


class SolutionPreferenceGeneration(LLM_Sequential_Generation):
    def __init__(self, background_dict, preference_dict, situation_dict, dialogue_situation_ids_dict, requirement_framework_dict, candidate_solution_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_id_list = self.preprocess_dataset(situation_dict, dialogue_situation_ids_dict, start_user_id, end_user_id)
        self.background_dict = background_dict
        self.preference_dict = preference_dict
        self.situation_dict = situation_dict
        self.requirement_framework_dict = requirement_framework_dict
        self.candidate_solution_dict = candidate_solution_dict
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)


    def preprocess_dataset(self, situation_dict, dialogue_situation_ids_dict, start_user_id, end_user_id):
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
            for set_name in ['history', 'query']:
                for date in dialogue_situation_ids_dict[user_id][set_name]:
                    requirement_ids = situation_dict[user_id][date]["requirement_ids"]
                    for i in range(len(requirement_ids)):
                        sample_id_list.append("{}_{}_{}".format(user_id, date, str(i+1)))
        return sample_id_list
    

    def get_prompt(self, sample_id):
        user_id, date, topic_id = sample_id.split('_')
        topic_key = 'topic-{}'.format(topic_id)
        requirement_id = self.situation_dict[user_id][date]['requirement_ids'][int(topic_id)-1]
        aspect = requirement_id.split('-')[0]

        personality_dict = copy(self.background_dict[user_id]['personality'])
        background_dict = copy(self.background_dict[user_id])
        del background_dict['personality']
        requirement_dict = self.requirement_framework_dict[user_id][date][topic_key]
        del requirement_dict["requirement_type"]
        implicit_needs_list = copy(requirement_dict["implicit_needs"])
        requirement_dict["implicit_needs"] = ["<implicit_needs>"]
        preference_dict = self.preference_dict[user_id][aspect][requirement_id]['preference']
        candidate_solution_list = self.candidate_solution_dict[user_id][date][topic_key]
        candidate_solutions_str = json.dumps(candidate_solution_list, indent=4, ensure_ascii=False)

        background_str = json.dumps(background_dict, ensure_ascii=False)
        personality_str = json.dumps(personality_dict, ensure_ascii=False)
        situation_str = self.situation_dict[user_id][date]['situation']
        requirement_str = json.dumps(requirement_dict, indent=4, ensure_ascii=False)
        preference_str = json.dumps(preference_dict, indent=4, ensure_ascii=False)
        implicit_needs_str = (',\n' + " "*4*2).join([json.dumps(i, ensure_ascii=False) for i in implicit_needs_list])
        requirement_str = requirement_str.replace("\"<implicit_needs>\"", implicit_needs_str)
        output_template_dict = {"analysis": "...", "pos_list": ["<solution_dict>", "<solution_dict>"], "neg_list": ["<solution_dict>", "<solution_dict>"]}
        output_template_str = json.dumps(output_template_dict, indent=4, ensure_ascii=False)
        output_template_str = output_template_str.replace("<solution_dict>", json.dumps({"solution": "...", "feedback_reason": "..."}, ensure_ascii=False))
        
        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.replace('<background>', background_str).replace('<personality>', personality_str).replace('<situation>', situation_str).replace('<requirement>', requirement_str).replace('<preference>', preference_str).replace('<candidate_solutions>', candidate_solutions_str).replace('<output_template>', output_template_str)
        
        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        raw_generation = self.generation_postprocess(raw_generation)
        user_id, date, topic_id = sample_id.split('_')
        topic_key = 'topic-{}'.format(topic_id)
        candidate_solution_list = self.candidate_solution_dict[user_id][date][topic_key]
        try:
            generation_json = json.loads(raw_generation)
            if list(generation_json.keys()) != ['analysis', 'pos_list', 'neg_list']:
                return False, "key error"
            if not isinstance(generation_json['pos_list'], list) or not isinstance(generation_json['neg_list'], list):
                return False, "list format error"
            if len(generation_json['pos_list']) != 2 or len(generation_json['neg_list']) != 2:
                return False, "too much or too few solutions"
            for solution_item in generation_json['pos_list']:
                if list(solution_item.keys()) != ['solution', 'feedback_reason']:
                    return False, "key error"
                if solution_item['solution'] not in candidate_solution_list:
                    return False, "wrong solution"
                if solution_item['feedback_reason'] == "..." or solution_item['feedback_reason'].strip() == "":
                    return False, "empty reason"
            for solution_item in generation_json['neg_list']:
                if list(solution_item.keys()) != ['solution', 'feedback_reason']:
                    return False, "key error"
                if solution_item['solution'] not in candidate_solution_list:
                    return False, "wrong solution"
                if solution_item['feedback_reason'] == "..." or solution_item['feedback_reason'].strip() == "":
                    return False, "empty reason"
        except:
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
                user_id, date, topic_id = sample_id.split('_')
                topic_key = 'topic-{}'.format(topic_id)
                if user_id not in result_dict.keys():
                    result_dict[user_id] = {}
                if date not in result_dict[user_id].keys():
                    result_dict[user_id][date] = {}
                result_dict[user_id][date][topic_key] = generation_json
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)




if __name__ == '__main__':
    model_name = 'qwen2.5-max'

    data_synthesis_root = '/data5/hzp/MemPAL/data_synthesis_v2/data'
    prompt_template_root = '/data5/hzp/MemPAL/data_synthesis_v2/prompt_template/dialogue_framework'
    save_root = os.path.join(data_synthesis_root, 'dialogue_framework')
    background_data_dir = os.path.join(data_synthesis_root, 'background')
    background_file_path = os.path.join(background_data_dir, 'background.json')
    with open(background_file_path,'r', encoding='utf-8') as f:
        background_dict = json.load(f)
    requirement_data_dir = os.path.join(data_synthesis_root, 'requirement')
    requirement_file_path = os.path.join(requirement_data_dir, 'requirement.json')
    with open(requirement_file_path,'r', encoding='utf-8') as f:
        requirement_dict = json.load(f)
    preference_data_dir = os.path.join(data_synthesis_root, 'preference')
    preference_file_path = os.path.join(preference_data_dir, 'preference.json')
    with open(preference_file_path,'r', encoding='utf-8') as f:
        preference_dict = json.load(f)
    situation_data_dir = os.path.join(data_synthesis_root, 'situation')
    situation_file_path = os.path.join(situation_data_dir, 'situation.json')
    dialogue_situation_ids_file_path = os.path.join(situation_data_dir, 'dialogue_situation_ids.json')
    with open(situation_file_path,'r', encoding='utf-8') as f:
        situation_dict = json.load(f)
    with open(dialogue_situation_ids_file_path,'r', encoding='utf-8') as f:
        dialogue_situation_ids_dict = json.load(f)
    experience_data_dir = os.path.join(data_synthesis_root, 'experience')
    experience_file_path = os.path.join(experience_data_dir, 'experience.json')
    with open(experience_file_path,'r', encoding='utf-8') as f:
        experience_dict = json.load(f)


    ### ----- Step 1: 需求框架生成 -----
    prompt_template_dir = os.path.join(prompt_template_root, 'requirement')
    save_dir = os.path.join(save_root, 'requirement')
    requirement_framework_file_path = 'requirement_framework.json'
    requirement_framework_generator = RequirementFrameworkGeneration(background_dict, requirement_dict, situation_dict, dialogue_situation_ids_dict, experience_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = requirement_framework_generator.get_prompt('0000_2024-02-25')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    requirement_framework_generator.sequential_generate()
    requirement_framework_generator.extract_and_save(requirement_framework_file_path)
    # ### ---------------------------


    ### ----- Step 2: 候选方案生成 -----
    prompt_template_dir = os.path.join(prompt_template_root, 'candidate_solution')
    requirement_framework_data_dir = os.path.join(save_root, 'requirement')
    requirement_framework_file_path = os.path.join(requirement_framework_data_dir, 'requirement_framework.json')
    with open(requirement_framework_file_path,'r', encoding='utf-8') as f:
        requirement_framework_dict = json.load(f)
    save_dir = os.path.join(save_root, 'candidate_solution')
    candidate_solution_file_path = 'candidate_solution.json'
    candidate_solution_generator = CandidateSolutionGeneration(dialogue_situation_ids_dict, requirement_framework_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = candidate_solution_generator.get_prompt('0000_2024-02-25')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    candidate_solution_generator.sequential_generate()
    candidate_solution_generator.extract_and_save(candidate_solution_file_path)
    # ### ---------------------------
    

    ### ----- Step 3: 方案偏好生成 -----
    prompt_template_dir = os.path.join(prompt_template_root, 'solution_preference')
    requirement_framework_data_dir = os.path.join(save_root, 'requirement')
    requirement_framework_file_path = os.path.join(requirement_framework_data_dir, 'requirement_framework.json')
    with open(requirement_framework_file_path,'r', encoding='utf-8') as f:
        requirement_framework_dict = json.load(f)
    candidate_solution_data_dir = os.path.join(save_root, 'candidate_solution')
    candidate_solution_file_path = os.path.join(candidate_solution_data_dir, 'candidate_solution.json')
    with open(candidate_solution_file_path,'r', encoding='utf-8') as f:
        candidate_solution_dict = json.load(f)
    save_dir = os.path.join(save_root, 'solution_preference')
    solution_preference_file_path = 'solution_preference.json'
    solution_preference_generator = SolutionPreferenceGeneration(background_dict, preference_dict, situation_dict, dialogue_situation_ids_dict, requirement_framework_dict, candidate_solution_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = solution_preference_generator.get_prompt('0000_2024-01-12_1')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    solution_preference_generator.sequential_generate()
    solution_preference_generator.extract_and_save(solution_preference_file_path)
    # ### ---------------------------