"""
第2步：合成用户的总体个性化信息（时间线、总体需求类型、总体偏好）
"""
import os
import json
import requests
import csv
from tqdm import tqdm
import pandas as pd
from copy import copy
from datetime import datetime
from dateutil.relativedelta import relativedelta
import random
from llm_generation import LLM_Sequential_Generation


class TimelineGeneration(LLM_Sequential_Generation):
    def __init__(self, background_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in background_dict.keys():
            background = background_dict[k]
            del background['personality']
        self.background_dict, self.sample_id_list = self.preprocess_dataset(background_dict, start_user_id, end_user_id)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)

    def preprocess_dataset(self, background_dict, start_user_id, end_user_id):
        all_user_ids = list(background_dict.keys())
        if start_user_id:
            start_user_idx = all_user_ids.index(start_user_id)
        else:
            start_user_idx = 0
        if end_user_id:
            end_user_idx = all_user_ids.index(end_user_id)
        else:
            end_user_idx = len(all_user_ids) - 1
        process_user_ids = all_user_ids[start_user_idx: end_user_idx+1]
        return background_dict, process_user_ids

    def add_months(self, date_str, months_to_add):
        current_date = datetime.strptime(date_str, "%Y年%m月") # 将输入的字符串转换为日期对象
        new_date = current_date + relativedelta(months=months_to_add) # 使用relativedelta进行月份的加减
        return new_date.strftime("%Y年") + str(new_date.month) + "月" # 将新的日期转换为所需格式并返回

    def get_prompt(self, sample_id):
        cur_month = '2025年1月'
        time_span = random.randint(7, 12) # user history的时间跨度（以月为单位）
        start_month = self.add_months(cur_month, -1 * time_span)
        second_month = self.add_months(start_month, 1)
        last_month = self.add_months(cur_month, -1)
        user_background = self.background_dict[sample_id]
        user_background_str = json.dumps(user_background, indent=4, ensure_ascii=False)

        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.replace('<user_background>', user_background_str).replace('<current_month>', cur_month).replace('<start_month>', start_month).replace('<second_month>', second_month).replace('<last_month>', last_month)
        return system_prompt, user_prompt
    

    def check_generation(self, sample_id, raw_generation):
        raw_generation = self.generation_postprocess(raw_generation)
        try: # 确保生成内容符合json格式
            generation_json = json.loads(raw_generation)
            month_list = [i for i in generation_json.keys()]
            for i in range(1, len(month_list)):
                if self.add_months(month_list[i-1], 1) != month_list[i]:
                    return False, "month key error"
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
                result_dict[sample_id] = generation_json
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)


class RequirementGeneration(LLM_Sequential_Generation):
    def __init__(self, background_dict, timeline_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in background_dict.keys():
            background = background_dict[k]
            del background['personality']
        self.background_dict, self.sample_id_list = self.preprocess_dataset(background_dict, start_user_id, end_user_id)
        self.timeline_dict = timeline_dict
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)

    def preprocess_dataset(self, background_dict, start_user_id, end_user_id):
        all_user_ids = list(background_dict.keys())
        if start_user_id:
            start_user_idx = all_user_ids.index(start_user_id)
        else:
            start_user_idx = 0
        if end_user_id:
            end_user_idx = all_user_ids.index(end_user_id)
        else:
            end_user_idx = len(all_user_ids) - 1
        process_user_ids = all_user_ids[start_user_idx: end_user_idx+1]
        return background_dict, process_user_ids

    def get_prompt(self, sample_id):
        user_background_str = json.dumps(self.background_dict[sample_id], ensure_ascii=False)
        timeline_str = json.dumps(self.timeline_dict[sample_id], ensure_ascii=False)

        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.replace('<user_background>', user_background_str).replace('<timeline>', timeline_str)
        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        aspect_list = ['work', 'health', 'family', 'leisure']
        raw_generation = self.generation_postprocess(raw_generation)
        try: # 确保生成内容符合json格式
            generation_json = json.loads(raw_generation)
            for aspect in aspect_list:
                if aspect not in generation_json:
                    return False, "key missing"
                if len(generation_json[aspect].keys()) < 4 or len(generation_json[aspect].keys()) > 5:
                    return False, "key error"
                cnt = 0
                for requirement_id in generation_json[aspect].keys():
                    cnt += 1
                    if requirement_id != '{}-{}'.format(aspect, str(cnt)):
                        return False, "key error"
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
                result_dict[sample_id] = generation_json
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)


class PreferenceGeneration(LLM_Sequential_Generation):
    def __init__(self, background_dict, timeline_dict, requirement_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_dict, self.sample_id_list = self.preprocess_dataset(background_dict, start_user_id, end_user_id)
        self.timeline_dict = timeline_dict
        self.requirement_dict = requirement_dict
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)


    def preprocess_dataset(self, background_dict, start_user_id, end_user_id):
        all_user_ids = list(background_dict.keys())
        if start_user_id:
            start_user_idx = all_user_ids.index(start_user_id)
        else:
            start_user_idx = 0
        if end_user_id:
            end_user_idx = all_user_ids.index(end_user_id)
        else:
            end_user_idx = len(all_user_ids) - 1
        process_user_ids = all_user_ids[start_user_idx: end_user_idx+1]

        aspect_list = ['work', 'health', 'family', 'leisure']
        sample_id_list = []
        for user_id in process_user_ids:
            for aspect in aspect_list:
                sample_id_list.append('{}_{}'.format(user_id, aspect))

        return background_dict, sample_id_list


    def get_prompt(self, sample_id):
        user_id, aspect = sample_id.split('_')
        aspect_dict = {'work': '工作', 'health': '健康', 'family': '家庭', 'leisure': '休闲'}

        personality_dict = copy(self.background_dict[user_id]['personality'])
        background_dict = copy(self.background_dict[user_id])
        del background_dict['personality']
        timeline_dict = self.timeline_dict[user_id]
        requirement_dict = self.requirement_dict[user_id][aspect]

        background_str = json.dumps(background_dict, ensure_ascii=False)
        personality_str = json.dumps(personality_dict, ensure_ascii=False)
        timeline_str = json.dumps(timeline_dict, ensure_ascii=False)
        requirement_str = json.dumps(requirement_dict, ensure_ascii=False)
        aspect_cn = aspect_dict[aspect]

        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.replace('<background>', background_str).replace('<personality>', personality_str).replace('<timeline>', timeline_str).replace('<requirement>', requirement_str).replace('<aspect>', aspect_cn)
        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        raw_generation = self.generation_postprocess(raw_generation)
        try:
            generation_json = json.loads(raw_generation)
            user_id, aspect = sample_id.split('_')
            r_id_list = list(self.requirement_dict[user_id][aspect].keys())
            gene_r_id_list = list(generation_json.keys())
            if r_id_list != gene_r_id_list:
                return False, "wrong requirement id"
            for r_id in gene_r_id_list:
                if list(generation_json[r_id].keys()) != ['requirement', 'analysis', 'preference']:
                    return False, "wrong key"
                if generation_json[r_id]['requirement'] != self.requirement_dict[user_id][aspect][r_id]:
                    return False, "wrong requirement value"
                if generation_json[r_id]['analysis'] == None or generation_json[r_id]['analysis'].strip() == "":
                    return False, "empty analysis"
                if list(generation_json[r_id]['preference'].keys()) != ['pos', 'neg']:
                    return False, "wrong preference key"
                for preference_key in ['pos', 'neg']:
                    if generation_json[r_id]['preference'][preference_key] == None or generation_json[r_id]['preference'][preference_key].strip() == "":
                        return False, "empty preference"
        except json.JSONDecodeError as e:
            return False, "JSON format error"
        return True, None


    def extract_and_save(self, save_path):
        result_dict = {}
        aspect_list = ['work', 'health', 'family', 'leisure']
        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)
                generation_json = json.loads(generation)
                user_id, aspect = sample_id.split('_')
                if user_id not in result_dict.keys():
                    result_dict[user_id] = {}
                assert aspect not in result_dict[user_id].keys()
                result_dict[user_id][aspect] = {}
                for requirement_id in generation_json.keys():
                    result_dict[user_id][aspect][requirement_id] = {}
                    result_dict[user_id][aspect][requirement_id]['requirement'] = generation_json[requirement_id]['requirement']
                    result_dict[user_id][aspect][requirement_id]['preference'] = generation_json[requirement_id]['preference']
        for sample_id in result_dict.keys():
            for aspect in aspect_list:
                assert aspect in result_dict[sample_id].keys()
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    model_name = 'qwen2.5-max'

    data_synthesis_root = '/data5/hzp/MemPAL/data_synthesis_v2/data'
    background_data_dir = os.path.join(data_synthesis_root, 'background')
    background_file_path = os.path.join(background_data_dir, 'background.json')
    with open(background_file_path,'r', encoding='utf-8') as f:
        background_dict = json.load(f)


    ### ----- Step 1: 时间线生成 -----
    prompt_template_dir = '/data5/hzp/MemPAL/data_synthesis_v2/prompt_template/timeline'
    save_dir = os.path.join(data_synthesis_root, 'timeline')
    timeline_file_path = 'timeline.json'
    timeline_generator = TimelineGeneration(background_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = timeline_generator.get_prompt('0010')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    timeline_generator.sequential_generate()
    timeline_generator.extract_and_save(timeline_file_path)
    # ### ---------------------------


    ### ----- Step 2: 总体需求生成 -----
    prompt_template_dir = '/data5/hzp/MemPAL/data_synthesis_v2/prompt_template/requirement'
    timeline_data_dir = os.path.join(data_synthesis_root, 'timeline')
    timeline_file_path = os.path.join(timeline_data_dir, 'timeline.json')
    with open(timeline_file_path,'r', encoding='utf-8') as f:
        timeline_dict = json.load(f)
    save_dir = os.path.join(data_synthesis_root, 'requirement')
    requirement_file_path = 'requirement.json'
    requirement_generator = RequirementGeneration(background_dict, timeline_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = requirement_generator.get_prompt('0010')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    requirement_generator.sequential_generate()
    requirement_generator.extract_and_save(requirement_file_path)
    # ### ---------------------------


    ### ----- Step 3: 总体偏好生成 -----
    prompt_template_dir = '/data5/hzp/MemPAL/data_synthesis_v2/prompt_template/preference'
    timeline_data_dir = os.path.join(data_synthesis_root, 'timeline')
    timeline_file_path = os.path.join(timeline_data_dir, 'timeline.json')
    with open(timeline_file_path,'r', encoding='utf-8') as f:
        timeline_dict = json.load(f)
    requirement_data_dir = os.path.join(data_synthesis_root, 'requirement')
    requirement_file_path = os.path.join(requirement_data_dir, 'requirement.json')
    with open(requirement_file_path,'r', encoding='utf-8') as f:
        requirement_dict = json.load(f)
    save_dir = os.path.join(data_synthesis_root, 'preference')
    preference_file_path = 'preference.json'
    preference_generator = PreferenceGeneration(background_dict, timeline_dict, requirement_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = preference_generator.get_prompt('0010_work')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    preference_generator.sequential_generate()
    preference_generator.extract_and_save(preference_file_path)
    # ### ---------------------------