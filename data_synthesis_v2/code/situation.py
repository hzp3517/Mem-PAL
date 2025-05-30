"""
第3步：合成用户的需求情境及详细经历描述
"""
import os
import json
import requests
import csv
from tqdm import tqdm
import pandas as pd
import re
import ast
import random
from copy import copy, deepcopy
from datetime import datetime
from dateutil.relativedelta import relativedelta
from llm_generation import LLM_Sequential_Generation


class SituationGeneration(LLM_Sequential_Generation):
    def __init__(self, background_dict, timeline_dict, requirement_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in background_dict.keys():
            background = background_dict[k]
            del background['personality']
        self.background_dict, self.timeline_dict, self.sample_id_list = self.preprocess_dataset(background_dict, timeline_dict, start_user_id, end_user_id)
        self.requirement_dict = requirement_dict
        self.situation_dict = {} # 用于在生成过程中存储已生成sample的内容，用于后续sample的输入
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)


    def monthstr_format(self, month_str):
        date_obj = datetime.strptime(month_str, "%Y年%m月")
        formatted_month = date_obj.strftime("%Y-%m")
        return formatted_month


    def monthstr_unformat(self, formatted_month):
        year, month = formatted_month.split('-')
        month = str(int(month))
        month_str = "{}年{}月".format(year, month)
        return month_str


    def add_months(self, date_str, months_to_add):
        current_date = datetime.strptime(date_str, "%Y年%m月") # 将输入的字符串转换为日期对象
        new_date = current_date + relativedelta(months=months_to_add) # 使用relativedelta进行月份的加减
        return new_date.strftime("%Y年") + str(new_date.month) + "月" # 将新的日期转换为所需格式并返回


    def preprocess_dataset(self, background_dict, timeline_dict, start_user_id, end_user_id):
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
        sample_id_list = []
        for user_id in process_user_ids:
            for month_str in timeline_dict[user_id].keys():
                sample_id_list.append("{}_{}".format(user_id, self.monthstr_format(month_str)))
        return background_dict, timeline_dict, sample_id_list


    def get_prompt(self, sample_id):
        user_id, cur_formatted_month = sample_id.split('_')
        cur_month_str = self.monthstr_unformat(cur_formatted_month)
        last_month_str = self.add_months(cur_month_str, -1)

        has_short_memory = True if last_month_str in self.timeline_dict[user_id].keys() else False

        background_dict = self.background_dict[user_id]
        timeline_dict = self.timeline_dict[user_id]

        user_requirement_dict = {}
        aspect_list = ['work', 'health', 'family', 'leisure']
        for aspect in aspect_list:
            for r_k in self.requirement_dict[user_id][aspect].keys():
                user_requirement_dict[r_k] = self.requirement_dict[user_id][aspect][r_k]

        background_str = json.dumps(background_dict, ensure_ascii=False)
        requirement_str = json.dumps(user_requirement_dict, ensure_ascii=False)
        cur_month_summary = timeline_dict[cur_month_str]

        pattern = re.compile(r'\$\$\{(.*?)\$\$\}', re.DOTALL)
        user_prompt_template = deepcopy(self.user_prompt_template)
        if has_short_memory:
            user_prompt_template = re.sub(pattern, r'\1', copy(user_prompt_template)) # 保留 $${ 和 $$} 之间的内容
            last_sample_id = "{}_{}".format(user_id, self.monthstr_format(last_month_str))
            assert last_sample_id in self.situation_dict.keys()
            last_month_end_date = list(self.situation_dict[last_sample_id].keys())[-1].split('~')[-1].strip()
            last_month_situation = json.dumps(self.situation_dict[last_sample_id], indent=4, ensure_ascii=False)
            user_prompt_template = user_prompt_template.replace('<last_month_str>', last_month_str).replace('<last_month_situation>', last_month_situation).replace('<last_month_end_date>', last_month_end_date)
        else:
            user_prompt_template = re.sub(pattern, '', copy(user_prompt_template)) # 去除 $${ 和 $$} 及其之间的内容
        
        system_prompt = self.system_prompt_template
        user_prompt = user_prompt_template.replace('<background>', background_str).replace('<requirement>', requirement_str).replace('<cur_month_str>', cur_month_str).replace('<cur_month_summary>', cur_month_summary)
        
        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        raw_generation = self.generation_postprocess(raw_generation)

        user_id, cur_formatted_month = sample_id.split('_')
        cur_month_str = self.monthstr_unformat(cur_formatted_month)
        user_requirement_dict = {}
        aspect_list = ['work', 'health', 'family', 'leisure']
        for aspect in aspect_list:
            for r_k in self.requirement_dict[user_id][aspect].keys():
                user_requirement_dict[r_k] = self.requirement_dict[user_id][aspect][r_k]

        last_month_str = self.add_months(cur_month_str, -1)
        has_short_memory = True if last_month_str in self.timeline_dict[user_id].keys() else False
        if has_short_memory:
            last_sample_id = "{}_{}".format(user_id, self.monthstr_format(last_month_str))
            last_month_end_date = list(self.situation_dict[last_sample_id].keys())[-1].split('~')[-1].strip()
            prev_end_date = datetime.strptime(last_month_end_date, "%Y-%m-%d")
        else:
            prev_end_date = None

        try:
            generation_list = ast.literal_eval(raw_generation)
            
            # 1. 情境数量符合要求
            if len(generation_list) < 4 or len(generation_list) > 6:
                return False, "Too much or too few situation items"
            
            for situation_item in generation_list:
                # 2. 字典格式以及键正确
                if not isinstance(situation_item, dict):
                    return False, "format error"
                if list(situation_item.keys()) != ['time_span', 'requirement_ids', 'situation']:
                    return False, "key error"
                
                # 3. 需求id内容以及数量正确
                if not isinstance(situation_item['requirement_ids'], list):
                    return False, "format error"
                if len(list(set(situation_item['requirement_ids']))) != len(situation_item['requirement_ids']):
                    return False, "duplicate requirement id"
                if len(situation_item['requirement_ids']) < 1 or len(situation_item['requirement_ids']) > 3:
                    return False, "Too much or too few requirement ids"
                for requirement_id in situation_item['requirement_ids']:
                    if requirement_id not in user_requirement_dict.keys():
                        return False, "wrong requirement id"

                # 4. 当前起始和结束日期合法
                try:
                    cur_start_date = datetime.strptime(situation_item['time_span'].split('~')[0].strip(), "%Y-%m-%d")
                    cur_end_date = datetime.strptime(situation_item['time_span'].split('~')[-1].strip(), "%Y-%m-%d")
                except ValueError:
                    return False, "wrong format of date"
                
                # 5. 当前月份正确，时间先后顺序正确
                if self.monthstr_unformat('-'.join(situation_item['time_span'].split('~')[-1].strip().split('-')[:-1])) != cur_month_str:
                    return False, "wrong month of end date"
                if prev_end_date:
                    if prev_end_date > cur_start_date or prev_end_date == cur_start_date:
                        return False, "date temporal error"
                prev_end_date = cur_end_date
                if cur_start_date > cur_end_date or cur_start_date == cur_end_date:
                    return False, "date temporal error"
                
                # 6. 情境内容不为空
                if situation_item['situation'] == None or situation_item['situation'].strip() == "":
                    return False, "empty situation"

        except:
            return False, "list format error"
        
        return True, None


    def postprocess_for_iterative_generation(self, sample_id, success, generation):
        if success:
            generation = self.generation_postprocess(generation)
            generation_json = json.loads(generation)
            sample_situation_dict = {}
            for situation_item in generation_json:
                sample_situation_dict[situation_item["time_span"]] = situation_item["situation"]
            self.situation_dict[sample_id] = sample_situation_dict
        else:
            self.situation_dict[sample_id] = {}


    def sample_time(self, start_hour, end_hour):
        assert start_hour >= 0 and start_hour < 24 and isinstance(start_hour, int)
        assert end_hour >= 0 and end_hour < 24 and isinstance(end_hour, int)
        assert end_hour > start_hour
        random_number = random.randint(start_hour, end_hour)
        return datetime.strptime("{}:00:00".format(str(random_number).zfill(2)), "%H:%M:%S").time()


    def extract_and_save(self, save_path):
        result_dict = {}

        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)

                generation_list = ast.literal_eval(generation)

                user_id, cur_formatted_month = sample_id.split('_')
                if user_id not in result_dict.keys():
                    result_dict[user_id] = {}
                
                for situation_item in generation_list:
                    cur_start_date = situation_item['time_span'].split('~')[0].strip()
                    cur_end_date = situation_item['time_span'].split('~')[-1].strip()
                    result_dict[user_id][cur_end_date] = {}
                    cur_start_timestamp = "{} {}".format(cur_start_date, self.sample_time(7, 12))
                    cur_end_timestamp = "{} {}".format(cur_end_date, self.sample_time(12, 21))
                    result_dict[user_id][cur_end_date]['start_timestamp'] = cur_start_timestamp
                    result_dict[user_id][cur_end_date]['end_timestamp'] = cur_end_timestamp
                    result_dict[user_id][cur_end_date]['requirement_ids'] = situation_item['requirement_ids']
                    result_dict[user_id][cur_end_date]['situation'] = situation_item['situation']
                
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)


class ExperienceGeneration(LLM_Sequential_Generation):
    def __init__(self, background_dict, situation_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in background_dict.keys():
            background = background_dict[k]
            del background['personality']
        self.background_dict, self.situation_dict, self.sample_id_list = self.preprocess_dataset(background_dict, situation_dict, start_user_id, end_user_id)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)


    def preprocess_dataset(self, background_dict, situation_dict, start_user_id, end_user_id):
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
        sample_id_list = []
        for user_id in process_user_ids:
            for end_date in situation_dict[user_id].keys():
                sample_id_list.append("{}_{}".format(user_id, end_date))
        return background_dict, situation_dict, sample_id_list
    

    def get_prompt(self, sample_id):
        user_id, cur_end_date = sample_id.split('_')
        end_date_list = list(self.situation_dict[user_id].keys())
        cur_situation_idx = end_date_list.index(cur_end_date)
        if cur_situation_idx == 0:
            last_end_date = None
        else:
            last_end_date = end_date_list[cur_situation_idx - 1]

        background_dict = self.background_dict[user_id]
        background_str = json.dumps(background_dict, ensure_ascii=False)

        cur_situation_start_timestamp = self.situation_dict[user_id][cur_end_date]["start_timestamp"]
        cur_situation_end_timestamp = self.situation_dict[user_id][cur_end_date]["end_timestamp"]
        cur_situation = self.situation_dict[user_id][cur_end_date]["situation"]

        pattern = re.compile(r'\$\$\{(.*?)\$\$\}', re.DOTALL)
        user_prompt_template = deepcopy(self.user_prompt_template)

        if last_end_date:
            user_prompt_template = re.sub(pattern, r'\1', copy(user_prompt_template)) # 保留 $${ 和 $$} 之间的内容
            last_situation_start_timestamp = self.situation_dict[user_id][last_end_date]["start_timestamp"]
            last_situation_end_timestamp = self.situation_dict[user_id][last_end_date]["end_timestamp"]
            last_situation = self.situation_dict[user_id][last_end_date]["situation"]
            user_prompt_template = user_prompt_template.replace('<last_situation_start_timestamp>', last_situation_start_timestamp).replace('<last_situation_end_timestamp>', last_situation_end_timestamp).replace('<last_situation>', last_situation)
        else:
            user_prompt_template = re.sub(pattern, '', copy(user_prompt_template)) # 去除 $${ 和 $$} 及其之间的内容
        
        system_prompt = self.system_prompt_template
        user_prompt = user_prompt_template.replace('<background>', background_str).replace('<cur_situation_start_timestamp>', cur_situation_start_timestamp).replace('<cur_situation_end_timestamp>', cur_situation_end_timestamp).replace('<cur_situation>', cur_situation)
        
        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation):
        if len(raw_generation) < 100:
            return False, "generation too short"
        else:
            return True, None


    def extract_and_save(self, save_path):
        result_dict = {}

        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)

                user_id, cur_end_date = sample_id.split('_')
                if user_id not in result_dict.keys():
                    result_dict[user_id] = {}

                result_dict[user_id][cur_end_date] = {}
                result_dict[user_id][cur_end_date]['start_timestamp'] = self.situation_dict[user_id][cur_end_date]['start_timestamp']
                result_dict[user_id][cur_end_date]['end_timestamp'] = self.situation_dict[user_id][cur_end_date]['end_timestamp']
                result_dict[user_id][cur_end_date]['experience'] = generation
                
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


    ### ----- Step 1: 需求情境生成 -----
    prompt_template_dir = '/data5/hzp/MemPAL/data_synthesis_v2/prompt_template/situation'
    timeline_data_dir = os.path.join(data_synthesis_root, 'timeline')
    timeline_file_path = os.path.join(timeline_data_dir, 'timeline.json')
    with open(timeline_file_path,'r', encoding='utf-8') as f:
        timeline_dict = json.load(f)
    requirement_data_dir = os.path.join(data_synthesis_root, 'requirement')
    requirement_file_path = os.path.join(requirement_data_dir, 'requirement.json')
    with open(requirement_file_path,'r', encoding='utf-8') as f:
        requirement_dict = json.load(f)
    save_dir = os.path.join(data_synthesis_root, 'situation')
    situation_file_path = 'situation.json'
    situation_generator = SituationGeneration(background_dict, timeline_dict, requirement_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = situation_generator.get_prompt('0000_2024-01')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    situation_generator.sequential_generate()
    situation_generator.extract_and_save(situation_file_path)
    # ### ---------------------------


    ### ----- Step 2: 详细经历生成 -----
    prompt_template_dir = '/data5/hzp/MemPAL/data_synthesis_v2/prompt_template/experience'
    
    situation_data_dir = os.path.join(data_synthesis_root, 'situation')
    situation_file_path = os.path.join(situation_data_dir, 'situation.json')
    with open(situation_file_path,'r', encoding='utf-8') as f:
        situation_dict = json.load(f)
    save_dir = os.path.join(data_synthesis_root, 'experience')
    experience_file_path = 'experience.json'
    experience_generator = ExperienceGeneration(background_dict, situation_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = experience_generator.get_prompt('0000_2024-01-05')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    experience_generator.sequential_generate()
    experience_generator.extract_and_save(experience_file_path)
    # ### ---------------------------


    ### ----- Step 3: 对话情境采样 -----
    '''
    采样规则：
    1. 每个原始情境以40%的概率丢弃（不合成对话，但仍将用于合成日志）
    2. 确保不会连续4个原始情境被丢弃（避免某个对话情境对应的logs过多）
    3. 确保12月（作为query的部分）至少有1段对话
    '''
    situation_data_dir = os.path.join(data_synthesis_root, 'situation')
    situation_file_path = os.path.join(situation_data_dir, 'situation.json')
    with open(situation_file_path,'r', encoding='utf-8') as f:
        situation_dict = json.load(f)
    save_path = os.path.join(situation_data_dir, 'dialogue_situation_ids.json')

    def situation_sampling(original_user_situation_list):
        history_user_situation_list = []
        query_user_situation_list = []
        cnt_drop = 0
        for situation_id in original_user_situation_list:
            if random.random() > 0.4 or cnt_drop >= 3:
                if situation_id.split('-')[1] != "12":
                    history_user_situation_list.append(situation_id)
                else:
                    query_user_situation_list.append(situation_id)
                cnt_drop = 0
            else:
                cnt_drop += 1
        return history_user_situation_list, query_user_situation_list
    
    dialogue_situation_dict = {}
    for user_id in situation_dict.keys():
        dialogue_situation_dict[user_id] = {}
        dialogue_situation_dict[user_id]['history'] = []
        dialogue_situation_dict[user_id]['query'] = []
        while len(dialogue_situation_dict[user_id]['query']) == 0:
            dialogue_situation_dict[user_id]['history'], dialogue_situation_dict[user_id]['query'] = situation_sampling(situation_dict[user_id].keys())
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dialogue_situation_dict, f, indent=4, ensure_ascii=False)