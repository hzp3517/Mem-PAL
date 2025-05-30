"""
Stage 2: background summary
- ① 输入: 先前的background summary，当前的situation list；输出：更新后的background summary
"""
import os
import json
import csv
from tqdm import tqdm
import pandas as pd
from copy import copy, deepcopy

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')

from llm_generation import LLM_Sequential_Generation


class BackgroundSummaryGeneration(LLM_Sequential_Generation):
    def __init__(self, dataset_dict, situation_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_dict, self.sample_id_list = self.preprocess(dataset_dict, situation_dict, start_user_id, end_user_id)
        self.background_summary_dict = {} # 用于在生成过程中存储已生成sample的内容，用于后续sample的输入
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)


    def preprocess(self, dataset_dict, situation_dict, start_user_id, end_user_id):
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

        sample_dict = {}
        for user_id in process_user_ids:
            last_background_start_date = dataset_dict[user_id]['history'][0]["logs"][0]["timestamp"].split(' ')[0]
            last_background_end_data = None
            user_sample_idx = 0
            for set_name in ['history', 'query']:
                for sample_item in dataset_dict[user_id][set_name]:
                    sample_id = sample_item['sample_id']
                    assert sample_id == '{}_sample{}'.format(user_id, str(user_sample_idx)) # 确保同一个user的sample_id之间没有间断
                    sample_dict[sample_id] = {}

                    if user_sample_idx > 0:
                        assert last_background_end_data != None
                        sample_dict[sample_id]['background_time_span'] = '{} ~ {}'.format(last_background_start_date, last_background_end_data)
                    else:
                        assert last_background_end_data == None
                        sample_dict[sample_id]['background_time_span'] = '{} ~ {}'.format(last_background_start_date, last_background_start_date)

                    cur_situation_start_date = sample_item["logs"][0]["timestamp"].split(' ')[0]
                    cur_situation_end_date = sample_item["logs"][-1]["timestamp"].split(' ')[0]
                    sample_dict[sample_id]["situation_time_span"] = '{} ~ {}'.format(cur_situation_start_date, cur_situation_end_date)

                    sample_dict[sample_id]["situation_list"] = [situation_dict[user_id][sample_id][i] for i in situation_dict[user_id][sample_id].keys()]

                    user_sample_idx += 1
                    last_background_end_data = sample_item["dialogue_timestamp"].split(' ')[0]

        sample_id_list = list(sample_dict.keys())
        return sample_dict, sample_id_list


    def get_prompt(self, sample_id):
        user_id = sample_id.split('_')[0]
        user_sample_idx = int(sample_id.split('sample')[-1])
        if user_sample_idx > 0:
            last_sample_id = '{}_sample{}'.format(user_id, str(user_sample_idx-1))
            assert user_id in self.background_summary_dict.keys() and last_sample_id in self.background_summary_dict[user_id].keys()
            last_background_summary_dict = self.background_summary_dict[user_id][last_sample_id]
        else:
            last_background_summary_dict = {"work": "", "health": "", "family": "", "leisure": ""}
        last_background_dict = {"time_span": self.sample_dict[sample_id]["background_time_span"], "background": last_background_summary_dict}
        last_background_str = json.dumps(last_background_dict, indent=4, ensure_ascii=False)
        cur_situations_dict = {"time_span": self.sample_dict[sample_id]["situation_time_span"], "situation_list": [i['situation'] for i in self.sample_dict[sample_id]["situation_list"]]}
        cur_situations_str = json.dumps(cur_situations_dict, indent=4, ensure_ascii=False)

        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.replace('<last_background>', last_background_str).replace('<cur_situations>', cur_situations_str)
        return system_prompt, user_prompt


    def generation_postprocess(self, raw_generation_str):
        if '```json' in raw_generation_str:
            raw_generation_str = raw_generation_str.split('```json')[-1]
            if '```' in raw_generation_str:
                raw_generation_str = raw_generation_str.split('```')[0]
        raw_generation_str = raw_generation_str.replace('```', '').strip()
        generation_str = raw_generation_str.replace('\"\"\"', '').strip()
        return generation_str


    def check_generation(self, sample_id, sample_raw_generation):
        generation = self.generation_postprocess(sample_raw_generation)
        try: # 确保生成内容符合json格式
            generation_dict = json.loads(generation)

            # 1. 确保"updating_aspects"和"updating_content"存在
            if list(generation_dict.keys()) != ["updating_aspects", "updating_content"]:
                return False, "wrong output keys"
            
            for aspect in generation_dict["updating_aspects"]:
                # 2. 确保"updating_aspects"中的aspect值正确（属于["work", "health", "family", "leisure"]）
                if aspect not in ["work", "health", "family", "leisure"]:
                    return False, "wrong aspect name"

                # 3. 确保"updating_aspects"中出现的aspect在"updating_content"中也出现，且对应的值有效
                if aspect not in generation_dict["updating_content"].keys() or generation_dict["updating_content"][aspect] == None or generation_dict["updating_content"][aspect].strip() == "":
                    return False, "missing updating content"

        except json.JSONDecodeError as e:
            return False, "JSON format error"

        return True, None


    def postprocess_for_iterative_generation(self, sample_id, success, generation):
        user_id = sample_id.split('_')[0]
        user_sample_idx = int(sample_id.split('sample')[-1])

        if user_id not in self.background_summary_dict.keys():
            self.background_summary_dict[user_id] = {}
            last_background_dict = {"work": "", "health": "", "family": "", "leisure": ""}
        else:
            last_sample_id = "{}_sample{}".format(user_id, str(user_sample_idx-1))
            last_background_dict = self.background_summary_dict[user_id][last_sample_id]

        if success:
            generation = self.generation_postprocess(generation)
            generation_json = json.loads(generation)
            cur_background_dict = deepcopy(last_background_dict)
            for aspect in ["work", "health", "family", "leisure"]:
                if aspect in generation_json["updating_aspects"]:
                    cur_background_dict[aspect] = generation_json["updating_content"][aspect]
            self.background_summary_dict[user_id][sample_id] = cur_background_dict
        else:
            self.background_summary_dict[user_id][sample_id] = deepcopy(last_background_dict)


    def extract_and_save(self, save_path):
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.background_summary_dict, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    model_name = 'qwen_max'

    dataset_path = '/data5/hzp/MemPAL/data_synthesis_v2/data/input.json'
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_dict = json.load(f)
    framework_root = '/data5/hzp/MemPAL/perassist_framework'
    data_dir = model_name
    prompt_dir = os.path.join(framework_root, 'prompt_template')


    ## ----- Step 1: 生成background_summary -----
    prompt_template_dir = os.path.join(prompt_dir, 'memory_rag_v2', 'background_summary')
    background_summary_dir = os.path.join(data_dir, 'memory_rag_v2', 'background_summary')
    raw_file_name = 'background_summary_raw.csv'
    background_summary_file_path = 'background_summary.json'
    situation_dir = os.path.join(data_dir, 'memory_rag_v2', 'log_analysis', 'situation')
    situation_file_path = 'situation.json'
    with open(os.path.join(situation_dir, situation_file_path), 'r', encoding='utf-8') as f:
        situation_dict = json.load(f)
    background_summary_data_generator = BackgroundSummaryGeneration(dataset_dict, situation_dict, prompt_template_dir, save_dir=background_summary_dir, raw_file_name=raw_file_name, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = background_summary_data_generator.get_prompt('0000_sample0')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    background_summary_data_generator.sequential_generate()
    background_summary_data_generator.extract_and_save(background_summary_file_path)
    ## -------------------------