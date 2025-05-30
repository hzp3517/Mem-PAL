"""
不加memory的需求预测任务baseline实现。
- 输入
    - 近期日志
    - 当前用户询问
- 输出
    - 用户需求
"""
import os
import json
import ast
import csv
import random
import torch
import numpy as np
from tqdm import tqdm
from copy import copy

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')

from llm_generation import LLM_Sequential_Generation
from evaluation.perform_evaluation import CalculateRequirementPredictionMetrics
from evaluation.perform_evaluation import CalculateRequirementPredictionScore


class RequirementPredictionVanilla(LLM_Sequential_Generation):
    def __init__(self, dataset_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        """
        - start_user_id & end_user_id: 当前批次生成哪些user的数据（两端均包含），可用于并行生成时。
        """
        super().__init__(*args, **kwargs)
        self.dataset_dict, self.sample_id_list = self.preprocess(dataset_dict, start_user_id, end_user_id)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)
    

    def preprocess(self, dataset_dict, start_user_id, end_user_id):
        processed_dataset_dict = {}
        sample_id_list = []

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
                processed_dataset_dict[user_id][dialogue_id] = {}
                processed_dataset_dict[user_id][dialogue_id]['logs'] = dialogue_item['logs']
                processed_dataset_dict[user_id][dialogue_id]['topics'] = dialogue_item['topics']
                for topic_id in dialogue_item['topics'].keys():
                    sample_id = "{}_{}".format(dialogue_id, topic_id.split('-')[-1]) # e.g. 0000_sample0_1 (代表该dialogue的topic_1)
                    sample_id_list.append(sample_id)
        return processed_dataset_dict, sample_id_list


    def get_prompt(self, sample_id):
        system_prompt = self.system_prompt_template
        user_prompt_template = copy(self.user_prompt_template)

        user_id = sample_id.split('_')[0]
        dialogue_id = '_'.join(sample_id.split('_')[:-1])
        topic_id = "topic-{}".format(sample_id.split('_')[-1])

        dialogue_item = self.dataset_dict[user_id][dialogue_id]
        logs = ["- [{}] {}".format(log_item['timestamp'], log_item['content']) for log_item in dialogue_item['logs']]
        logs_str = "\n".join(logs)
        user_query = dialogue_item['topics'][topic_id]['user_query']

        user_prompt = user_prompt_template.replace('<logs>', logs_str).replace('<user_query>', user_query)
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
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['requirement']:
                return False, "key error"
            if generation_dict['requirement'] == "..." or generation_dict['requirement'].strip() == "":
                return False, "empty output"
        except:
            return False, "json format error"
        return True, None


    def extract_and_save(self, save_path):
        result_dict = {}
        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)
                generation_dict = json.loads(generation)

                user_id = sample_id.split('_')[0]
                dialogue_id = '_'.join(sample_id.split('_')[:-1])
                dialogue_item = self.dataset_dict[user_id][dialogue_id]
                topic_id = "topic-{}".format(sample_id.split('_')[-1])

                if user_id not in result_dict.keys():
                    result_dict[user_id] = {}
                if dialogue_id not in result_dict[user_id].keys():
                    result_dict[user_id][dialogue_id] = {}
                result_dict[user_id][dialogue_id][topic_id] = {}
                result_dict[user_id][dialogue_id][topic_id]['ground_truth'] = dialogue_item['topics'][topic_id]['requirement']
                result_dict[user_id][dialogue_id][topic_id]['generation'] = generation_dict['requirement']

        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)




if __name__ == '__main__':
    model_name = 'qwen_max'


    dataset_path = '/data5/hzp/MemPAL/data_synthesis_v2/data/input.json'
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_dict = json.load(f)
    framework_root = '/data5/hzp/MemPAL/perassist_framework'
    data_dir = model_name
    prompt_dir = os.path.join(framework_root, 'prompt_template')


    ### ----- 需求预测 -----
    prompt_template_dir = os.path.join(prompt_dir, 'vanilla', 'requirement_prediction')
    save_dir = os.path.join(data_dir, 'vanilla', 'requirement_prediction')
    output_save_path = 'output.json'
    perassist = RequirementPredictionVanilla(dataset_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = perassist.get_prompt("0000_sample30_1")
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    # perassist.sequential_generate()
    # perassist.extract_and_save(output_save_path)

    caculator = CalculateRequirementPredictionMetrics(batch_size=128, device="cuda:0", save_dir=save_dir, generation_path=output_save_path, result_path='result.json')
    caculator()

    # calculator = CalculateRequirementPredictionScore(dataset_path, generation_path=output_save_path, save_dir=save_dir, result_path='llm_score_result.json')
    # # system_prompt, user_prompt = calculator.get_prompt("0000_sample30_2")
    # # print('【system prompt】:\n{}'.format(system_prompt))
    # # print('【user prompt】:\n{}'.format(user_prompt))
    # calculator()
    # ### -------------------------------