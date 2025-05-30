"""
第1步：合成用户的基本背景
"""
import os
import json
import requests
import csv
from tqdm import tqdm
import pandas as pd
from copy import copy, deepcopy
from llm_generation import LLM_Sequential_Generation


class BackgroundGeneration(LLM_Sequential_Generation):
    def __init__(self, prompt_template_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(os.path.join(prompt_template_dir, 'seed_users.json'), 'r', encoding="utf-8") as f:
            self.seed_users = json.load(f)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)


    def get_prompt(self, sample_id):
        user_num_per_generation = 10 # 每次生成10个用户
        example_json = self.seed_users[int(sample_id)]
        system_prompt = self.system_prompt_template
        start_id_str = str(int(sample_id) * user_num_per_generation).zfill(4)
        end_id_str = str(int(sample_id) * user_num_per_generation + 9).zfill(4)
        user_prompt = self.user_prompt_template.replace('<example_json>', json.dumps(example_json, indent=4, ensure_ascii=False)).replace('<start_id>', start_id_str).replace('<end_id>', end_id_str)
        return system_prompt, user_prompt
    
    
    def check_generation(self, sample_id, raw_generation):
        user_num_per_generation = 10 # 每次生成10个用户
        background_keys = ['id', 'name', 'gender', 'age', 'occupation', 'health status', 'family members', 'hobbies', 'personality']
        personality_keys = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        personality_values = ['low', 'medium', 'high']
        generation = self.generation_postprocess(raw_generation)

        try: # 确保生成内容符合json格式
            generation_list = json.loads(generation)

            if len(generation_list) != user_num_per_generation:
                return False, "wrong user num"

            user_cnt = int(sample_id) * user_num_per_generation
            for generation_sample in generation_list:
                for k in background_keys:
                    if k not in generation_sample.keys():
                        return False, "key error"
                for k in personality_keys:
                    if k not in generation_sample['personality'].keys():
                        return False, "personality key error"
                    if generation_sample['personality'][k] not in personality_values:
                        return False, "personality value error"
                if int(generation_sample['id']) != user_cnt:
                    return False, "id error of sample"
                user_cnt += 1

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
                generation_json_list = json.loads(generation)
                for sample in generation_json_list:
                    sample_id = sample['id']
                    del sample['id']
                    result_dict[sample_id] = sample
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    model_name = 'qwen2.5-max'

    prompt_template_dir = '/data5/hzp/MemPAL/data_synthesis_v2/prompt_template/background'
    save_dir = '/data5/hzp/MemPAL/data_synthesis_v2/data/background/'
    background_file_path = 'background.json'

    sample_id_list = ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009"]
    background_generator = BackgroundGeneration(prompt_template_dir, save_dir=save_dir, model_name=model_name, sample_id_list=sample_id_list)

    # # check the prompt
    # system_prompt, user_prompt = background_generator.get_prompt('0001')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    background_generator.sequential_generate()
    background_generator.extract_and_save(background_file_path)