"""
Based on the "FairEval" method [https://github.com/i-Eval/FairEval]
Apply the Multiple Evidence Calibration (MEC) and Balanced Position Calibration (BPC) methods to calibrate the positional bias of LLMs.
"""

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

from llm_generation import LLM_Sequential_Generation


class Assistant_Evaluator(LLM_Sequential_Generation):
    def __init__(self, evaluate_aspect, test_method, baseline_method, k_mec, evaluation_id, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_path = '/data5/hzp/MemPAL/data_synthesis_v2/data/input.json'
        self.data_synthesis_root = '/data5/hzp/MemPAL/data_synthesis_v2'
        self.framework_root = '/data5/hzp/MemPAL/perassist_framework'
        self.dialogue_evaluation_dir = os.path.join(self.framework_root, 'code', 'dialogue_evaluation')
        self.evaluate_aspect = evaluate_aspect
        self.test_method = test_method
        self.baseline_method = baseline_method
        self.k_mec = k_mec
        self.evaluation_id = evaluation_id

        self.background_dict, self.dataset_dict, self.sample_id_list = self.preprocess_dataset(start_user_id, end_user_id)
        

    def preprocess_dataset(self, start_user_id, end_user_id):
        synthesis_data_dir = os.path.join(self.data_synthesis_root, 'data')

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset_dict = json.load(f)
        situation_path = os.path.join(synthesis_data_dir, 'situation', 'situation.json')
        background_path = os.path.join(self.data_synthesis_root, 'data', 'background', 'background.json')
        with open(background_path, 'r', encoding='utf-8') as f:
            background_dict = json.load(f)
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
                for topic_id in dialogue_item['topics'].keys():
                    topic_idx = int(topic_id.split('-')[-1]) - 1
                    topic_sample_id = "{}_{}".format(dialogue_id, topic_id) # e.g. 0000_sample0_topic-1
                    sample_id_list.append(topic_sample_id)
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

        return background_dict, processed_dataset_dict, sample_id_list


    def get_prompt(self, sample_id):
        user_id = sample_id.split('_')[0]
        
        eval_prompt_template_dir = os.path.join(self.dialogue_evaluation_dir, 'eval_llm_prompt')
        with open(os.path.join(eval_prompt_template_dir, self.evaluate_aspect, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            system_prompt_template = ''.join(lines)
        system_prompt = copy(system_prompt_template)
        with open(os.path.join(eval_prompt_template_dir, self.evaluate_aspect, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            user_prompt_template = ''.join(lines)

        sample_personality_dict = copy(self.background_dict[user_id]['personality'])
        sample_background_dict = copy(self.background_dict[user_id])
        topic_data_dict = self.dataset_dict[user_id][sample_id]
        del sample_background_dict['personality']
        background_str = json.dumps(sample_background_dict, ensure_ascii=False)
        personality_str = json.dumps(sample_personality_dict, ensure_ascii=False)
        situation_str = json.dumps(topic_data_dict['situation'], ensure_ascii=False)

        if self.evaluate_aspect == 'requirement':
            requirement_dict = {
                'user_query': topic_data_dict['user_query'],
                'implicit_needs': ['<need_dict_{}>'.format(str(i)) for i in range(1, len(topic_data_dict['implicit_needs'])+1)],
                'requirement': topic_data_dict['requirement']
            }
            requirement_str = json.dumps(requirement_dict, indent=4, ensure_ascii=False)
            for i in range(1, len(topic_data_dict['implicit_needs'])+1):
                requirement_str = requirement_str.replace('<need_dict_{}>'.format(str(i)), json.dumps(topic_data_dict['implicit_needs'][i-1], ensure_ascii=False))
            user_prompt_template = user_prompt_template.replace('<requirement>', requirement_str)
        else:
            assert self.evaluate_aspect == 'preference'
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
            user_prompt_template = user_prompt_template.replace('<preference>', solutions_str)

        dialogue_context_str_list = []
        for method_info in [self.test_method, self.baseline_method]:
            dialogue_context_list = []
            dialogue_path = os.path.join(method_info['data_dir'], 'dialogue.json')
            with open(dialogue_path, 'r', encoding='utf-8') as f:
                dialogue_dict = json.load(f)
            for turn_id in dialogue_dict[user_id][sample_id].keys():
                for role in ['user', 'assistant']:
                    dialogue_context_list.append("- \"{}\": \"{}\"".format(role, dialogue_dict[user_id][sample_id][turn_id][role]))
            dialogue_context_str = '\n'.join(dialogue_context_list)
            dialogue_context_str_list.append(dialogue_context_str)

        if self.evaluation_id.split('-')[0] == 'p1': # test_method作为assistant-1，baseline_method作为assistant-2
            user_prompt_template = user_prompt_template.replace('<dialogue_assistant_1>', dialogue_context_str_list[0]).replace('<dialogue_assistant_2>', dialogue_context_str_list[1])
        else:
            assert self.evaluation_id.split('-')[0] == 'p2' # test_method作为assistant-2，baseline_method作为assistant-1
            user_prompt_template = user_prompt_template.replace('<dialogue_assistant_1>', dialogue_context_str_list[1]).replace('<dialogue_assistant_2>', dialogue_context_str_list[0])

        user_prompt = user_prompt_template.replace('<background>', background_str).replace('<personality>', personality_str).replace('<situation>', situation_str)
        return system_prompt, user_prompt


    def check_generation(self, sample_id, sample_raw_generation):
        generation = self.generation_postprocess(sample_raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['analysis', 'scores']:
                return False, "wrong keys"
            if list(generation_dict['analysis'].keys()) != ['assistant-1', 'assistant-2', 'overall']:
                return False, "wrong analysis keys"
            if list(generation_dict['scores'].keys()) != ['assistant-1', 'assistant-2']:
                return False, "wrong score keys"
            for assistant_id in generation_dict['scores'].keys():
                score = float(generation_dict['scores'][assistant_id])
                if score < 1 or score > 10:
                    return False, "invalid score"
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
                generation_dict = json.loads(generation)

                user_id = sample_id.split('_')[0]
                if user_id not in result_dict.keys():
                    result_dict[user_id] = {}
                result_dict[user_id][sample_id] = {}
                result_dict[user_id][sample_id]['analysis'] = generation_dict['analysis']
                result_dict[user_id][sample_id]['scores'] = {}
                if self.evaluation_id.split('-')[0] == 'p1': # test_method作为assistant-1，baseline_method作为assistant-2
                    result_dict[user_id][sample_id]['scores'][self.test_method['name']] = generation_dict['scores']['assistant-1']
                    result_dict[user_id][sample_id]['scores'][self.baseline_method['name']] = generation_dict['scores']['assistant-2']
                else:
                    assert self.evaluation_id.split('-')[0] == 'p2' # test_method作为assistant-2，baseline_method作为assistant-1
                    result_dict[user_id][sample_id]['scores'][self.baseline_method['name']] = generation_dict['scores']['assistant-1']
                    result_dict[user_id][sample_id]['scores'][self.test_method['name']] = generation_dict['scores']['assistant-2']

        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)





if __name__ == '__main__':
    import multiprocessing

    evaluate_llm_name = "gpt4"
    # evaluate_aspect = "requirement" # ["requirement", "preference"]
    evaluate_aspect = "preference"
    k_mec = 3 # samples `k` scores for the multiple evidence calibration (and another `k` scores by swapping the position of the 2 responses)

    data_root = '/data5/hzp/MemPAL/perassist_framework/data/qwen_max'
    test_method = { # 本次对比评价的结果会存在该方法对应的'data_dir'下
        'name': 'memory_rag_v2_situquery',
        'data_dir': os.path.join(data_root, 'memory_rag_v2/dialogue_interaction_situquery')
    }
    baseline_method = {
        'name': 'memorybank',
        'data_dir': os.path.join(data_root, 'memorybank/dialogue_interaction')
    }

    save_dir = os.path.join(test_method['data_dir'], 'llm_compare_evaluation', baseline_method['name'], evaluate_aspect)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)


    # # 检查评价生成的prompt
    # evaluation_id = "p1-1"
    # process_save_dir = os.path.join(save_dir, evaluation_id)
    # evaluator = Assistant_Evaluator(evaluate_aspect, test_method, baseline_method, k_mec, evaluation_id=evaluation_id, model_name=evaluate_llm_name, save_dir=process_save_dir)
    # system_prompt, user_prompt = evaluator.get_prompt('0070_sample32_topic-2')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))


    # 生成多次评价分数
    n_process = k_mec * 2
    evaluation_id_list = ["p1-{}".format(str(i)) for i in range(1, k_mec+1)] + ["p2-{}".format(str(i)) for i in range(1, k_mec+1)]
    def generation_func(evaluation_id):
        process_save_dir = os.path.join(save_dir, evaluation_id)
        if not os.path.exists(process_save_dir):
            os.makedirs(process_save_dir, exist_ok=True)
        process_result_save_path = 'process_evaluation_output.json'
        evaluator = Assistant_Evaluator(evaluate_aspect, test_method, baseline_method, k_mec, evaluation_id=evaluation_id, model_name=evaluate_llm_name, save_dir=process_save_dir)
        evaluator.sequential_generate()
        evaluator.extract_and_save(process_result_save_path)
    with multiprocessing.Pool(processes=n_process) as pool:
        pool.map(generation_func, evaluation_id_list) # 提交任务到进程池
        pool.close()  # 关闭进程池，防止新任务被提交
        pool.join()   # 等待所有子进程执行完毕
    print("All processes have finished. Proceeding to the next step...")


    # 多次分数取平均后得出每个sample的对比评价结果
    avg_evaluation_score_path = os.path.join(save_dir, 'avg_evaluation_score.json')
    evaluation_result_path = os.path.join(save_dir, 'result.json')

    avg_evaluation_score_dict = {}
    evaluation_result_dict = {'win': 0, 'tie': 0, 'lose': 0} # test_method分数更高算win

    process_evaluation_dict_list = []
    for evaluation_id in evaluation_id_list:
        with open(os.path.join(save_dir, evaluation_id, 'process_evaluation_output.json'), 'r', encoding='utf-8') as f:
            proess_evaluation_dict = json.load(f)
        process_evaluation_dict_list.append(deepcopy(proess_evaluation_dict))
    
    for user_id in process_evaluation_dict_list[0].keys():
        avg_evaluation_score_dict[user_id] = {}
        for sample_id in process_evaluation_dict_list[0][user_id].keys():
            avg_evaluation_score_dict[user_id][sample_id] = {}
            avg_evaluation_score_dict[user_id][sample_id][test_method['name']] = sum([process_evaluation_dict[user_id][sample_id]['scores'][test_method['name']] * 1.0 for process_evaluation_dict in process_evaluation_dict_list]) / n_process
            avg_evaluation_score_dict[user_id][sample_id][baseline_method['name']] = sum([process_evaluation_dict[user_id][sample_id]['scores'][baseline_method['name']] * 1.0 for process_evaluation_dict in process_evaluation_dict_list]) / n_process
            if avg_evaluation_score_dict[user_id][sample_id][test_method['name']] > avg_evaluation_score_dict[user_id][sample_id][baseline_method['name']]:
                evaluation_result_dict['win'] += 1
            elif avg_evaluation_score_dict[user_id][sample_id][test_method['name']] < avg_evaluation_score_dict[user_id][sample_id][baseline_method['name']]:
                evaluation_result_dict['lose'] += 1
            else:
                evaluation_result_dict['tie'] += 1

    with open(avg_evaluation_score_path, 'w', encoding='utf-8') as f:
        json.dump(avg_evaluation_score_dict, f, indent=4, ensure_ascii=False)
    with open(evaluation_result_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result_dict, f, indent=4, ensure_ascii=False)
