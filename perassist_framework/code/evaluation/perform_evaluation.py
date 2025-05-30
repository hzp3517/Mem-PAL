import os
import json
import torch
import numpy as np
import csv
from tqdm import tqdm
from copy import copy, deepcopy
from evaluation.metrics import calculate_single_bleu

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')
from llm_generation import LLM_Sequential_Generation


class CalculateNLGMetrics(object):
    def __init__(self, batch_size, device, save_dir, result_path='result.json'):
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir
        self.result_path = os.path.join(self.save_dir, result_path)
        self.n_nums = []


    def init_metrics_record(self):
        '''
        如果需要削减评价指标类型，在子类中重写该函数
        '''
        self.metric_types = ['bleu']
        self.metrics = {}
        for metric in self.metric_types:
            if metric == 'bleu':
                for i in range(1, 5):
                    self.metrics['bleu-{}'.format(str(i))] = []
            else:
                self.metrics[metric] = []


    def split_list(self, lst):
        return [lst[i:i + self.batch_size] for i in range(0, len(lst), self.batch_size)]
    

    def get_data(self):
        '''
        需要在子类中重写该函数，在该函数中读取GT和prediction并分别加载为列表的形式
        '''
        gt_list = []
        pred_list = []

        # 读取GT和prediction序列

        gt_batchs = self.split_list(gt_list)
        pred_batchs = self.split_list(pred_list)
        return gt_batchs, pred_batchs


    def cal_batch_metrics(self, batch_gt_list, batch_pred_list):
        assert len(batch_gt_list) == len(batch_pred_list)

        self.n_nums.append(len(batch_gt_list))
        for metric in self.metric_types:
            if metric == 'bleu':
                # bleu-1 ~ bleu-4
                batch_metrics = {}
                for i in range(1, 5):
                    batch_metrics['bleu-{}'.format(str(i))] = []
                for gt, pred in zip(batch_gt_list, batch_pred_list):
                    raw_bleu_scores = calculate_single_bleu(gt, pred)
                    for k in raw_bleu_scores.keys():
                        batch_metrics[k].append(raw_bleu_scores[k])
                for k in raw_bleu_scores.keys():
                    batch_metrics[k] = np.mean(batch_metrics[k])
                    self.metrics[k].append(batch_metrics[k])
            else:
                raise NotImplementedError("metric {} has not been implemented.".format(metric))


    def __call__(self):
        self.init_metrics_record()
        gt_batchs, pred_batchs = self.get_data()
        assert len(gt_batchs) == len(gt_batchs)
        for i in tqdm(range(len(gt_batchs)), desc='perform evaluation'):
            batch_gt_list = gt_batchs[i]
            batch_pred_list = pred_batchs[i]
            self.cal_batch_metrics(batch_gt_list, batch_pred_list)
        weights = np.array(self.n_nums)
        for metric in self.metrics.keys():
            self.metrics[metric] = np.sum(np.array(self.metrics[metric]) * weights) / np.sum(weights)
        with open(self.result_path, "w") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=4)



class CalculateRequirementPredictionMetrics(CalculateNLGMetrics):
    def __init__(self, batch_size, device, save_dir, generation_path, result_path='result.json', users=None):
        super().__init__(batch_size, device, save_dir, result_path)
        self.users = users
        self.generation_path = os.path.join(self.save_dir, generation_path)


    def get_data(self):
        gt_list = []
        pred_list = []

        with open(self.generation_path, 'r', encoding='utf-8') as f:
            generation_dict = json.load(f)
        for user_id in generation_dict.keys():
            if self.users != None and user_id not in self.users:
                continue
            for dialogue_id in generation_dict[user_id].keys():
                for topic_id in generation_dict[user_id][dialogue_id].keys():
                    gt_list.append(generation_dict[user_id][dialogue_id][topic_id]['ground_truth'])
                    pred_list.append(generation_dict[user_id][dialogue_id][topic_id]['generation'])

        gt_batchs = self.split_list(gt_list)
        pred_batchs = self.split_list(pred_list)
        return gt_batchs, pred_batchs


class CalculateRequirementPredictionScore(LLM_Sequential_Generation):
    def __init__(self, dataset_file_path, generation_path, save_dir, result_path='llm_score_result.json', users=None, model_name="gpt4o", *args, **kwargs):
        """
        - start_user_id & end_user_id: 当前批次生成哪些user的数据（两端均包含），可用于并行生成时。
        """
        prompt_template_dir = '/data5/hzp/MemPAL/perassist_framework/code/evaluation/requirement_prediction_evaluation_prompt'
        self.result_path = os.path.join(save_dir, result_path)
        self.generation_path = os.path.join(save_dir, generation_path)

        save_dir = os.path.join(save_dir, 'llm_evaluation')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        super().__init__(save_dir, model_name=model_name, *args, **kwargs)

        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)

        self.users = users
        self.dataset_file_path = dataset_file_path

        self.dataset_dict, self.generation_dict, self.sample_id_list = self.preprocess()
        
        
    def preprocess(self):
        processed_dataset_dict = {}
        sample_id_list = []
        user_id_list = []

        with open(self.dataset_file_path, 'r', encoding='utf-8') as f:
            dataset_dict = json.load(f)
        with open(self.generation_path, 'r', encoding='utf-8') as f:
            generation_dict = json.load(f)
        for user_id in generation_dict.keys():
            if self.users != None and user_id not in self.users:
                continue
            user_id_list.append(user_id)
        
        for user_id in user_id_list:
            processed_dataset_dict[user_id] = {}
            for dialogue_item in dataset_dict[user_id]['query']:
                dialogue_id = dialogue_item['sample_id']
                processed_dataset_dict[user_id][dialogue_id] = {}
                for topic_id in dialogue_item['topics'].keys():
                    sample_id = "{}_{}".format(dialogue_id, topic_id.split('-')[-1])
                    sample_id_list.append(sample_id)
                    processed_dataset_dict[user_id][dialogue_id][topic_id] = {}
                    processed_dataset_dict[user_id][dialogue_id][topic_id]['user_query'] = dialogue_item['topics'][topic_id]['user_query']
                    processed_dataset_dict[user_id][dialogue_id][topic_id]['implicit_needs'] = dialogue_item['topics'][topic_id]['implicit_needs']
                    processed_dataset_dict[user_id][dialogue_id][topic_id]['requirement'] = dialogue_item['topics'][topic_id]['requirement']

        return processed_dataset_dict, generation_dict, sample_id_list


    def get_prompt(self, sample_id):
        system_prompt = self.system_prompt_template
        user_id = sample_id.split('_')[0]
        dialogue_id = "_".join(sample_id.split('_')[:-1])
        topic_id = "topic-{}".format(sample_id.split('_')[-1])
        user_query_str = self.dataset_dict[user_id][dialogue_id][topic_id]['user_query']
        reference = json.dumps({"requirement": self.dataset_dict[user_id][dialogue_id][topic_id]['requirement'], "implicit_needs": self.dataset_dict[user_id][dialogue_id][topic_id]['implicit_needs']}, indent=4, ensure_ascii=False)
        prediction = self.generation_dict[user_id][dialogue_id][topic_id]['generation']
        user_prompt = self.user_prompt_template.replace('<user_query>', user_query_str).replace('<reference>', reference).replace('<prediction>', prediction)
        return system_prompt, user_prompt


    def check_generation(self, sample_id, sample_raw_generation):
        generation = self.generation_postprocess(sample_raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['analysis', 'score']:
                return False, "key error"
            if float(generation_dict['score']) not in [0.0, 0.5, 1.0, 1.5, 2.0]:
                return False, "wrong score"
        except:
            return False, "json format error"
        return True, None


    def extract_and_save(self):
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
                topic_id = "topic-{}".format(sample_id.split('_')[-1])

                if user_id not in result_dict.keys():
                    result_dict[user_id] = {}
                if dialogue_id not in result_dict[user_id].keys():
                    result_dict[user_id][dialogue_id] = {}
                result_dict[user_id][dialogue_id][topic_id] = {}
                result_dict[user_id][dialogue_id][topic_id]['analysis'] = generation_dict['analysis']
                result_dict[user_id][dialogue_id][topic_id]['score'] = generation_dict['score']

        save_path = os.path.join(self.save_dir, "evaluation_output.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)

        return result_dict


    def __call__(self):
        per_score_list = []
        score_list = []

        self.sequential_generate()
        evaluation_result_dict = self.extract_and_save()

        for user_id in evaluation_result_dict.keys():
            user_score_list = []
            for dialogue_id in evaluation_result_dict[user_id].keys():
                for topic_id in evaluation_result_dict[user_id][dialogue_id].keys():
                    user_score_list.append(evaluation_result_dict[user_id][dialogue_id][topic_id]['score'])

            user_score = sum(user_score_list) / len(user_score_list)
            per_score_list.append(user_score)
            score_list += user_score_list
        
        all_per_score = sum(per_score_list) / len(per_score_list)
        all_score = sum(score_list) / len(score_list)

        self.metrics = {'score': all_score, 'score_user_avg': all_per_score}
        with open(self.result_path, "w") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=4)


class CalculateSolutionQAPosMetrics(CalculateNLGMetrics):
    def __init__(self, batch_size, device, save_dir, generation_path, result_path='result.json', users=None):
        super().__init__(batch_size, device, save_dir, result_path)
        self.users = users
        self.generation_path = os.path.join(self.save_dir, generation_path)
        self.reference_num = 2 # 每个sample有2个pos ref


    def get_data(self):
        ref_list = []
        pred_list = []

        with open(self.generation_path, 'r', encoding='utf-8') as f:
            generation_dict = json.load(f)
        for user_id in generation_dict.keys():
            if self.users != None and user_id not in self.users:
                continue
            for dialogue_id in generation_dict[user_id].keys():
                for topic_id in generation_dict[user_id][dialogue_id].keys():
                    cur_pred = generation_dict[user_id][dialogue_id][topic_id]['generation']
                    ref_list += generation_dict[user_id][dialogue_id][topic_id]['references']['pos']
                    pred_list += [cur_pred] * 2

        ref_batchs = self.split_list(ref_list)
        pred_batchs = self.split_list(pred_list)
        return ref_batchs, pred_batchs


    def cal_batch_metrics(self, batch_gt_list, batch_pred_list):
        assert len(batch_gt_list) == len(batch_pred_list)
        for metric in self.metric_types:
            if metric == 'bleu':
                # bleu-1 ~ bleu-4
                for gt, pred in zip(batch_gt_list, batch_pred_list):
                    raw_bleu_scores = calculate_single_bleu(gt, pred)
                    for k in raw_bleu_scores.keys():
                        self.metrics[k].append(raw_bleu_scores[k])
            else:
                raise NotImplementedError("metric {} has not been implemented.".format(metric))


    def chunk_list(self, lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]


    def __call__(self):
        self.init_metrics_record()
        gt_batchs, pred_batchs = self.get_data()
        assert len(gt_batchs) == len(gt_batchs)
        for i in tqdm(range(len(gt_batchs)), desc='perform evaluation'):
            batch_gt_list = gt_batchs[i]
            batch_pred_list = pred_batchs[i]
            self.cal_batch_metrics(batch_gt_list, batch_pred_list)

        return_metrics = {}
        for k in self.metrics.keys():
            return_metrics[k] = []
            for chunk in self.chunk_list(self.metrics[k], self.reference_num):
                sample_metric = max(chunk[0], chunk[1])
                return_metrics[k].append(sample_metric)
            return_metrics[k] = np.sum(np.array(return_metrics[k])) * 1.0 / len(return_metrics[k])
        with open(self.result_path, "w") as f:
            json.dump(return_metrics, f, ensure_ascii=False, indent=4)



class CalculateSolutionSelectionMetrics(object):
    def __init__(self, save_dir, dataset_file_path, generation_path, result_path='result.json', users=None):
        self.save_dir = save_dir
        self.result_path = os.path.join(self.save_dir, result_path)
        self.users = users
        self.n_nums = []
        self.dataset_file_path = dataset_file_path
        self.generation_path = os.path.join(self.save_dir, generation_path)
    

    def preprocess(self):
        label_dict = {}
        new_generation_dict = {}

        with open(self.dataset_file_path, 'r', encoding='utf-8') as f:
            dataset_dict = json.load(f)
        with open(self.generation_path, 'r', encoding='utf-8') as f:
            generation_dict = json.load(f)
        for user_id in generation_dict.keys():
            if self.users != None and user_id not in self.users:
                continue
            label_dict[user_id] = {}
            new_generation_dict[user_id] = {}
        for user_id in label_dict.keys():
            for dialogue_item in dataset_dict[user_id]['query']:
                dialogue_id = dialogue_item['sample_id']
                for topic_id in dialogue_item['topics'].keys():
                    sample_id = "{}_{}".format(dialogue_id, topic_id)
                    new_generation_dict[user_id][sample_id] = deepcopy(generation_dict[user_id][dialogue_id][topic_id]['generation'])
                    candidate_solutions = dialogue_item['topics'][topic_id]['candidate_solutions']
                    sample_label_dict = {'pos': [], 'neg': []}
                    for i, solution_item in enumerate(candidate_solutions):
                        solution_id = 'S{}'.format(str(i+1))
                        if solution_item['feedback'] == 'pos':
                            sample_label_dict['pos'].append(solution_id)
                        elif solution_item['feedback'] == 'neg':
                            sample_label_dict['neg'].append(solution_id)
                    label_dict[user_id][sample_id] = deepcopy(sample_label_dict)

        return label_dict, new_generation_dict
    

    def __call__(self):
        per_score_list = []
        score_list = []

        label_dict, generation_dict = self.preprocess()
        for user_id in generation_dict.keys():
            user_score_list = []
            for sample_id in generation_dict[user_id].keys():
                sample_pred_list = generation_dict[user_id][sample_id]
                sample_label_dict = label_dict[user_id][sample_id]

                sample_score = 0.0
                for solution_id in sample_pred_list:
                    if solution_id in sample_label_dict['pos']:
                        sample_score += 1
                    elif solution_id in sample_label_dict['neg']:
                        sample_score -= 1
                
                user_score_list.append(sample_score)

            user_score = sum(user_score_list) / len(user_score_list)
            per_score_list.append(user_score)
            score_list += user_score_list
        
        all_per_score = sum(per_score_list) / len(per_score_list)
        all_score = sum(score_list) / len(score_list)

        self.metrics = {'score': all_score, 'score_user_avg': all_per_score}
        with open(self.result_path, "w") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=4)