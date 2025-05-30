"""
Stage 1: log analysis
- ① 输入: 当前情境的logs；输出: log graph（带有因果、时序和相似主题关联连边）
- ② 输入：每个子图；输出：子图对应的situation及其方面类型
"""
import os
import json
import requests
import ast
import csv
from tqdm import tqdm
import pandas as pd
from copy import copy
import random
import time
import math

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')

from llm_generation import LLM_Sequential_Generation


class LogGraphGeneration(LLM_Sequential_Generation):
    def __init__(self, dataset_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = 10 # 一次处理多少条logs
        self.logs_dict, self.sample_id_list = self.preprocess(dataset_dict, start_user_id, end_user_id, self.window_size)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)
        self.tolerable_error_type_list = ["wrong \"caused_by\" log_id", "wrong \"follows\" log_id"]

    
    def preprocess(self, dataset_dict, start_user_id, end_user_id, window_size):
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

        logs_dict = {}
        for user_id in process_user_ids:
            for set_name in ['history', 'query']:
                for sample_item in dataset_dict[user_id][set_name]:
                    sample_id = sample_item['sample_id']
                    sample_logs_num = len(sample_item['logs'])
                    for i in range(int(math.ceil(sample_logs_num*1.0/window_size))):
                        if i == 0:
                            start_log_idx = 0
                        else:
                            start_log_idx = (i-1) * window_size
                        if i == int(math.ceil(sample_logs_num*1.0/window_size)):
                            end_log_idx = sample_logs_num - 1
                        else:
                            end_log_idx = (i+1)*window_size - 1
                        log_sample_id = "{}_{}".format(sample_id, str(i))
                        logs_dict[log_sample_id] = {}
                        cur_log_idx = copy(start_log_idx)
                        for log_item in sample_item['logs'][start_log_idx: end_log_idx+1]:
                            logs_dict[log_sample_id]['log_{}'.format(str(cur_log_idx))] = copy(log_item)
                            cur_log_idx += 1

        sample_id_list = list(logs_dict.keys())
        return logs_dict, sample_id_list


    def get_prompt(self, sample_id):
        logs_dict = self.logs_dict[sample_id]
        start_analysis_log_id = "log_{}".format(str(self.window_size * int(sample_id.split('_')[-1])))
        end_log_id = list(logs_dict.keys())[-1]
        logs_id_span = "{} ~ {}".format(start_analysis_log_id, end_log_id)
        logs_str = json.dumps(logs_dict, ensure_ascii=False)
        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.replace('<start_log_id>', start_analysis_log_id).replace('<logs_id_span>', logs_id_span).replace('<logs>', logs_str)
        return system_prompt, user_prompt


    def check_generation(self, sample_id, sample_raw_generation):
        sample_log_dict = self.logs_dict[sample_id]
        valid_relation_log_id_list = list(sample_log_dict.keys())
        
        start_analysis_log_id = "log_{}".format(str(self.window_size * int(sample_id.split('_')[-1])))
        start_analysis_idx = valid_relation_log_id_list.index(start_analysis_log_id)
        valid_analysis_log_id_list = valid_relation_log_id_list[start_analysis_idx:]

        generation = self.generation_postprocess(sample_raw_generation)
        try: # 确保生成内容符合json格式
            generation_dict = json.loads(generation)

            # 1. 确保每条日志均进行了分析
            if list(generation_dict.keys()) != valid_analysis_log_id_list:
                return False, "wrong log id"

            # 2. 确保每个日志均包含"caused_by", "follows"
            for log_id in valid_analysis_log_id_list:
                if list(generation_dict[log_id].keys()) != ["caused_by", "follows"]:
                    return False, "wrong relation keys"
                
                # 3. 确保关联的log属于log_id_list，且在时序上位于当前log之前
                for related_log_id in generation_dict[log_id]["caused_by"]:
                    if (related_log_id not in valid_relation_log_id_list) or (int(related_log_id.split('_')[-1]) >= int(log_id.split('_')[-1])):
                        return False, "wrong \"caused_by\" log_id"
                for related_log_id in generation_dict[log_id]["follows"]:
                    if (related_log_id not in valid_relation_log_id_list) or (int(related_log_id.split('_')[-1]) >= int(log_id.split('_')[-1])):
                        return False, "wrong \"follows\" log_id"

        except json.JSONDecodeError as e:
            return False, "JSON format error"

        return True, None


    def data_postprocess(self, sample_id, sample_dict):
        '''
        check_generation中的第三点要求对于有些sample即使多次生成也难以实现，因此通过后处理的方式解决
        '''
        sample_log_dict = self.logs_dict[sample_id]
        valid_relation_log_id_list = list(sample_log_dict.keys())
        
        start_analysis_log_id = "log_{}".format(str(self.window_size * int(sample_id.split('_')[-1])))
        start_analysis_idx = valid_relation_log_id_list.index(start_analysis_log_id)
        valid_analysis_log_id_list = valid_relation_log_id_list[start_analysis_idx:]

        for log_id in valid_analysis_log_id_list:
            valid_caused_by_list = []
            valid_follows_list = []
            for related_log_id in sample_dict[log_id]["caused_by"]:
                if (related_log_id not in valid_relation_log_id_list) or (int(related_log_id.split('_')[-1]) >= int(log_id.split('_')[-1])):
                    continue
                else:
                    valid_caused_by_list.append(related_log_id)
            for related_log_id in sample_dict[log_id]["follows"]:
                if (related_log_id not in valid_relation_log_id_list) or (int(related_log_id.split('_')[-1]) >= int(log_id.split('_')[-1])):
                    continue
                else:
                    valid_follows_list.append(related_log_id)
            sample_dict[log_id]["caused_by"] = valid_caused_by_list
            sample_dict[log_id]["follows"] = valid_follows_list
        return sample_dict


    def extract_and_save(self, save_path):
        generation_dict = {}
        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                log_sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)
                generation_dict[log_sample_id] = generation

        result_dict = {}
        for log_sample_id in generation_dict.keys():
            user_id = log_sample_id.split('_')[0]
            sample_id = '_'.join(log_sample_id.split('_')[:-1])
            if user_id not in result_dict.keys():
                result_dict[user_id] = {}
            if sample_id not in result_dict[user_id].keys():
                result_dict[user_id][sample_id] = {}
            generation = generation_dict[log_sample_id]
            generation_json = json.loads(generation)
            generation_json = self.data_postprocess(log_sample_id, generation_json)
            result_dict[user_id][sample_id].update(generation_json)

        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)


class SituationGeneration(LLM_Sequential_Generation):
    def __init__(self, dataset_dict, log_graph_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_subgraph_dict, self.sample_id_list = self.preprocess(dataset_dict, log_graph_dict, start_user_id, end_user_id)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)


    def find_connected_subgraphs(self, graph):
        # 先构建一个双向的邻接表，确保无向图的连通性
        adj_list = {}
        
        for node in graph:
            if node not in adj_list:
                adj_list[node] = set()
            for neighbor in graph[node]:
                adj_list[node].add(neighbor)
                if neighbor not in adj_list:
                    adj_list[neighbor] = set()
                adj_list[neighbor].add(node)
        
        # 用于存储已访问的节点
        visited = set()
        
        # 用于存储所有连通子图
        connected_components = []
        
        def dfs(node, component):
            """深度优先搜索遍历图，找到所有连通的节点"""
            visited.add(node)
            component.append(node)
            
            # 遍历当前节点的所有邻接节点
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        # 遍历每个节点，检查是否已经被访问过
        for node in graph:
            if node not in visited:
                component = []
                dfs(node, component)
                sorted_component = sorted(component, key=lambda x: int(x.split('_')[-1])) # 将一个子图内的logs按id顺序（时序）排序
                connected_components.append(sorted_component)
        
        return connected_components


    def preprocess(self, dataset_dict, log_graph_dict, start_user_id, end_user_id):
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

        logs_dict = {}
        for user_id in process_user_ids:
            for set_name in ['history', 'query']:
                for sample_item in dataset_dict[user_id][set_name]:
                    cnt_log = 0
                    log_sample_id = sample_item['sample_id']
                    logs_dict[log_sample_id] = {}
                    for log_item in sample_item['logs']:
                        log_id = 'log_{}'.format(str(cnt_log))
                        if len(log_graph_dict[user_id][log_sample_id][log_id]['caused_by']) > 0:
                            log_item['caused_by'] = log_graph_dict[user_id][log_sample_id][log_id]['caused_by']
                        elif len(log_graph_dict[user_id][log_sample_id][log_id]['follows']) > 0:
                            log_item['follows'] = log_graph_dict[user_id][log_sample_id][log_id]['follows']
                        logs_dict[log_sample_id][log_id] = log_item
                        cnt_log += 1

        log_subgraph_dict = {}
        for user_id in process_user_ids:
            for log_sample_id in log_graph_dict[user_id].keys():
                user_id = log_sample_id.split('_')[0]
                if user_id not in process_user_ids:
                    continue
                graph = {}
                for log_id in log_graph_dict[user_id][log_sample_id]:
                    graph[log_id] = list(set(log_graph_dict[user_id][log_sample_id][log_id]['caused_by'] + log_graph_dict[user_id][log_sample_id][log_id]['follows']))
                subgraph_list = self.find_connected_subgraphs(graph)
                for i, subgraph in enumerate(subgraph_list):
                    situation_id = '{}_situation{}'.format(log_sample_id, str(i))
                    log_subgraph_dict[situation_id] = {}
                    for log_id in subgraph:
                        log_subgraph_dict[situation_id][log_id] = logs_dict[log_sample_id][log_id]

        sample_id_list = list(log_subgraph_dict.keys())
        return log_subgraph_dict, sample_id_list


    def get_prompt(self, sample_id):
        system_prompt = self.system_prompt_template
        logs_str = json.dumps(self.log_subgraph_dict[sample_id], ensure_ascii=False)
        user_prompt = self.user_prompt_template.replace('<subgraph>', logs_str)
        return system_prompt, user_prompt


    def check_generation(self, sample_id, sample_raw_generation):
        generation = self.generation_postprocess(sample_raw_generation)
        try: # 确保生成内容符合json格式
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['situation', 'situation_aspects']:
                return False, "wrong keys"
            if generation_dict['situation'].strip() == "" or generation_dict['situation'] == "...":
                return False, "empty situation"
            if not isinstance(generation_dict['situation_aspects'], list):
                return False, "list format error"
            for aspect in generation_dict['situation_aspects']:
                if aspect not in ['work', 'health', 'family', 'leisure']:
                    return False, "wrong aspect"
        except json.JSONDecodeError as e:
            return False, "JSON format error"
        return True, None


    def extract_and_save(self, save_path):
        generation_dict = {}
        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                situation_sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)
                generation_json = json.loads(generation)
                sample_id, situation_id = situation_sample_id.split('_situation')
                user_id = sample_id.split('_')[0]
                if user_id not in generation_dict.keys():
                    generation_dict[user_id] = {}
                if sample_id not in generation_dict[user_id].keys():
                    generation_dict[user_id][sample_id] = {}
                generation_json['corr_log_ids'] = list(self.log_subgraph_dict[situation_sample_id].keys()) # 可以不加
                generation_dict[user_id][sample_id][situation_id] = copy(generation_json)

        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(generation_dict, f, indent=4, ensure_ascii=False)





if __name__ == '__main__':
    model_name = 'qwen_max'

    dataset_path = '/data5/hzp/MemPAL/data_synthesis_v2/data/input.json'
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_dict = json.load(f)
    framework_root = '/data5/hzp/MemPAL/perassist_framework'
    data_dir = model_name
    prompt_dir = os.path.join(framework_root, 'prompt_template')


    ### ----- Step 1: 生成log graph -----
    prompt_template_dir = os.path.join(prompt_dir, 'memory_rag_v2', 'log_analysis', 'log_graph')
    log_graph_dir = os.path.join(data_dir, 'memory_rag_v2', 'log_analysis', 'log_graph')
    raw_file_name = 'log_graph_raw.csv'
    log_graph_file_path = 'log_graph.json'
    log_graph_data_generator = LogGraphGeneration(dataset_dict, prompt_template_dir, save_dir=log_graph_dir, raw_file_name=raw_file_name, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = log_graph_data_generator.get_prompt('0000_sample2_1')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    log_graph_data_generator.sequential_generate()
    log_graph_data_generator.extract_and_save(log_graph_file_path)
    ### -------------------------


    ### ----- Step 2：对log graph中的每个全连通子图生成对应situation -----
    prompt_template_dir = os.path.join(prompt_dir, 'memory_rag_v2', 'log_analysis', 'situation')
    log_graph_dir = os.path.join(data_dir, 'memory_rag_v2', 'log_analysis', 'log_graph')
    log_graph_file_path = 'log_graph.json'
    with open(os.path.join(log_graph_dir, log_graph_file_path), 'r', encoding='utf-8') as f:
        log_graph_dict = json.load(f)
    situation_dir = os.path.join(data_dir, 'memory_rag_v2', 'log_analysis', 'situation')
    raw_file_name = 'situation_raw.csv'
    situation_file_path = 'situation.json'
    situation_data_generator = SituationGeneration(dataset_dict, log_graph_dict, prompt_template_dir, save_dir=situation_dir, raw_file_name=raw_file_name, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = situation_data_generator.get_prompt('0000_sample0_situation0')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    situation_data_generator.sequential_generate()
    situation_data_generator.extract_and_save(situation_file_path)
    ### -------------------------