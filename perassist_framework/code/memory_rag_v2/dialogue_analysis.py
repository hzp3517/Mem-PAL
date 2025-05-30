"""
Stage 3: dialogue analysis
- ① 输入: dialogue；输出: topic schemas
- ② 输入：requirement；输出：refined requirement
"""
import os
import json
import ast
import csv
from tqdm import tqdm
import pandas as pd
from copy import copy, deepcopy
import torch
import numpy as np
import h5py
from sentence_transformers import SentenceTransformer
import faiss

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')

from llm_generation import LLM_Sequential_Generation


class TopicSchemaGeneration(LLM_Sequential_Generation):
    def __init__(self, dataset_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dialogues_dict, self.sample_id_list = self.preprocess(dataset_dict, start_user_id, end_user_id)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)
        self.tolerable_error_type_list = ["bad turn_span"]


    def preprocess(self, dataset_dict, start_user_id, end_user_id):
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

        dialogues_dict = {}
        for user_id in process_user_ids:
            for set_name in ['history', 'query']:
                for sample_item in dataset_dict[user_id][set_name]:
                    sample_id = sample_item['sample_id']
                    dialogues_dict[sample_id] = {}
                    for turn_id in sample_item['dialogue'].keys():
                        dialogues_dict[sample_id][turn_id] = {}
                        for role in ['user', 'assistant']:
                            dialogues_dict[sample_id][turn_id][role] = sample_item['dialogue'][turn_id][role]["content"]
        
        sample_id_list = list(dialogues_dict.keys())
        return dialogues_dict, sample_id_list


    def get_prompt(self, sample_id):
        system_prompt = self.system_prompt_template
        dialogue_str = json.dumps(self.dialogues_dict[sample_id], indent=4, ensure_ascii=False)
        user_prompt = self.user_prompt_template.replace('<dialogue>', dialogue_str)
        return system_prompt, user_prompt


    def check_generation(self, sample_id, sample_raw_generation):
        generation = self.generation_postprocess(sample_raw_generation)
        try: # 确保生成内容符合json格式
            generation_list = json.loads(generation)
            sample_turn_num = len(self.dialogues_dict[sample_id].keys())
            turn_span_list = []
            for topic_item in generation_list:
                # 确保每个item的key正确
                if list(topic_item.keys()) != ["turn_span", "requirement", "solution_list", "preference"]:
                    return False, "wrong topic keys"
                if not isinstance(topic_item['turn_span'], list):
                    return False, "wrong turn_span format"
                elif len(topic_item['turn_span']) != 2 or int(topic_item['turn_span'][1].split('_')[-1]) < int(topic_item['turn_span'][0].split('_')[-1]):
                    return False, "invalid turn_span content"
                turn_span_list.append(topic_item['turn_span'])
                if topic_item['requirement'] == '...' or topic_item['requirement'].strip() == '':
                    return False, "empty requirement"
                if topic_item['preference'] == '...' or topic_item['preference'].strip() == '':
                    return False, "empty preference"
                if not isinstance(topic_item['solution_list'], list):
                    return False, "wrong solution_list format"
                for solution_item in topic_item['solution_list']:
                    if list(solution_item.keys()) != ["solution", "user_feedback", "feedback_type"]:
                        return False, "wrong solution keys"
                    if solution_item['solution'] == '...' or solution_item['solution'].strip() == '':
                        return False, "empty solution"
                    if solution_item['user_feedback'] == '...' or solution_item['user_feedback'].strip() == '':
                        return False, "empty user_feedback"
                    if solution_item['feedback_type'] not in ['pos', 'neg', 'others']:
                        return False, "wrong feedback type"
            cur_turn_id = 1
            for turn_span in turn_span_list:
                if int(turn_span[0].split('_')[-1]) != cur_turn_id:
                    return False, "bad turn_span"
                cur_turn_id = int(turn_span[1].split('_')[-1]) + 1
            if cur_turn_id != sample_turn_num + 1:
                return False, "bad turn_span"
        except json.JSONDecodeError as e:
            return False, "JSON format error"
        return True, None


    def extract_and_save(self, save_path):
        generation_dict = {}
        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)
                generation_dict[sample_id] = generation

        result_dict = {}
        for sample_id in generation_dict.keys():
            generation = generation_dict[sample_id]
            generation_json = json.loads(generation)
            user_id = sample_id.split('_')[0]
            if user_id not in result_dict.keys():
                result_dict[user_id] = {}
            result_dict[user_id][sample_id] = generation_json

        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)


class RequirementRefinement(LLM_Sequential_Generation):
    def __init__(self, background_dict, situation_dict, topic_schema_dict, prompt_template_dir, embedding_model_path, retrieval_top_k, batch_size, device, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_id_list = self.preprocess(topic_schema_dict, start_user_id, end_user_id)
        self.background_dict = background_dict
        self.situation_dict = situation_dict
        self.topic_schema_dict = topic_schema_dict
        self.embedding_model = SentenceTransformer(embedding_model_path)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)
        self.batch_size = batch_size
        self.device = device

        retrieval_results_file = os.path.join(self.save_dir, "retrieval_results.json")
        if os.path.exists(retrieval_results_file):
            print('Load retrieval results from exist files.')
        else:
            print('Perform retrieval now...')
            self.extract_and_save_fts()
            self.save_retrieval_result(retrieval_top_k, retrieval_results_file)
        with open(retrieval_results_file, 'r', encoding='utf-8') as f:
            self.retrieval_results_dict = json.load(f)


    def preprocess(self, topic_schema_dict, start_user_id, end_user_id):
        all_user_ids = list(topic_schema_dict.keys())
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
            for dialogue_id in topic_schema_dict[user_id].keys():
                for i, topic_item in enumerate(topic_schema_dict[user_id][dialogue_id]):
                    sample_id_list.append("{}_topic{}".format(dialogue_id, str(i)))
        return sample_id_list


    def extract_fts(self, text_list, use_tqdm=False, desc_info=None):
        if len(text_list):
            desc = 'extract fts: {}'.format(desc_info) if desc_info else 'extract fts'
            encoded_text = []
            if use_tqdm:
                for idx in tqdm(range(0, len(text_list), self.batch_size), desc=desc):
                    with torch.no_grad():
                        encoded_text.append(self.embedding_model.encode(text_list[idx:idx+self.batch_size], batch_size=self.batch_size, device=self.device))
            else:
                for idx in range(0, len(text_list), self.batch_size):
                    with torch.no_grad():
                        encoded_text.append(self.embedding_model.encode(text_list[idx:idx+self.batch_size], batch_size=self.batch_size, device=self.device))
            encoded_text = np.concatenate(encoded_text)
        else:
            encoded_text = np.array([])
        return encoded_text


    def extract_and_save_fts(self):
        # 抽取query的特征
        query_fts_file = os.path.join(self.save_dir, "query_fts.h5")
        query_fts_h5f = h5py.File(query_fts_file, 'w')
        for user_id in tqdm(list(self.topic_schema_dict.keys()), desc="extract query fts"):
            user_query_fts_h5f = query_fts_h5f.create_group(user_id)
            user_sample_id_list = []
            user_text_list = []
            for dialogue_id in self.topic_schema_dict[user_id].keys():
                for i, topic_item in enumerate(self.topic_schema_dict[user_id][dialogue_id]):
                    topic_id = "{}_topic{}".format(dialogue_id, str(i))
                    user_sample_id_list.append(topic_id)
                    user_text_list.append(topic_item["requirement"])
            user_query_fts_h5f['ids'] = user_sample_id_list
            user_query_fts_h5f['features'] = self.extract_fts(user_text_list)

        # 抽取situation的特征
        situation_fts_file = os.path.join(self.save_dir, "situation_fts.h5")
        situation_fts_h5f = h5py.File(situation_fts_file, 'w')
        for user_id in tqdm(list(self.situation_dict.keys()), desc="extract situation fts"):
            user_situation_fts_h5f = situation_fts_h5f.create_group(user_id)
            user_sample_id_list = []
            user_text_list = []
            for dialogue_id in self.situation_dict[user_id].keys():
                for k in self.situation_dict[user_id][dialogue_id].keys():
                    situation_id = "{}_situation{}".format(dialogue_id, k)
                    user_sample_id_list.append(situation_id)
                    user_text_list.append(self.situation_dict[user_id][dialogue_id][k]["situation"])
            user_situation_fts_h5f['ids'] = user_sample_id_list
            user_situation_fts_h5f['features'] = self.extract_fts(user_text_list)


    def retrieval(self, query_ft, text_fts, top_k):
        '''
        - query_fts: (dim,)
        - text_fts: (len, dim)
        '''
        xq = query_ft.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(xq) # 使用 faiss.normalize_L2 对query_ft进行归一化
        xb = text_fts.astype(np.float32)
        faiss.normalize_L2(xb)
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
        similarity, indices = index.search(xq, top_k) # indices: [[list of index]]
        return indices[0]


    def save_retrieval_result(self, retrieval_top_k, retrieval_results_file):
        query_fts_h5f = h5py.File(os.path.join(self.save_dir, "query_fts.h5"), 'r')
        situation_fts_h5f = h5py.File(os.path.join(self.save_dir, "situation_fts.h5"), 'r')
        retrieval_results_dict = {}

        query_fts_dict = {}
        query_fts_dict['ids'] = []
        query_fts_dict['features'] = []
        for user_id in query_fts_h5f.keys():
            id_list = [i.decode() for i in query_fts_h5f[user_id]['ids'][()]]
            feature_list = query_fts_h5f[user_id]['features'][()]
            for i in range(len(id_list)):
                query_fts_dict['ids'].append(id_list[i])
                query_fts_dict['features'].append(feature_list[i])

        for i, retrieval_id in tqdm(enumerate(query_fts_dict['ids']), total=len(query_fts_dict['ids']), desc="retrieve for situation"):
            query_ft = query_fts_dict['features'][i]
            user_id = retrieval_id.split('_')[0]
            dialogue_id = '_'.join(retrieval_id.split('_')[:-1])
            start_key_id = dialogue_id + '_situation' + list(self.situation_dict[user_id][dialogue_id].keys())[0]
            end_key_id = dialogue_id + '_situation' + list(self.situation_dict[user_id][dialogue_id].keys())[-1]
            user_situation_id_list = [i.decode() for i in situation_fts_h5f[user_id]['ids'][()]]
            start_key_idx = user_situation_id_list.index(start_key_id)
            end_key_idx = user_situation_id_list.index(end_key_id)
            text_fts = situation_fts_h5f[user_id]['features'][()][start_key_idx: end_key_idx+1]
            if len(text_fts) <= retrieval_top_k:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=len(text_fts))
            else:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=retrieval_top_k)
            retrieved_situation_list = [user_situation_id_list[start_key_idx: end_key_idx+1][i] for i in retrieval_indices] # situation_id
            retrieval_results_dict[retrieval_id] = []
            for situation_id in retrieved_situation_list:
                retrieval_results_dict[retrieval_id].append(situation_id)

        with open(retrieval_results_file, 'w', encoding='utf-8') as f:
            json.dump(retrieval_results_dict, f, indent=4, ensure_ascii=False)


    def get_prompt(self, sample_id):
        background_aspect_dict = {'work': '工作方面', 'health': '健康方面', 'family': '家庭方面', 'leisure': '休闲方面'}
        user_id = sample_id.split('_')[0]
        dialogue_id = "_".join(sample_id.split('_')[:-1])
        topic_idx = int(sample_id.split('_topic')[-1])
        situation_id_list = [i.split('_situation')[-1] for i in self.retrieval_results_dict[sample_id]]
        situation_str_list = []
        aspect_list = []
        background_str_list = []
        for situation_idx in situation_id_list:
            situation_str_list.append("- \"{}\"".format(self.situation_dict[user_id][dialogue_id][situation_idx]['situation']))
            for aspect in self.situation_dict[user_id][dialogue_id][situation_idx]['situation_aspects']:
                if aspect not in aspect_list and aspect in self.background_dict[user_id][dialogue_id].keys():
                    aspect_list.append(aspect)
                    background_str_list.append("- {}：\"{}\"".format(background_aspect_dict[aspect], self.background_dict[user_id][dialogue_id][aspect]))
        situation_str = "\n".join(situation_str_list)
        if len(aspect_list) == 0:
            background_str = '无'
        else:
            background_str = "\n".join(background_str_list)
        requirement = json.dumps(self.topic_schema_dict[user_id][dialogue_id][topic_idx]['requirement'], indent=4, ensure_ascii=False)
        user_prompt_template = deepcopy(self.user_prompt_template)
        system_prompt = self.system_prompt_template
        user_prompt = user_prompt_template.replace('<background>', background_str).replace('<situation>', situation_str).replace('<requirement>', requirement)
        return system_prompt, user_prompt
        

    def check_generation(self, sample_id, sample_raw_generation):
        generation = self.generation_postprocess(sample_raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['requirement']:
                return False, "key error"
            if generation_dict['requirement'] == "..." or generation_dict['requirement'].strip() == "":
                return False, "empty response"
        except:
            return False, "json format error"
        return True, None


    def extract_and_save(self, save_path): # 替换原来的schema中的requirement内容
        generation_dict = {}
        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)
                generation_dict[sample_id] = generation
        # result_dict = deepcopy(self.topic_schema_dict)
        result_dict = {}
        for sample_id in generation_dict.keys():
            generation = generation_dict[sample_id]
            generation_json = json.loads(generation)
            user_id = sample_id.split('_')[0]
            if user_id not in result_dict.keys():
                result_dict[user_id] = {}
            dialogue_id = "_".join(sample_id.split('_')[:-1])
            if dialogue_id not in result_dict[user_id].keys():
                result_dict[user_id][dialogue_id] = []
            topic_idx = int(sample_id.split('_topic')[-1])
            assert len(result_dict[user_id][dialogue_id]) == topic_idx
            tmp_sample_dict = deepcopy(self.topic_schema_dict[user_id][dialogue_id][topic_idx])
            tmp_sample_dict['requirement'] = copy(generation_json['requirement'])
            result_dict[user_id][dialogue_id].append(copy(tmp_sample_dict))
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
    embedding_model_path = 'pretrained_models/paraphrase-multilingual-mpnet-base-v2'


    ### ----- Step 1: 生成topic_schema -----
    prompt_template_dir = os.path.join(prompt_dir, 'memory_rag_v2', 'dialogue_analysis', 'topic_schema')
    topic_schema_dir = os.path.join(data_dir, 'memory_rag_v2', 'dialogue_analysis', 'topic_schema')
    raw_file_name = 'topic_schema_raw.csv'
    topic_schema_file_path = 'topic_schema.json'
    topic_schema_data_generator = TopicSchemaGeneration(dataset_dict, prompt_template_dir, save_dir=topic_schema_dir, raw_file_name=raw_file_name, model_name=model_name)

    # check the prompt
    system_prompt, user_prompt = topic_schema_data_generator.get_prompt('0000_sample0')
    print('【system prompt】:\n{}'.format(system_prompt))
    print('【user prompt】:\n{}'.format(user_prompt))

    topic_schema_data_generator.sequential_generate()
    topic_schema_data_generator.extract_and_save(topic_schema_file_path)
    ### -------------------------


    ### ----- Step 2: 生成requirement_refinement -----
    device = 'cuda:0'
    retrieval_top_k = 3
    batch_size = 128
    prompt_template_dir = os.path.join(prompt_dir, 'memory_rag_v2', 'dialogue_analysis', 'requirement_refinement')
    requirement_refinement_dir = os.path.join(data_dir, 'memory_rag_v2', 'dialogue_analysis', 'requirement_refinement')
    raw_file_name = 'requirement_refinement_raw.csv'
    requirement_refinement_file_path = 'requirement_refinement.json'
    background_dir = os.path.join(data_dir, 'memory_rag_v2', 'background_summary')
    background_file_path = 'background_summary.json'
    with open(os.path.join(background_dir, background_file_path), 'r', encoding='utf-8') as f:
        background_dict = json.load(f)
    situation_dir = os.path.join(data_dir, 'memory_rag_v2', 'log_analysis', 'situation')
    situation_file_path = 'situation.json'
    with open(os.path.join(situation_dir, situation_file_path), 'r', encoding='utf-8') as f:
        situation_dict = json.load(f)
    topic_schema_dir = os.path.join(data_dir, 'memory_rag_v2', 'dialogue_analysis', 'topic_schema')
    topic_schema_file_path = 'topic_schema.json'
    with open(os.path.join(topic_schema_dir, topic_schema_file_path), 'r', encoding='utf-8') as f:
        topic_schema_dict = json.load(f)
    requirement_refinement_data_generator = RequirementRefinement(background_dict, situation_dict, topic_schema_dict, prompt_template_dir, embedding_model_path, retrieval_top_k, batch_size, device, save_dir=requirement_refinement_dir, raw_file_name=raw_file_name, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = requirement_refinement_data_generator.get_prompt('0000_sample0_topic0')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    requirement_refinement_data_generator.sequential_generate()
    requirement_refinement_data_generator.extract_and_save(requirement_refinement_file_path)
    ### -------------------------