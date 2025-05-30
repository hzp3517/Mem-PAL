"""
pip install faiss-gpu

- 输入
    - memory（检索到的背景、情境、需求）
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
from sentence_transformers import SentenceTransformer
import faiss
import h5py
import copy
import re

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')

from llm_generation import LLM_Sequential_Generation
from evaluation.perform_evaluation import CalculateRequirementPredictionMetrics
from evaluation.perform_evaluation import CalculateRequirementPredictionScore



class MemoryRetrieval(object):
    def __init__(self, dataset_dict, situation_dict, topic_schema_dict, save_dir, embedding_model_path, batch_size, device):
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.situation_dict = situation_dict
        self.topic_schema_dict = topic_schema_dict
        self.dataset_dict, self.sample_id_list = self.preprocess_dataset(dataset_dict)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.batch_size = batch_size
        self.device = device


    def preprocess_dataset(self, dataset_dict):
        processed_dataset_dict = {}
        sample_id_list = []
        for user_id in dataset_dict.keys():
            processed_dataset_dict[user_id] = {}
            for dialogue_item in dataset_dict[user_id]['query']:
                dialogue_id = dialogue_item['sample_id']
                processed_dataset_dict[user_id][dialogue_id] = {}
                processed_dataset_dict[user_id][dialogue_id]['topics'] = dialogue_item['topics']
                for topic_id in dialogue_item['topics'].keys():
                    sample_id = "{}_{}".format(dialogue_id, topic_id.split('-')[-1]) # e.g. 0000_sample0_1 (代表该dialogue的topic_1)
                    sample_id_list.append(sample_id)
        return processed_dataset_dict, sample_id_list


    def extract_fts(self, text_list, use_tqdm=False, desc_info=None):
        if len(text_list):
            desc = 'extract fts: {}'.format(desc_info) if desc_info else 'extract fts'
            encoded_text = []
            if use_tqdm:
                for idx in tqdm(range(0, len(text_list), self.batch_size), desc=desc):
                    with torch.no_grad():
                        encoded_text.append(self.embedding_model.encode(text_list[idx:idx+self.batch_size], batch_size=self.batch_size, device=device))
            else:
                for idx in range(0, len(text_list), self.batch_size):
                    with torch.no_grad():
                        encoded_text.append(self.embedding_model.encode(text_list[idx:idx+self.batch_size], batch_size=self.batch_size, device=device))
            encoded_text = np.concatenate(encoded_text)
        else:
            encoded_text = np.array([])
        return encoded_text


    def extract_and_save_fts(self):
        # 抽取query的特征
        query_fts_file = os.path.join(self.save_dir, "query_fts.h5")
        query_fts_h5f = h5py.File(query_fts_file, 'w')
        for user_id in tqdm(list(self.dataset_dict.keys()), desc="extract query fts"):
            user_query_fts_h5f = query_fts_h5f.create_group(user_id)
            user_sample_id_lists = []
            user_text_lists = []
            for dialogue_id in self.dataset_dict[user_id].keys():
                for topic_id in self.dataset_dict[user_id][dialogue_id]['topics'].keys():
                    user_sample_id_lists.append("{}_{}".format(dialogue_id, topic_id.split('-')[-1]))
                    user_text_lists.append(self.dataset_dict[user_id][dialogue_id]['topics'][topic_id]['user_query'])
            user_query_fts_h5f['ids'] = user_sample_id_lists
            user_query_fts_h5f['features'] = self.extract_fts(user_text_lists)

        # 抽取所有情境条目的特征
        situation_fts_file = os.path.join(self.save_dir, "situation_fts.h5")
        situation_fts_h5f = h5py.File(situation_fts_file, 'w')
        for user_id in tqdm(self.situation_dict.keys(), desc="extract situation fts"):
            user_sample_id_list = []
            user_text_list = []
            for sample_id in self.situation_dict[user_id].keys():
                for situation_id in self.situation_dict[user_id][sample_id].keys():
                    user_sample_id_list.append("{}_{}".format(sample_id, situation_id))
                    user_text_list.append(self.situation_dict[user_id][sample_id][situation_id]['situation'])
            user_situation_fts_h5f = situation_fts_h5f.create_group(user_id)
            user_situation_fts_h5f['ids'] = user_sample_id_list
            user_situation_fts_h5f['features'] = self.extract_fts(user_text_list)

        # 抽取所有需求的特征
        requirement_fts_file = os.path.join(self.save_dir, "requirement_fts.h5")
        requirement_fts_h5f = h5py.File(requirement_fts_file, 'w')
        for user_id in tqdm(self.topic_schema_dict.keys(), desc="extract requirement fts"):
            user_sample_id_list = []
            user_text_list = []
            for sample_id in self.topic_schema_dict[user_id].keys():
                for topic_idx, topic_item in enumerate(self.topic_schema_dict[user_id][sample_id]):
                    user_sample_id_list.append("{}_{}".format(sample_id, str(topic_idx)))
                    user_text_list.append(topic_item["requirement"])
            user_requirement_fts_h5f = requirement_fts_h5f.create_group(user_id)
            user_requirement_fts_h5f['ids'] = user_sample_id_list
            user_requirement_fts_h5f['features'] = self.extract_fts(user_text_list)


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


    def save_retrieval_result(self):
        query_fts_h5f = h5py.File(os.path.join(self.save_dir, "query_fts.h5"), 'r')
        situation_fts_h5f = h5py.File(os.path.join(self.save_dir, "situation_fts.h5"), 'r')
        requirement_fts_h5f = h5py.File(os.path.join(self.save_dir, "requirement_fts.h5"), 'r')
        query_fts_dict = {}
        query_fts_dict['ids'] = []
        query_fts_dict['features'] = []
        for user_id in query_fts_h5f.keys():
            id_list = [i.decode() for i in query_fts_h5f[user_id]['ids'][()]]
            feature_list = query_fts_h5f[user_id]['features'][()]
            for i in range(len(id_list)):
                query_fts_dict['ids'].append(id_list[i])
                query_fts_dict['features'].append(feature_list[i])

        retrieval_results_file = os.path.join(self.save_dir, "retrieval_results.json")
        retrieval_result_dict = {}

        # "situation"检索：query和所有当前sample中情境条目做语义相似度匹配，返回top-3条目。（如果当前sample少于3条情境就无需检索）
        aspect = 'situation'
        top_k = 3
        retrieval_result_dict[aspect] = {}
        for i, retrieval_id in tqdm(enumerate(query_fts_dict['ids']), total=len(query_fts_dict['ids']), desc="retrieve for {}".format(aspect)):
            query_ft = query_fts_dict['features'][i]
            user_id = retrieval_id.split('_')[0]
            dialogue_id = '_'.join(retrieval_id.split('_')[:-1])
            start_key_id = dialogue_id + '_' + list(self.situation_dict[user_id][dialogue_id].keys())[0]
            end_key_id = dialogue_id + '_' + list(self.situation_dict[user_id][dialogue_id].keys())[-1]
            user_situation_id_list = [i.decode() for i in situation_fts_h5f[user_id]['ids'][()]]
            start_key_idx = user_situation_id_list.index(start_key_id)
            end_key_idx = user_situation_id_list.index(end_key_id)
            text_fts = situation_fts_h5f[user_id]['features'][()][start_key_idx: end_key_idx+1]
            if len(text_fts) <= top_k:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=len(text_fts))
            else:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=top_k)
            # retrieved_situation_list = sorted([user_situation_id_list[start_key_idx: end_key_idx+1][i] for i in retrieval_indices], key=lambda x: int(x.split('_')[-1])) # situation_id
            retrieved_situation_list = [user_situation_id_list[start_key_idx: end_key_idx+1][i] for i in retrieval_indices] # situation_id
            retrieval_result_dict[aspect][retrieval_id] = []
            for situation_id in retrieved_situation_list:
                new_situation_id = "{}_situation{}".format(dialogue_id, situation_id.split('_')[-1])
                retrieval_result_dict[aspect][retrieval_id].append(new_situation_id)

        # "requirement"检索：将历史需求作为key。根据当前的情境query进行检索。返回top-3条目。
        aspect = 'requirement'
        top_k = 3
        retrieval_result_dict[aspect] = {}
        for i, retrieval_id in tqdm(enumerate(query_fts_dict['ids']), total=len(query_fts_dict['ids']), desc="retrieve for {}".format(aspect)):
            query_ft = query_fts_dict['features'][i]
            user_id = retrieval_id.split('_')[0]
            dialogue_id = '_'.join(retrieval_id.split('_')[:-1])
            cur_key_id = dialogue_id + '_' + '0'
            user_requirement_id_list = [i.decode() for i in requirement_fts_h5f[user_id]['ids'][()]]
            end_key_idx = user_requirement_id_list.index(cur_key_id) - 1
            text_fts = requirement_fts_h5f[user_id]['features'][()][0: end_key_idx+1]
            if len(text_fts) <= top_k:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=len(text_fts))
            else:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=top_k)
            retrieved_topic_list = [user_requirement_id_list[0: end_key_idx+1][i] for i in retrieval_indices]
            retrieval_result_dict[aspect][retrieval_id] = []
            for topic_id in retrieved_topic_list:
                retrieved_dialogue_id = "_".join(topic_id.split('_')[:-1])
                new_topic_id = "{}_topic{}".format(retrieved_dialogue_id, topic_id.split('_')[-1])
                retrieval_result_dict[aspect][retrieval_id].append(new_topic_id)

        with open(retrieval_results_file, 'w', encoding='utf-8') as f:
            json.dump(retrieval_result_dict, f, indent=4, ensure_ascii=False)



class RequirementPredictionMemoryRAGV2(LLM_Sequential_Generation):
    def __init__(self, dataset_dict, situation_dict, background_dict, topic_schema_dict, cluster_assignments_dict, principle_dict, retrieval_results_dict, prompt_template_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        """
        - start_user_id & end_user_id: 当前批次生成哪些user的数据（两端均包含），可用于并行生成时。
        """
        super().__init__(*args, **kwargs)
        self.dataset_dict, self.sample_id_list = self.preprocess_dataset(dataset_dict, start_user_id, end_user_id)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)

        self.situation_dict = situation_dict
        self.background_dict = background_dict
        self.topic_schema_dict = topic_schema_dict
        self.cluster_assignments_dict = cluster_assignments_dict
        self.principle_dict = principle_dict
        self.retrieval_results_dict = retrieval_results_dict

        self.situation_memory_template = "## 用户相关背景与近期情境\n以下为该用户相关方面的总体背景以及近期情境。\n\"\"\"\n总体背景：\n<background>\n近期情境：\n<situation>\n\"\"\""
        self.requirement_memory_template = "## 用户历史相关需求\n以下列举了一些交互历史中用户在与当前相似的情境下曾经出现过的需求，包括总体的需求类型以及具体的需求内容。\n\"\"\"\n总体需求类型：\n<general_requirement>\n具体需求内容：\n<requirements>\n\"\"\""
    

    def preprocess_dataset(self, dataset_dict, start_user_id, end_user_id):
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
                processed_dataset_dict[user_id][dialogue_id]['topics'] = dialogue_item['topics']
                for topic_id in dialogue_item['topics'].keys():
                    sample_id = "{}_{}".format(dialogue_id, topic_id.split('-')[-1]) # e.g. 0000_sample0_1 (代表该dialogue的topic_1)
                    sample_id_list.append(sample_id)
        return processed_dataset_dict, sample_id_list


    def get_prompt(self, sample_id):
        background_aspect_dict = {'work': '工作方面', 'health': '健康方面', 'family': '家庭方面', 'leisure': '休闲方面'}

        system_prompt = self.system_prompt_template

        user_id = sample_id.split('_')[0]
        dialogue_id = '_'.join(sample_id.split('_')[:-1])
        cur_topic_id = "topic-{}".format(sample_id.split('_')[-1])

        dialogue_item = self.dataset_dict[user_id][dialogue_id]

        # 考虑导入memory
        user_prompt_template = copy.deepcopy(self.user_prompt_template)
        pattern = re.compile(r'\$\$\{(.*?)\$\$\}', re.DOTALL)
        
        user_prompt_template = re.sub(pattern, r'\1', copy.copy(user_prompt_template)) # 保留 $${ 和 $$} 之间的内容
        memory_str_list = []

        assert sample_id in self.retrieval_results_dict['situation'].keys()
        situation_id_list = [i.split('_situation')[-1] for i in self.retrieval_results_dict['situation'][sample_id]]
        situation_str_list = []
        aspect_list = []
        background_str_list = []
        for situation_idx in situation_id_list:
            situation_str_list.append("- \"{}\"".format(self.situation_dict[user_id][dialogue_id][situation_idx]['situation']))
            for aspect in self.situation_dict[user_id][dialogue_id][situation_idx]['situation_aspects']:
                if aspect not in aspect_list:
                    aspect_list.append(aspect)
                    background_str_list.append("- {}：\"{}\"".format(background_aspect_dict[aspect], self.background_dict[user_id][dialogue_id][aspect]))
        situation_str = "\n".join(situation_str_list)
        background_str = "\n".join(background_str_list)
        situation_memory = self.situation_memory_template.replace('<background>', background_str).replace('<situation>', situation_str)
        memory_str_list.append(situation_memory)

        assert sample_id in self.retrieval_results_dict['requirement'].keys()
        requirement_str_list = []
        cluster_idx_list = []
        general_requirement_str_list = []
        for topic_id in self.retrieval_results_dict['requirement'][sample_id]:
            retrieved_dialogue_id, topic_idx = topic_id.split('_topic')
            last_dialogue_idx = list(self.principle_dict[user_id].keys()).index(dialogue_id) - 1
            last_dialogue_id = list(self.principle_dict[user_id].keys())[last_dialogue_idx]
            topic_idx = int(topic_idx)
            requirement_str_list.append('- \"{}\"'.format(self.topic_schema_dict[user_id][retrieved_dialogue_id][topic_idx]["requirement"]))
            cluster_idx = self.cluster_assignments_dict[user_id][retrieved_dialogue_id]['{}_topic{}'.format(retrieved_dialogue_id, str(topic_idx))]
            if cluster_idx not in cluster_idx_list:
                cluster_idx_list.append(cluster_idx)
                general_requirement_str_list.append("- \"{}\"".format(self.principle_dict[user_id][last_dialogue_id][cluster_idx]['requirement']))
        requirement_str = "\n".join(requirement_str_list)
        general_requirement_str = "\n".join(general_requirement_str_list)
        requirement_memory = self.requirement_memory_template.replace('<general_requirement>', general_requirement_str).replace('<requirements>', requirement_str)
        memory_str_list.append(requirement_memory)

        memory_str = '\n\n'.join(memory_str_list)
        user_prompt_template = user_prompt_template.replace('<memory>', memory_str)

        user_query = dialogue_item['topics'][cur_topic_id]['user_query']
        
        user_prompt = user_prompt_template.replace('<user_query>', user_query)
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
    device = 'cuda:2'
    model_name = 'qwen_max'


    dataset_path = '/data5/hzp/MemPAL/data_synthesis_v2/data/input.json'
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_dict = json.load(f)
    framework_root = '/data5/hzp/MemPAL/perassist_framework'
    data_dir = model_name
    prompt_dir = os.path.join(framework_root, 'prompt_template')
    embedding_model_path = 'pretrained_models/paraphrase-multilingual-mpnet-base-v2'


    # # ### ----- Step 1: 记忆检索 -----
    # situation_path = os.path.join(data_dir, 'memory_rag_v2', 'log_analysis', 'situation', 'situation.json')
    # topic_schema_path = os.path.join(data_dir, 'memory_rag_v2', 'dialogue_analysis', 'requirement_refinement', 'requirement_refinement.json')
    # with open(situation_path, 'r', encoding='utf-8') as f:
    #     situation_dict = json.load(f)
    # with open(topic_schema_path, 'r', encoding='utf-8') as f:
    #     topic_schema_dict = json.load(f)
    # retrieval_save_dir = os.path.join(data_dir, 'memory_rag_v2', 'requirement_prediction', 'memory_retrieval')
    # memory_retriever = MemoryRetrieval(dataset_dict, situation_dict, topic_schema_dict, save_dir=retrieval_save_dir, embedding_model_path=embedding_model_path, batch_size=128, device=device)
    # memory_retriever.extract_and_save_fts() # 抽取query和key特征
    # memory_retriever.save_retrieval_result() # 进行检索
    # # ### ---------------------------



    # ### ----- Step 2: 检索增强生成 -----
    situation_path = os.path.join(data_dir, 'memory_rag_v2', 'log_analysis', 'situation', 'situation.json')
    with open(situation_path, 'r', encoding='utf-8') as f:
        situation_dict = json.load(f)
    background_path = os.path.join(data_dir, 'memory_rag_v2', 'background_summary', 'background_summary.json')
    with open(background_path, 'r', encoding='utf-8') as f:
        background_dict = json.load(f)
    topic_schema_path = os.path.join(data_dir, 'memory_rag_v2', 'dialogue_analysis', 'requirement_refinement', 'requirement_refinement.json')
    with open(topic_schema_path, 'r', encoding='utf-8') as f:
        topic_schema_dict = json.load(f)
    cluster_assignments_path = os.path.join(data_dir, 'memory_rag_v2', 'principle', 'cluster_assignments.json')
    with open(cluster_assignments_path, 'r', encoding='utf-8') as f:
        cluster_assignments_dict = json.load(f)
    principle_path = os.path.join(data_dir, 'memory_rag_v2', 'principle', 'principle.json')
    with open(principle_path, 'r', encoding='utf-8') as f:
        principle_dict = json.load(f)

    retrieval_results_path = os.path.join(data_dir, 'memory_rag_v2', 'requirement_prediction', 'memory_retrieval', 'retrieval_results.json')
    with open(retrieval_results_path, 'r', encoding='utf-8') as f:
        retrieval_results_dict = json.load(f)
    
    prompt_template_dir = os.path.join(prompt_dir, 'memory_rag_v2', 'requirement_prediction')
    save_dir = os.path.join(data_dir, 'memory_rag_v2', 'requirement_prediction')
    output_save_path = 'output.json'
    perassist = RequirementPredictionMemoryRAGV2(dataset_dict, situation_dict, background_dict, topic_schema_dict, cluster_assignments_dict, principle_dict, retrieval_results_dict, prompt_template_dir, save_dir=save_dir, model_name=model_name)

    # # # check the prompt
    # # system_prompt, user_prompt = perassist.get_prompt("0000_sample30_1")
    # system_prompt, user_prompt = perassist.get_prompt("0003_sample19_1")
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    # perassist.sequential_generate()
    # perassist.extract_and_save(output_save_path)

    calculator = CalculateRequirementPredictionMetrics(batch_size=128, device="cuda:0", save_dir=save_dir, generation_path=output_save_path, result_path='result.json')
    calculator()

    # calculator = CalculateRequirementPredictionScore(dataset_path, generation_path=output_save_path, save_dir=save_dir, result_path='llm_score_result.json')
    # system_prompt, user_prompt = calculator.get_prompt("0000_sample30_2")
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))
    # calculator()
    ### -------------------------------