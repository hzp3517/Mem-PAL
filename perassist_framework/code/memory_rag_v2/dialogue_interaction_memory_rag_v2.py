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
from copy import copy, deepcopy

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')

from llm_generation import LLM_Individual_Generation
from dialogue_evaluation.user_assistant_dialogue import User_Assistant_Dialogue


class RequirementAnalysisGeneration(LLM_Individual_Generation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.framework_root = '/data5/hzp/MemPAL/perassist_framework'


    def get_prompt(self, sample_id):
        user_id = sample_id.split('_')[0]
        topic_sample_id = '_'.join(sample_id.split('_')[:-1])
        cur_turn_id = 'turn_{}'.format(sample_id.split('_')[-1])

        prompt_template_dir = os.path.join(self.framework_root, 'prompt_template', 'memory_rag_v2', 'dialogue_interaction', 'requirement_analysis')
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        system_prompt = copy(self.system_prompt_template)
        
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)

        dialogue_context_list = []
        for i in range(1, int(cur_turn_id.split('_')[-1])):
            turn_id = "turn_{}".format(i)
            for role in ['user', 'assistant']:
                dialogue_context_list.append("- \"{}\": \"{}\"".format(role, deepcopy(self.dialogue_context_dict[turn_id][role])))
        dialogue_context_list.append("- \"{}\": \"{}\"".format('user', deepcopy(self.dialogue_context_dict[cur_turn_id]['user'])))
        dialogue_context_list.append("- \"{}\": <待生成>".format('assistant'))
        dialogue_context_str = '\n'.join(dialogue_context_list)
        user_prompt = self.user_prompt_template.replace("<dialogue_context>", dialogue_context_str)
        
        return system_prompt, user_prompt
        

    def check_generation(self, sample_id, raw_generation):
        generation = self.generation_postprocess(raw_generation)
        try:
            generation_dict = json.loads(generation)
            if list(generation_dict.keys()) != ['requirement']:
                return False, "wrong key"
            if generation_dict["requirement"].strip() == "" or generation_dict["requirement"] == "...":
                return False, "empty requirement"
        except json.JSONDecodeError as e:
            return False, "JSON format error"
        return True, None


    def generate(self, sample_id, dialogue_context_dict, record_prompt=False, *args, **kwargs):
        '''
        - dialogue_context_dict: 传入的应该是某个topic_sample_id下的内容
        return：success, output
        '''
        self.dialogue_context_dict = dialogue_context_dict
        return super().generate(sample_id, record_prompt=record_prompt, *args, **kwargs)




class MemoryRetrieval(object):
    def __init__(self, situation_dict, topic_schema_dict, situation_retrieval_top_k, topic_retrieval_top_k, save_dir, embedding_model_path, batch_size, device, start_user_id=None, end_user_id=None):
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.situation_dict = situation_dict
        self.topic_schema_dict = topic_schema_dict
        self.situation_retrieval_top_k = situation_retrieval_top_k
        self.topic_retrieval_top_k = topic_retrieval_top_k
        self.sample_id_list = self.preprocess(start_user_id, end_user_id)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.batch_size = batch_size
        self.device = device


    def preprocess(self, start_user_id, end_user_id):
        all_user_ids = list(self.situation_dict.keys())
        if start_user_id:
            start_user_idx = all_user_ids.index(start_user_id)
        else:
            start_user_idx = 0
        if end_user_id:
            end_user_idx = all_user_ids.index(end_user_id)
        else:
            end_user_idx = len(all_user_ids) - 1
        process_user_ids = all_user_ids[start_user_idx: end_user_idx+1]
        return process_user_ids


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


    def extract_memory_fts(self):
        # 抽取所有情境条目的特征
        situation_fts_file = os.path.join(self.save_dir, "situation_fts.h5")
        with h5py.File(situation_fts_file, 'w') as situation_fts_h5f:
            for user_id in tqdm(self.sample_id_list, desc="extract situation fts"):
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
        with h5py.File(requirement_fts_file, 'w') as requirement_fts_h5f:
            for user_id in tqdm(self.sample_id_list, desc="extract requirement fts"):
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
    

    def online_retrieval(self, query_sample_id, query_text):
        '''
        - query_sample_id: 0000_sample30_topic-1_1
        '''
        user_id = query_sample_id.split('_')[0]
        dialogue_id = '_'.join(query_sample_id.split('_')[:-2])
        query_ft = self.extract_fts([query_text])[0]

        retrieval_result_dict = {}

        # "situation"检索：query和所有当前sample中情境条目做语义相似度匹配，返回top-3条目。（如果当前sample少于3条情境就无需检索）
        with h5py.File(os.path.join(self.save_dir, "situation_fts.h5"), 'r') as situation_fts_h5f:
            retrieval_result_dict['situation'] = []
            start_key_id = dialogue_id + '_' + list(self.situation_dict[user_id][dialogue_id].keys())[0]
            end_key_id = dialogue_id + '_' + list(self.situation_dict[user_id][dialogue_id].keys())[-1]
            user_situation_id_list = [i.decode() for i in situation_fts_h5f[user_id]['ids'][()]]
            start_key_idx = user_situation_id_list.index(start_key_id)
            end_key_idx = user_situation_id_list.index(end_key_id)
            text_fts = situation_fts_h5f[user_id]['features'][()][start_key_idx: end_key_idx+1]
            if len(text_fts) <= self.situation_retrieval_top_k:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=len(text_fts))
            else:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=self.situation_retrieval_top_k)
            retrieved_situation_list = [user_situation_id_list[start_key_idx: end_key_idx+1][i] for i in retrieval_indices] # situation_id
            for situation_id in retrieved_situation_list:
                new_situation_id = "{}_situation{}".format(dialogue_id, situation_id.split('_')[-1])
                retrieval_result_dict['situation'].append(new_situation_id)

        # "topic"检索：需要将历史中所有的需求作为key，经验作为value。根据当前的需求query进行检索。返回top-3条目。
        with h5py.File(os.path.join(self.save_dir, "requirement_fts.h5"), 'r') as requirement_fts_h5f:
            retrieval_result_dict['topic'] = []
            cur_key_id = dialogue_id + '_' + '0'
            user_requirement_id_list = [i.decode() for i in requirement_fts_h5f[user_id]['ids'][()]]
            end_key_idx = user_requirement_id_list.index(cur_key_id) - 1
            text_fts = requirement_fts_h5f[user_id]['features'][()][0: end_key_idx+1]
            if len(text_fts) <= self.topic_retrieval_top_k:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=len(text_fts))
            else:
                retrieval_indices = self.retrieval(query_ft, text_fts, top_k=self.topic_retrieval_top_k)
            retrieved_topic_list = [user_requirement_id_list[0: end_key_idx+1][i] for i in retrieval_indices]
            for topic_id in retrieved_topic_list:
                retrieved_dialogue_id = "_".join(topic_id.split('_')[:-1])
                new_topic_id = "{}_topic{}".format(retrieved_dialogue_id, topic_id.split('_')[-1])
                retrieval_result_dict['topic'].append(new_topic_id)

        return retrieval_result_dict



class User_Assistant_Dialogue_MemoryRAGV2(User_Assistant_Dialogue):
    def __init__(self, embedding_model_path, situation_retrieval_top_k=3, topic_retrieval_top_k=3, batch_size=128, device='cpu', start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(start_user_id=start_user_id, end_user_id=end_user_id, *args, **kwargs)
        self.requirement_analysis_save_dir = os.path.join(self.save_dir, 'requirement_analysis')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.requirement_analysis_save_dir, exist_ok=True)
        self.memory_retrieval_save_dir = os.path.join(self.save_dir, 'memory_retrieval')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.memory_retrieval_save_dir, exist_ok=True)

        self.situation_dict, self.background_dict, self.topic_schema_dict, self.cluster_assignments_dict, self.principle_dict = self.load_assistant_input()
        self.requirement_analysis_generator = RequirementAnalysisGeneration(save_dir=self.requirement_analysis_save_dir, model_name=self.assistant_model_name)
        self.memory_retriever = MemoryRetrieval(self.situation_dict, self.topic_schema_dict, situation_retrieval_top_k, topic_retrieval_top_k, self.memory_retrieval_save_dir, embedding_model_path, batch_size, device, start_user_id, end_user_id)

        # memory encoding
        situation_fts_file = os.path.join(self.memory_retriever.save_dir, "situation_fts.h5")
        requirement_fts_file = os.path.join(self.memory_retriever.save_dir, "requirement_fts.h5")
        if os.path.exists(situation_fts_file) and os.path.exists(requirement_fts_file):
            print('Load memory fts from exist files.')
        else:
            print('Extract memory fts now...')
            self.memory_retriever.extract_memory_fts()

        self.retrieval_results_dict = {}


    def load_assistant_input(self):
        memory_rag_v2_data_dir = os.path.join(self.framework_root, 'data', self.model_dir_name, 'memory_rag_v2')
        situation_path = os.path.join(memory_rag_v2_data_dir, 'log_analysis', 'situation', 'situation.json')
        with open(situation_path, 'r', encoding='utf-8') as f:
            situation_dict = json.load(f)
        background_path = os.path.join(memory_rag_v2_data_dir, 'background_summary', 'background_summary.json')
        with open(background_path, 'r', encoding='utf-8') as f:
            background_dict = json.load(f)
        topic_schema_path = os.path.join(memory_rag_v2_data_dir, 'dialogue_analysis', 'requirement_refinement', 'requirement_refinement.json')
        with open(topic_schema_path, 'r', encoding='utf-8') as f:
            topic_schema_dict = json.load(f)
        cluster_assignments_path = os.path.join(memory_rag_v2_data_dir, 'principle', 'cluster_assignments.json')
        with open(cluster_assignments_path, 'r', encoding='utf-8') as f:
            cluster_assignments_dict = json.load(f)
        principle_path = os.path.join(memory_rag_v2_data_dir, 'principle', 'principle.json')
        with open(principle_path, 'r', encoding='utf-8') as f:
            principle_dict = json.load(f)
        return situation_dict, background_dict, topic_schema_dict, cluster_assignments_dict, principle_dict


    def create_log_files(self, record_prompt):
        super().create_log_files(record_prompt)
        
        # 需求分析的log_files初始化
        self.requirement_analysis_raw_path = os.path.join(self.requirement_analysis_save_dir, 'raw.csv')
        with open(self.requirement_analysis_raw_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'generation'])
        with open(self.requirement_analysis_generator.err_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'generation'])
        with open(self.requirement_analysis_generator.err_generation_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'generation'])
        err_log_file = open(self.requirement_analysis_generator.err_log_path, 'w')
        err_log_file.close()
        if record_prompt:
            requirement_analysis_prompt_log_file = open(self.requirement_analysis_generator.prompt_log_path, 'w')
            requirement_analysis_prompt_log_file.close()

        # 记忆检索结果文件初始化
        self.retrieval_result_log_path = os.path.join(self.memory_retriever.save_dir, 'retrieval_result.log')
        retrieval_result_log_file = open(self.retrieval_result_log_path, 'w')
        retrieval_result_log_file.close()


    def assistant_process_before_generation(self, sample_id, record_prompt):
        # 需求分析
        topic_sample_id = '_'.join(sample_id.split('_')[:-1])
        dialogue_context_dict = deepcopy(self.interaction_context_dict[topic_sample_id])
        success, cur_requirement = self.requirement_analysis_generator.generate(sample_id, dialogue_context_dict, record_prompt)
        if not success:
            cur_requirement = '无' # avoid the empty query
        with open(self.requirement_analysis_raw_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([sample_id, cur_requirement])

        # 记忆检索
        sample_retrieval_results_dict = self.memory_retriever.online_retrieval(sample_id, cur_requirement)
        with open(self.retrieval_result_log_path, 'a') as retrieval_result_log_file:
            retrieval_result_log_file.write("{} | {}\n".format(sample_id, json.dumps(sample_retrieval_results_dict, ensure_ascii=False)))
        self.retrieval_results_dict[sample_id] = copy(sample_retrieval_results_dict)


    def get_assistant_prompt(self, sample_id):
        self.memory_template_dict = {}
        self.memory_template_dict['requirement'] = {}
        self.memory_template_dict['requirement']['background_memory'] = "## 用户总体背景\n\"\"\"\n<background>\n\"\"\""
        self.memory_template_dict['requirement']['situation_memory'] = "## 用户近期情境经历\n\"\"\"\n<situation>\n\"\"\""
        self.memory_template_dict['requirement']['requirement_memory'] = "## 用户历史相关需求\n以下列举了一些交互历史中用户在与当前相似的情境下曾经出现过的需求，包括总体的需求类型以及具体的需求内容。\n\"\"\"\n总体需求类型：\n<general_requirement>\n具体需求内容：\n<requirements>\n\"\"\""
        self.memory_template_dict['solution'] = {}
        self.memory_template_dict['solution']['background_memory'] = "## 用户总体背景\n\"\"\"\n<background>\n\"\"\""
        self.memory_template_dict['solution']['situation_memory'] = "## 用户近期情境经历\n\"\"\"\n<situation>\n\"\"\""
        self.memory_template_dict['solution']['principle_memory'] = "## 用户偏好准则\n以下列举了一些用户在相关需求类型下的偏好准则。\n\"\"\"\n<principles>\n\"\"\""
        self.memory_template_dict['solution']['topic_memory'] = "## 用户历史偏好表现\n以下列举了一些交互历史中用户在与当前相似的需求下曾经对于不同方案的反馈，可作为当前的参考。\n\"\"\"\n<solutions>\n\"\"\""

        assistant_prompt_template_dir = os.path.join(self.framework_root, 'prompt_template', 'memory_rag_v2', 'dialogue_interaction', 'dialogue_generation')
        background_aspect_dict = {'work': '工作方面', 'health': '健康方面', 'family': '家庭方面', 'leisure': '休闲方面'}

        user_id = sample_id.split('_')[0]
        dialogue_id = '_'.join(sample_id.split('_')[:-2])
        topic_sample_id = '_'.join(sample_id.split('_')[:-1])
        cur_turn_id = 'turn_{}'.format(sample_id.split('_')[-1])
        cur_action = self.dialogue_template_dict[topic_sample_id][cur_turn_id]['assistant']
        assert cur_action in ['<需求推测>', '<方案提议>', '<方案讨论>', '<反馈回应>']
        
        with open(os.path.join(assistant_prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            system_prompt_template = ''.join(lines)
        system_prompt = copy(system_prompt_template)

        with open(os.path.join(assistant_prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            user_prompt_template = ''.join(lines)

        topic_item = self.dataset_dict[user_id][topic_sample_id]
        logs = ["- [{}] {}".format(log_item['timestamp'], log_item['content']) for log_item in topic_item['logs']]
        logs_str = "\n".join(logs)

        # 考虑导入memory
        memory_str_list = []

        if cur_action == "<需求推测>":
            situation_id_list = [i.split('_situation')[-1] for i in self.retrieval_results_dict[sample_id]['situation']]
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
            background_memory = self.memory_template_dict['requirement']['background_memory'].replace('<background>', background_str)
            memory_str_list.append(copy(background_memory))
            situation_memory = self.memory_template_dict['requirement']['situation_memory'].replace('<situation>', situation_str)
            memory_str_list.append(copy(situation_memory))
            
            requirement_str_list = []
            cluster_idx_list = []
            general_requirement_str_list = []
            for topic_id in self.retrieval_results_dict[sample_id]['topic']:
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
            requirement_memory = self.memory_template_dict['requirement']['requirement_memory'].replace('<general_requirement>', general_requirement_str).replace('<requirements>', requirement_str)
            memory_str_list.append(copy(requirement_memory))

        else:
            assert cur_action in ["<方案提议>", "<方案讨论>", "<反馈回应>"]

            situation_id_list = [i.split('_situation')[-1] for i in self.retrieval_results_dict[sample_id]['situation']]
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
            background_memory = self.memory_template_dict['requirement']['background_memory'].replace('<background>', background_str)
            memory_str_list.append(copy(background_memory))
            situation_memory = self.memory_template_dict['requirement']['situation_memory'].replace('<situation>', situation_str)
            memory_str_list.append(copy(situation_memory))

            principle_str_template = "### 准则<principle_idx>\n需求类型：\"<requirement>\"\n偏好准则：\"<preference>\""
            topic_str_template = "### 样例<topic_idx>\n需求内容：\"<requirement>\"\n<solutions>"
            solution_str_template = '<solution_idx>. \"<solution>\" [用户反馈：\"<user_feedback>\"]'
            topic_str_list = []
            cluster_idx_list = []
            principle_str_list = []
            for j, topic_id in enumerate(self.retrieval_results_dict[sample_id]['topic']):
                retrieved_dialogue_id, topic_idx = topic_id.split('_topic')
                last_dialogue_idx = list(self.principle_dict[user_id].keys()).index(dialogue_id) - 1
                last_dialogue_id = list(self.principle_dict[user_id].keys())[last_dialogue_idx]
                topic_idx = int(topic_idx)
                solution_list = self.topic_schema_dict[user_id][retrieved_dialogue_id][topic_idx]["solution_list"]
                processed_solution_dict = {}
                for solution_item in solution_list:
                    feedback_type = solution_item['feedback_type']
                    if feedback_type not in processed_solution_dict.keys():
                        processed_solution_dict[feedback_type] = []
                    processed_solution_dict[feedback_type].append({'solution': solution_item['solution'], 'user_feedback': solution_item['user_feedback']})
                solution_str_list = []
                if 'pos' in processed_solution_dict.keys():
                    tmp_solution_str_list = ['正向反馈方案：']
                    for i, solution_item in enumerate(processed_solution_dict['pos']):
                        tmp_solution_str_list.append(solution_str_template.replace('<solution_idx>', str(i+1)).replace('<solution>', processed_solution_dict['pos'][i]['solution']).replace('<user_feedback>', processed_solution_dict['pos'][i]['user_feedback']))
                    solution_str_list.append('\n'.join(tmp_solution_str_list))
                if 'neg' in processed_solution_dict.keys():
                    tmp_solution_str_list = ['负向反馈方案：']
                    for i, solution_item in enumerate(processed_solution_dict['neg']):
                        tmp_solution_str_list.append(solution_str_template.replace('<solution_idx>', str(i+1)).replace('<solution>', processed_solution_dict['neg'][i]['solution']).replace('<user_feedback>', processed_solution_dict['neg'][i]['user_feedback']))
                    solution_str_list.append('\n'.join(tmp_solution_str_list))
                if 'others' in processed_solution_dict.keys():
                    tmp_solution_str_list = ['其它方案：']
                    for i, solution_item in enumerate(processed_solution_dict['others']):
                        tmp_solution_str_list.append(solution_str_template.replace('<solution_idx>', str(i+1)).replace('<solution>', processed_solution_dict['others'][i]['solution']).replace('<user_feedback>', processed_solution_dict['others'][i]['user_feedback']))
                    solution_str_list.append('\n'.join(tmp_solution_str_list))
                tmp_topic_str = topic_str_template.replace('<topic_idx>', str(j+1)).replace('<requirement>', self.topic_schema_dict[user_id][retrieved_dialogue_id][topic_idx]["requirement"]).replace('<solutions>', '\n'.join(solution_str_list))
                topic_str_list.append(tmp_topic_str)
                cluster_idx = self.cluster_assignments_dict[user_id][retrieved_dialogue_id]['{}_topic{}'.format(retrieved_dialogue_id, str(topic_idx))]
                if cluster_idx not in cluster_idx_list:
                    cluster_idx_list.append(cluster_idx)
                    principle_str_list.append(principle_str_template.replace('<principle_idx>', str(len(cluster_idx_list))).replace('<requirement>', self.principle_dict[user_id][last_dialogue_id][cluster_idx]['requirement']).replace('<preference>', self.principle_dict[user_id][last_dialogue_id][cluster_idx]['preference']))
            topic_str = "\n\n".join(topic_str_list)
            principle_str = "\n".join(principle_str_list)
            principle_memory = self.memory_template_dict['solution']['principle_memory'].replace('<principles>', principle_str)
            memory_str_list.append(copy(principle_memory))
            solution_memory = self.memory_template_dict['solution']['topic_memory'].replace('<solutions>', topic_str)
            memory_str_list.append(copy(solution_memory))
            
        memory_str = '\n\n'.join(memory_str_list)
        user_prompt_template = user_prompt_template.replace('<memory>', memory_str)

        # 导入当前对话上文及当前句action
        dialogue_context_list = []
        current_turn_list = []
        for i in range(1, int(cur_turn_id.split('_')[-1])):
            turn_id = "turn_{}".format(i)
            for role in ['user', 'assistant']:
                dialogue_context_list.append("- \"{}\": \"{}\"".format(role, self.interaction_context_dict[topic_sample_id][turn_id][role]))
        dialogue_context_list.append("- \"{}\": \"{}\"".format('user', self.interaction_context_dict[topic_sample_id][cur_turn_id]['user']))
        current_turn_list.append("- \"{}\": \"{}\"".format('user', self.interaction_context_dict[topic_sample_id][cur_turn_id]['user']))
        dialogue_context_list.append("- \"{}\": {}".format('assistant', cur_action))
        current_turn_list.append("- \"{}\": {}".format('assistant', cur_action))
        dialogue_context_str = '\n'.join(dialogue_context_list)
        current_turn_str = '\n'.join(current_turn_list)

        user_prompt = user_prompt_template.replace('<dialogue_context>', dialogue_context_str).replace('<current_turn>', current_turn_str).replace('<action>', "{}".format(cur_action)).replace('<action_description>', self.assistant_action_description_dict[cur_action])
        return system_prompt, user_prompt



if __name__ == '__main__':
    import torch.multiprocessing as mp

    user_model = 'qwen2.5-max'
    assistant_model = 'qwen_max'


    # ------------
    device='cuda:2'
    embedding_model_path = 'pretrained_models/paraphrase-multilingual-mpnet-base-v2'
    framework_root = '/data5/hzp/MemPAL/perassist_framework'
    data_dir = assistant_model
    save_dir = os.path.join(data_dir, 'memory_rag_v2', 'dialogue_interaction')

    dialogue_save_path = 'dialogue.json'
    dialogue_generator = User_Assistant_Dialogue_MemoryRAGV2(embedding_model_path, situation_retrieval_top_k=3, topic_retrieval_top_k=3, batch_size=128, device=device, user_model_name=user_model, assistant_model_name=assistant_model, save_dir=save_dir)

    dialogue_generator.interaction_generate()
    # dialogue_generator.interaction_generate(record_prompt=True)
    dialogue_generator.extract_and_save(dialogue_save_path)
    # -----------------