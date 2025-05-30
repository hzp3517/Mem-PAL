"""
Stage 4: preference extraction
- ① 对所有dialogue requirement抽特征
- ② 将每个user的所有history set的需求聚类，随后对每个聚簇提取出一条总体需求类型或偏好准则
- ③ 对每个query sample的具体需求内容，先将其分配到最相近的聚簇上并更新该聚簇；随后判断该聚簇对应的总体需求类型或偏好准则是否需要更新，给出更新后的内容
"""
import os
import json
import ast
import csv
import random
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import h5py
from copy import copy, deepcopy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import sys
sys.path.append('/data5/hzp/MemPAL/perassist_framework/code')

from llm_generation import LLM_Sequential_Generation



class RequirementFeatureExtraction():
    def __init__(self, dataset_dict, topic_schema_dict, embedding_model_path, batch_size, device):
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.batch_size = batch_size
        self.device = device
        self.topic_id_dict, self.topic_id_list, self.requirement_list = self.preprocess(dataset_dict, topic_schema_dict)


    def preprocess(self, dataset_dict, topic_schema_dict):
        topic_id_dict = {}
        topic_id_list = []
        requirement_list = []
        for user_id in dataset_dict.keys():
            topic_id_dict[user_id] = {}
            for set_name in ['history', 'query']:
                topic_id_dict[user_id][set_name] = []
                for sample_item in dataset_dict[user_id][set_name]:
                    sample_id = sample_item['sample_id']
                    for topic_idx, topic_item in enumerate(topic_schema_dict[user_id][sample_id]):
                        topic_id = "{}_topic{}".format(sample_id, str(topic_idx))
                        topic_id_dict[user_id][set_name].append(topic_id)
                        topic_id_list.append(topic_id)
                        requirement_list.append(topic_item["requirement"])
        return topic_id_dict, topic_id_list, requirement_list


    def extract_fts(self):
        encoded_text = []
        for idx in tqdm(range(0, len(self.requirement_list), self.batch_size), desc='extract fts'):
            with torch.no_grad():
                encoded_text.append(self.embedding_model.encode(self.requirement_list[idx:idx+self.batch_size], batch_size=self.batch_size, device=device))
        encoded_text = np.concatenate(encoded_text)
        return encoded_text


    def extract_and_save_fts(self, save_dir):
        """
        requirement_ft.h5:
        - <user_id>
            - <set_name>
                - "topic_ids": ["0000_sample0_topic0", ...]
                - "features": <npy arrary: (len, dim)>
        """
        fts = self.extract_fts()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'requirement_ft.h5')
        save_h5f = h5py.File(save_path, 'w')

        for user_id in tqdm(list(self.topic_id_dict.keys()), desc='save fts'):
            user_h5f = save_h5f.create_group(user_id)
            for set_name in self.topic_id_dict[user_id].keys():
                set_h5f = user_h5f.create_group(set_name)
                set_h5f['topic_ids'] = self.topic_id_dict[user_id][set_name]
                start_idx = self.topic_id_list.index(self.topic_id_dict[user_id][set_name][0])
                end_idx = self.topic_id_list.index(self.topic_id_dict[user_id][set_name][-1])
                set_h5f['features'] = fts[start_idx: end_idx+1]



class PrincipleAbstract(LLM_Sequential_Generation):
    def __init__(self, topic_schema_dict, requirement_fts_h5f, prompt_template_root, n_clusters, max_input_samples, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topic_schema_dict = topic_schema_dict
        self.requirement_fts_h5f = requirement_fts_h5f
        self.n_clusters = n_clusters
        self.max_input_samples = max_input_samples
        with open(os.path.join(prompt_template_root, 'requirement_abstract', 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.requirement_system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_root, 'requirement_abstract', 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.requirement_user_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_root, 'preference_abstract', 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.preference_system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_root, 'preference_abstract', 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.preference_user_prompt_template = ''.join(lines)

        cluster_centers_save_path = os.path.join(self.save_dir, 'cluster_centers.h5')
        cluster_samples_save_path = os.path.join(self.save_dir, 'cluster_samples.json')
        if os.path.exists(cluster_centers_save_path) and os.path.exists(cluster_samples_save_path):
            print('Load cluster from exist files.')
        else:
            print('Perform clustering now...')
            self.history_clustering(cluster_centers_save_path, cluster_samples_save_path)
        with open(cluster_samples_save_path, 'r', encoding='utf-8') as f:
            cluster_samples_dict = json.load(f)

        self.cluster_dict, self.sample_id_list = self.preprocess(cluster_samples_dict, start_user_id, end_user_id)
        self.general_requirement_dict = {}

    
    def history_clustering(self, cluster_centers_save_path, cluster_samples_save_path, min_samples=2, retry_num=100):
        """
        - `min_samples`: 每个cluster中最少的sample数，如果过少就重新聚类
        - `retry_num`: 最多尝试重新聚类的次数
        """
        cluster_centers_h5f = h5py.File(cluster_centers_save_path, 'w')
        cluster_samples_dict = {}

        for user_id in tqdm(list(self.requirement_fts_h5f.keys()), desc='history clustering'):
            cluster_samples_dict[user_id] = []

            retry_count = 0
            while True:
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(self.requirement_fts_h5f[user_id]['history']['features'][()])

                labels = kmeans.labels_
                unique, counts = np.unique(labels, return_counts=True)

                # 检查是否有簇的样本数小于 min_samples
                if (not all(count >= min_samples for count in counts)) and (retry_count < retry_num - 1):
                    retry_count += 1
                    continue
                else:
                    if retry_count >= retry_num - 1:
                        print('user {} have retried {} times'.format(user_id, retry_num))

                    cluster_centers = kmeans.cluster_centers_
                    cluster_centers_h5f[user_id] = cluster_centers

                    for cluster_idx in range(self.n_clusters):
                        cluster_ids = np.where(labels == cluster_idx)[0]
                        cluster_dict_list = []
                        for topic_idx in cluster_ids:
                            topic_id = self.requirement_fts_h5f[user_id]["history"]["topic_ids"][()][topic_idx].decode()
                            user_id = topic_id.split('_')[0]
                            sample_id = topic_id.split('_topic')[0]
                            topic_idx = int(topic_id.split('_topic')[-1])
                            requirement_content = self.topic_schema_dict[user_id][sample_id][topic_idx]["requirement"]
                            preference_content = self.topic_schema_dict[user_id][sample_id][topic_idx]["preference"]
                            cluster_dict_list.append({"topic_id": topic_id, "requirement": requirement_content, "preference": preference_content})
                        cluster_samples_dict[user_id].append(cluster_dict_list)
                    break

        with open(cluster_samples_save_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_samples_dict, f, indent=4, ensure_ascii=False)


    def preprocess(self, cluster_samples_dict, start_user_id, end_user_id):
        all_user_ids = list(cluster_samples_dict.keys())
        if start_user_id:
            start_user_idx = all_user_ids.index(start_user_id)
        else:
            start_user_idx = 0
        if end_user_id:
            end_user_idx = all_user_ids.index(end_user_id)
        else:
            end_user_idx = len(all_user_ids) - 1
        process_user_ids = all_user_ids[start_user_idx: end_user_idx+1]

        cluster_dict = {}
        sample_id_list = []
        for user_id in process_user_ids:
            for cluster_idx, cluster_item in enumerate(cluster_samples_dict[user_id]):
                cluster_id = "{}_cluster{}".format(user_id, str(cluster_idx))
                cluster_dict[cluster_id] = [{"requirement": copy(topic_item["requirement"]), "preference": copy(topic_item["preference"])} for topic_item in cluster_item]
                if len(cluster_dict[cluster_id]) > self.max_input_samples:
                    cluster_dict[cluster_id] = random.sample(cluster_dict[cluster_id], self.max_input_samples)
                else:
                    random.shuffle(cluster_dict[cluster_id])
                sample_id_list += ['requirement_{}'.format(cluster_id), 'preference_{}'.format(cluster_id)]
        return cluster_dict, sample_id_list


    def get_prompt(self, sample_id):
        generation_type, cluster_id = sample_id.split('_', 1)
        cluster_str = json.dumps([topic_item[generation_type] for topic_item in self.cluster_dict[cluster_id]], indent=4, ensure_ascii=False)
        if generation_type == 'requirement':
            system_prompt = self.requirement_system_prompt_template
            user_prompt = self.requirement_user_prompt_template.replace('<requirements>', cluster_str)
        else:
            assert generation_type == 'preference'
            system_prompt = self.preference_system_prompt_template
            user_prompt = self.preference_user_prompt_template.replace('<general_requirement>', self.general_requirement_dict[cluster_id]).replace('<preferences>', cluster_str)
        return system_prompt, user_prompt

        
    def check_generation(self, sample_id, sample_raw_generation):
        generation_type, cluster_id = sample_id.split('_', 1)
        generation = self.generation_postprocess(sample_raw_generation)
        try:
            generation_dict = json.loads(generation)
            if generation_type == 'requirement':
                if "general_requirement" not in generation_dict.keys():
                    return False, "key error"
                if generation_dict["general_requirement"].strip() == "" or generation_dict["general_requirement"] == "...":
                    return False, "empty content"
            else:
                assert generation_type == 'preference'
                if "principle" not in generation_dict.keys():
                    return False, "key error"
                if generation_dict["principle"].strip() == "" or generation_dict["principle"] == "...":
                    return False, "empty content"
        except:
            return False, "json format error"
        return True, None


    def postprocess_for_iterative_generation(self, sample_id, success, generation):
        # 记录self.general_requirement_dict
        generation_type, cluster_id = sample_id.split('_', 1)
        if generation_type == 'requirement':
            if success:
                generation_json = json.loads(generation)
                self.general_requirement_dict[cluster_id] = generation_json['general_requirement']


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
            generation_type, cluster_id = sample_id.split('_', 1)
            cluster_idx = int(cluster_id.split("_cluster")[-1])
            user_id = cluster_id.split('_cluster')[0]
            if user_id not in result_dict.keys():
                result_dict[user_id] = [{"requirement": None, "preference": None} for _ in range(self.n_clusters)]
            generation = generation_dict[sample_id]
            generation_json = json.loads(generation)
            if generation_type == 'requirement':
                general_requirement = generation_json['general_requirement']
                result_dict[user_id][cluster_idx]["requirement"] = general_requirement
            else:
                assert generation_type == 'preference'
                principle = generation_json['principle']
                result_dict[user_id][cluster_idx]["preference"] = principle

        for user_id in result_dict.keys():
            for cluster_item in result_dict[user_id]:
                assert (cluster_item["requirement"] != None) and (cluster_item["preference"] != None)

        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4, ensure_ascii=False)


    def cluster_assignment(self, user_id, query_ft):
        cluster_centers_save_path = os.path.join(self.save_dir, 'cluster_centers.h5')
        cluster_samples_save_path = os.path.join(self.save_dir, 'cluster_samples.json')
        cluster_centers_h5f = h5py.File(cluster_centers_save_path, 'r')
        with open(cluster_samples_save_path, 'r', encoding='utf-8') as f:
            cluster_samples_dict = json.load(f)

        clusters_dict = {}
        for user_id in cluster_centers_h5f.keys():
            clusters_dict[user_id] = {}
            clusters_dict[user_id]['ft'] = cluster_centers_h5f[user_id][()]
            clusters_dict[user_id]['num'] = np.array([len(cluster_item) for cluster_item in cluster_samples_dict[user_id]])

        cluster_centers = np.copy(clusters_dict[user_id]['ft'])
        distances = np.linalg.norm(cluster_centers - np.expand_dims(query_ft, axis=0), axis=1) # 计算新数据点与所有聚类中心的欧几里得距离
        assigned_cluster_idx = int(np.argmin(distances))
        return assigned_cluster_idx


    def make_cluster_assignment(self):
        cluster_assignments_dict = {}
        for user_id in self.requirement_fts_h5f.keys():
            topic_id_list = [i.decode() for i in self.requirement_fts_h5f[user_id]['history']['topic_ids'][()]]
            cluster_assignments_dict[user_id] = {}
            user_history_fts = self.requirement_fts_h5f[user_id]['history']['features'][()]
            for topic_idx, topic_id in enumerate(topic_id_list):
                sample_id = topic_id.split('_topic')[0]
                if sample_id not in cluster_assignments_dict[user_id].keys():
                    cluster_assignments_dict[user_id][sample_id] = {}
                history_ft = user_history_fts[topic_idx]
                assigned_cluster_idx = self.cluster_assignment(user_id, history_ft)
                cluster_assignments_dict[user_id][sample_id][topic_id] = assigned_cluster_idx
        
        cluster_assignments_save_path = os.path.join(self.save_dir, 'cluster_assignments.json')
        with open(cluster_assignments_save_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_assignments_dict, f, indent=4, ensure_ascii=False)



class PrincipleUpdate(LLM_Sequential_Generation):
    def __init__(self, topic_schema_dict, requirement_fts_h5f, cluster_centers_h5f, cluster_samples_dict, principle_history_dict, prompt_template_root, n_clusters, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topic_schema_dict = topic_schema_dict
        self.n_clusters = n_clusters
        with open(os.path.join(prompt_template_root, 'requirement_update', 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.requirement_system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_root, 'requirement_update', 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.requirement_user_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_root, 'preference_update', 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.preference_system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_root, 'preference_update', 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.preference_user_prompt_template = ''.join(lines)

        cluster_assignments_save_path = os.path.join(self.save_dir, 'cluster_assignments.json')
        if os.path.exists(cluster_assignments_save_path):
            print('Load cluster assignments from exist files.')
        else:
            print('Perform cluster assignment now...')
            self.clusters_dict = self.load_history_clusters(cluster_centers_h5f, cluster_samples_dict)
            self.make_cluster_assignment(requirement_fts_h5f, cluster_assignments_save_path)
        with open(cluster_assignments_save_path, 'r', encoding='utf-8') as f:
            self.cluster_assignments_dict = json.load(f)
        
        self.principle_history_dict = principle_history_dict
        self.updated_principle, self.query_cluster_dict, self.sample_id_list = self.preprocess(start_user_id, end_user_id) # self.updated_principle: 用于记录每个user当前的更新情况
        self.query_principle = {} # 用于保存所有query sample对应的principle


    def load_history_clusters(self, cluster_centers_h5f, cluster_samples_dict):
        clusters_dict = {}
        for user_id in cluster_centers_h5f.keys():
            clusters_dict[user_id] = {}
            clusters_dict[user_id]['ft'] = cluster_centers_h5f[user_id][()]
            clusters_dict[user_id]['num'] = np.array([len(cluster_item) for cluster_item in cluster_samples_dict[user_id]])
        return clusters_dict


    def cluster_assignment(self, user_id, query_ft):
        cluster_centers = np.copy(self.clusters_dict[user_id]['ft'])
        distances = np.linalg.norm(cluster_centers - np.expand_dims(query_ft, axis=0), axis=1) # 计算新数据点与所有聚类中心的欧几里得距离
        assigned_cluster_idx = int(np.argmin(distances))
        return assigned_cluster_idx
    

    def center_update(self, user_id, query_ft, assigned_cluster_idx):
        origin_ft = np.copy(self.clusters_dict[user_id]['ft'][assigned_cluster_idx])
        origin_num = self.clusters_dict[user_id]['num'][assigned_cluster_idx]
        update_ft = (origin_ft * origin_num + query_ft * 1.0) / (origin_num + 1)

        self.clusters_dict[user_id]['ft'][assigned_cluster_idx] = update_ft
        self.clusters_dict[user_id]['num'][assigned_cluster_idx] += 1


    def make_cluster_assignment(self, requirement_fts_h5f, cluster_assignments_save_path):
        cluster_assignments_dict = {}
        for user_id in requirement_fts_h5f.keys():
            topic_id_list = [i.decode() for i in requirement_fts_h5f[user_id]['query']['topic_ids'][()]]
            cluster_assignments_dict[user_id] = {}
            user_query_fts = requirement_fts_h5f[user_id]['query']['features'][()]
            for topic_idx, topic_id in enumerate(topic_id_list):
                sample_id = topic_id.split('_topic')[0]
                if sample_id not in cluster_assignments_dict[user_id].keys():
                    cluster_assignments_dict[user_id][sample_id] = {}
                query_ft = user_query_fts[topic_idx]
                assigned_cluster_idx = self.cluster_assignment(user_id, query_ft)
                cluster_assignments_dict[user_id][sample_id][topic_id] = assigned_cluster_idx
                self.center_update(user_id, query_ft, assigned_cluster_idx)
        
        with open(cluster_assignments_save_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_assignments_dict, f, indent=4, ensure_ascii=False)


    def preprocess(self, start_user_id, end_user_id):
        all_user_ids = list(self.cluster_assignments_dict.keys())
        if start_user_id:
            start_user_idx = all_user_ids.index(start_user_id)
        else:
            start_user_idx = 0
        if end_user_id:
            end_user_idx = all_user_ids.index(end_user_id)
        else:
            end_user_idx = len(all_user_ids) - 1
        process_user_ids = all_user_ids[start_user_idx: end_user_idx+1]

        updated_principle = {}
        query_cluster_dict = {}
        sample_id_list = []
        for user_id in process_user_ids:
            # 用每个user history中的principle初始化self.updated_principle
            if user_id not in updated_principle.keys():
                updated_principle[user_id] = deepcopy(self.principle_history_dict[user_id])
            # 列出sample_id_list
            query_cluster_dict[user_id] = {}
            for dialogue_id in self.cluster_assignments_dict[user_id].keys():
                query_cluster_dict[user_id][dialogue_id] = {}
                for topic_id in self.cluster_assignments_dict[user_id][dialogue_id].keys():
                    cluster_idx = self.cluster_assignments_dict[user_id][dialogue_id][topic_id]
                    if cluster_idx not in query_cluster_dict[user_id][dialogue_id].keys():
                        query_cluster_dict[user_id][dialogue_id][cluster_idx] = []
                    query_cluster_dict[user_id][dialogue_id][cluster_idx].append(topic_id)
                for cluster_idx in query_cluster_dict[user_id][dialogue_id].keys():
                    sample_id = "{}_cluster{}".format(dialogue_id, str(cluster_idx))
                    sample_id_list += ['requirement_{}'.format(sample_id), 'preference_{}'.format(sample_id)]
        return updated_principle, query_cluster_dict, sample_id_list


    def get_prompt(self, sample_id):
        generation_type, sample_cluster_id = sample_id.split('_', 1)
        user_id = sample_cluster_id.split('_')[0]
        dialogue_id = sample_cluster_id.split('_cluster')[0]
        cluster_idx = int(sample_cluster_id.split('_cluster')[-1])
        topic_id_list = self.query_cluster_dict[user_id][dialogue_id][cluster_idx] # e.g. ["0000_sample27_topic0", "0000_sample27_topic2"]
        topic_idx_list = [int(topic_id.split('_topic')[-1]) for topic_id in topic_id_list] # e.g. [0, 2]
        new_content_list = [self.topic_schema_dict[user_id][dialogue_id][topic_idx][generation_type] for topic_idx in topic_idx_list]
        new_content_list_str = json.dumps(new_content_list, indent=4, ensure_ascii=False)

        general_requirement_str = json.dumps({"general_requirement": self.updated_principle[user_id][cluster_idx]['requirement']}, ensure_ascii=False)
        principle_str = json.dumps({"principle": self.updated_principle[user_id][cluster_idx]['preference']}, ensure_ascii=False)

        if generation_type == 'requirement':
            system_prompt = self.requirement_system_prompt_template
            user_prompt = self.requirement_user_prompt_template.replace('<general_requirement>', general_requirement_str).replace('<requirements>', new_content_list_str)
        else:
            assert generation_type == 'preference'
            system_prompt = self.preference_system_prompt_template
            user_prompt = self.preference_user_prompt_template.replace('<general_requirement>', general_requirement_str).replace('<principle>', principle_str).replace('<preferences>', new_content_list_str)

        return system_prompt, user_prompt


    def check_generation(self, sample_id, sample_raw_generation):
        generation_type, sample_cluster_id = sample_id.split('_', 1)
        generation = self.generation_postprocess(sample_raw_generation)
        try: # 确保生成内容符合json格式
            generation_dict = json.loads(generation)

            # 1. 确保"adjust"键存在
            if "adjust" not in generation_dict.keys():
                return False, "misssing keys"
            
            # 2. 确保"adjust"键的值合法
            if generation_dict["adjust"] not in ["Yes", "No"]:
                return False, "wrong 'adjust' value"
            
            # 3. 确保"adjust"为"Yes"时对应的键存在且值有效
            if generation_dict["adjust"] == "Yes":
                valid_key = "general_requirement" if generation_type == "requirement" else "principle"
                if valid_key not in generation_dict.keys():
                    return False, "misssing keys"
                if generation_dict[valid_key].strip() == "" or generation_dict[valid_key] == "...":
                    return False, "empty content"

        except json.JSONDecodeError as e:
            return False, "JSON format error"

        return True, None


    def postprocess_for_iterative_generation(self, sample_id, success, generation):
        # 更新self.updated_principle和self.query_principle
        generation_type, sample_cluster_id = sample_id.split('_', 1)
        valid_key = "general_requirement" if generation_type == "requirement" else "principle"
        user_id = sample_cluster_id.split('_')[0]
        dialogue_id = sample_cluster_id.split('_cluster')[0]
        cluster_idx = int(sample_cluster_id.split('_cluster')[-1])

        if success:
            generation = self.generation_postprocess(generation)
            generation_json = json.loads(generation)
            if generation_json["adjust"] == "Yes":
                self.updated_principle[user_id][cluster_idx][generation_type] = deepcopy(generation_json[valid_key]) # 更新self.updated_principle
            else:
                assert generation_json["adjust"] == "No"
        
        if generation_type == "requirement": # 更新self.query_principle
            if user_id not in self.query_principle.keys():
                self.query_principle[user_id] = {}
            self.query_principle[user_id][dialogue_id] = deepcopy([{'requirement': cluster_item['requirement']} for cluster_item in self.updated_principle[user_id]]) # 更新self.query_principle
        else:
            for i, cluster_item in enumerate(self.updated_principle[user_id]):
                self.query_principle[user_id][dialogue_id][i]['preference'] = deepcopy(cluster_item['preference'])


    def extract_and_save(self, save_path):
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.query_principle, f, indent=4, ensure_ascii=False)





if __name__ == '__main__':
    n_clusters = 8
    max_input_samples = 10 # 归纳经验时输入中最多采样的具体内容数量
    device = 'cuda:0' # 运行前根据gpu占用情况修改
    model_name = 'qwen_max'


    dataset_path = '/data5/hzp/MemPAL/data_synthesis_v2/data/input.json'
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset_dict = json.load(f)
    framework_root = '/data5/hzp/MemPAL/perassist_framework'
    data_dir = model_name
    prompt_template_root = os.path.join(framework_root, 'prompt_template', 'memory_rag_v2', 'principle_extraction')
    topic_schema_path = os.path.join(data_dir, 'memory_rag_v2', 'dialogue_analysis', 'requirement_refinement', 'requirement_refinement.json')
    principle_dir = os.path.join(data_dir, 'memory_rag_v2', 'principle')
    with open(topic_schema_path, 'r', encoding='utf-8') as f:
        topic_schema_dict = json.load(f)


    ### ----- Step 1: dialogue requirement抽特征 -----
    batch_size = 128
    embedding_model_path = 'pretrained_models/paraphrase-multilingual-mpnet-base-v2'
    extractor = RequirementFeatureExtraction(dataset_dict, topic_schema_dict, embedding_model_path, batch_size, device)
    extractor.extract_and_save_fts(principle_dir)
    ### -------------------------


    ### ----- Step 2: history集合 principle提炼 -----
    principle_abstract_dir = os.path.join(principle_dir, 'history')
    requirement_fts_h5f = h5py.File(os.path.join(principle_dir, 'requirement_ft.h5'), 'r')
    principle_file_path = 'principle.json'
    principle_clustering = PrincipleAbstract(topic_schema_dict, requirement_fts_h5f, prompt_template_root, n_clusters, max_input_samples, save_dir=principle_abstract_dir, model_name=model_name)
    
    # # check the prompt
    # system_prompt, user_prompt = principle_clustering.get_prompt('requirement_0000_cluster0')
    # # system_prompt, user_prompt = principle_clustering.get_prompt('preference_0000_cluster0')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    principle_clustering.sequential_generate()
    principle_clustering.extract_and_save(principle_file_path)
    principle_clustering.make_cluster_assignment()
    ### -------------------------


    ### ----- Step 3: query集合 principle更新 -----
    principle_update_dir = os.path.join(principle_dir, 'query')
    requirement_fts_h5f = h5py.File(os.path.join(principle_dir, 'requirement_ft.h5'), 'r')
    cluster_centers_h5f = h5py.File(os.path.join(principle_dir, 'history', 'cluster_centers.h5'), 'r')
    cluster_samples_file_path = os.path.join(principle_dir, 'history', 'cluster_samples.json')
    with open(cluster_samples_file_path, 'r', encoding='utf-8') as f:
        cluster_samples_dict = json.load(f)
    principle_clustering_file_path = os.path.join(principle_dir, 'history', 'principle.json')
    with open(principle_clustering_file_path, 'r', encoding='utf-8') as f:
        principle_history_dict = json.load(f)
    principle_update_file_path = 'query_principle.json'
    principle_update = PrincipleUpdate(topic_schema_dict, requirement_fts_h5f, cluster_centers_h5f, cluster_samples_dict, principle_history_dict, prompt_template_root, n_clusters, save_dir=principle_update_dir, model_name=model_name)

    # # check the prompt
    # system_prompt, user_prompt = principle_update.get_prompt("requirement_0000_sample30_cluster1")
    # # system_prompt, user_prompt = principle_update.get_prompt("preference_0000_sample30_cluster1")
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    principle_update.sequential_generate()
    principle_update.extract_and_save(principle_update_file_path)
    ### -------------------------


    # ----- Step 4：后处理：合并history和query集合的principle.json和cluster_assignments.json -----
    principle_abstract_dir = os.path.join(principle_dir, 'history')
    principle_update_dir = os.path.join(principle_dir, 'query')

    # 合并principle.json
    history_principle_path = os.path.join(principle_abstract_dir, 'principle.json')
    with open(history_principle_path, 'r', encoding='utf-8') as f:
        history_principle_dict = json.load(f)
    query_principle_path = os.path.join(principle_update_dir, 'query_principle.json')
    with open(query_principle_path, 'r', encoding='utf-8') as f:
        query_principle_dict = json.load(f)
    principle_save_path = os.path.join(principle_dir, 'principle.json')

    combined_principle_dict = {}
    for user_id in dataset_dict.keys():
        combined_principle_dict[user_id] = {}
        last_history_sample_id = dataset_dict[user_id]['history'][-1]['sample_id']
        combined_principle_dict[user_id][last_history_sample_id] = history_principle_dict[user_id]
        combined_principle_dict[user_id].update(query_principle_dict[user_id])

    with open(principle_save_path, 'w', encoding='utf-8') as f:
        json.dump(combined_principle_dict, f, indent=4, ensure_ascii=False)

    # 合并cluster_assignments.json
    history_assignment_path = os.path.join(principle_abstract_dir, 'cluster_assignments.json')
    with open(history_assignment_path, 'r', encoding='utf-8') as f:
        history_assignment_dict = json.load(f)
    query_assignment_path = os.path.join(principle_update_dir, 'cluster_assignments.json')
    with open(query_assignment_path, 'r', encoding='utf-8') as f:
        query_assignment_dict = json.load(f)
    assignment_save_path = os.path.join(principle_dir, 'cluster_assignments.json')
    
    combined_assignment_dict = {}
    for user_id in history_assignment_dict.keys():
        combined_assignment_dict[user_id] = {}
        combined_assignment_dict[user_id].update(history_assignment_dict[user_id])
        combined_assignment_dict[user_id].update(query_assignment_dict[user_id])
    
    with open(assignment_save_path, 'w', encoding='utf-8') as f:
        json.dump(combined_assignment_dict, f, indent=4, ensure_ascii=False)
    ### -------------------------

