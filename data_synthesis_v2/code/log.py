import os
import json
from llm_generation import LLM_Sequential_Generation
from datetime import datetime
import csv
import ast
from copy import deepcopy

class LogGeneration(LLM_Sequential_Generation):
    def __init__(self, background_dict, experience_dict, prompt_template_dir, save_dir, start_user_id=None, end_user_id=None, *args, **kwargs):
        super().__init__(save_dir=save_dir, *args, **kwargs)
        for k in background_dict.keys():
            background = background_dict[k]
            del background['personality']
        self.background_dict = background_dict
        self.experience_dict = experience_dict
        self.sample_id_list = self.get_sample_id_list(experience_dict, start_user_id, end_user_id)
        with open(os.path.join(prompt_template_dir, 'system_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.system_prompt_template = ''.join(lines)
        with open(os.path.join(prompt_template_dir, 'user_prompt.txt'), 'r', encoding="utf-8") as f:
            lines = f.readlines()
            self.user_prompt_template = ''.join(lines)
        self.tolerable_error_type_list = ["wrong timestamp (cur_timestamp < last_time)", "wrong timestamp (cur_timestamp > end_timestamp)", "wrong timestamp (cur_timestamp <= last_time)"]


    def get_sample_id_list(self, experience_dict, start_user_id, end_user_id):
        all_user_ids = list(experience_dict.keys())
        if start_user_id:
            start_user_index = all_user_ids.index(start_user_id)
        else:
            start_user_index = 0
        if end_user_id:
            end_user_index = all_user_ids.index(end_user_id)
        else:
            end_user_index = len(all_user_ids) - 1
        processed_user_ids = all_user_ids[start_user_index:end_user_index+1]
        sample_id_list = []
        for user_id in processed_user_ids:
            for date_str in experience_dict[user_id].keys():
                sample_id_list.append("{}_{}".format(user_id, date_str))
        return sample_id_list


    def get_prompt(self, sample_id):
        user_id, date_str = sample_id.split('_')

        background_dict = self.background_dict[user_id]
        background_str = json.dumps(background_dict, ensure_ascii=False)
        experience_str = self.experience_dict[user_id][date_str]['experience']
        start_timestamp = datetime.strptime(self.experience_dict[user_id][date_str]['start_timestamp'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
        end_timestamp = datetime.strptime(self.experience_dict[user_id][date_str]['end_timestamp'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
        time_span = "{} ~ {}".format(start_timestamp, end_timestamp)

        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.replace('<background>', background_str).replace('<experience>', experience_str).replace('<time_span>', time_span)
        return system_prompt, user_prompt
    

    def check_generation(self, sample_id, raw_generation):
        raw_generation = self.generation_postprocess(raw_generation)
        user_id, date_str = sample_id.split('_')
        start_timestamp = datetime.strptime(self.experience_dict[user_id][date_str]['start_timestamp'], "%Y-%m-%d %H:%M:%S").replace(second=0, microsecond=0)
        end_timestamp = datetime.strptime(self.experience_dict[user_id][date_str]['end_timestamp'], "%Y-%m-%d %H:%M:%S").replace(second=0, microsecond=0)
        last_time = deepcopy(start_timestamp)
        try:
            generation_list = ast.literal_eval(raw_generation)
            if len(generation_list) < 15:
                return False, "less than 15 logs"
            for i, generation_item in enumerate(generation_list):
                if list(generation_item.keys()) != ['timestamp', 'event', 'type', 'content']:
                    return False, "key error"

                try:
                    cur_timestamp = datetime.strptime(generation_item["timestamp"], "%Y-%m-%d %H:%M")
                except:
                    False, "wrong timestamp format"
                if i == 0:
                    if cur_timestamp < last_time:
                        return False, "wrong timestamp (cur_timestamp < last_time)"
                    if cur_timestamp > end_timestamp:
                        return False, "wrong timestamp (cur_timestamp > end_timestamp)"
                else:
                    if cur_timestamp <= last_time:
                        return False, "wrong timestamp (cur_timestamp <= last_time)"
                    if cur_timestamp > end_timestamp:
                        return False, "wrong timestamp (cur_timestamp > end_timestamp)"
                last_time = deepcopy(cur_timestamp)

                if generation_item["type"] not in ["网页搜索", "内容发布", "内容浏览", "消息发送", "消息接收", "日程管理", "交易记录", "设备操作"]:
                    return False, "wrong log type"
                
                if generation_item["content"] == "..." or generation_item["content"].strip() == "":
                    return False, "empty log"

        except:
            return False, "list format error"

        return True, None
    

    def log_postprocess(self, sample_id, logs):
        '''
        由于timestamp的时序错误难以完全避免，需要对生成结果做后处理，去掉时间范围外的logs，同时把所有logs按时间顺序排列
        '''
        user_id, date_str = sample_id.split('_')
        start_timestamp = datetime.strptime(self.experience_dict[user_id][date_str]['start_timestamp'], "%Y-%m-%d %H:%M:%S").replace(second=0, microsecond=0)
        end_timestamp = datetime.strptime(self.experience_dict[user_id][date_str]['end_timestamp'], "%Y-%m-%d %H:%M:%S").replace(second=0, microsecond=0)
        processed_logs = []
        for log in logs:
            cur_timestamp = datetime.strptime(log['timestamp'], "%Y-%m-%d %H:%M")
            if cur_timestamp >= start_timestamp and cur_timestamp <= end_timestamp:
                processed_logs.append(log)
        processed_logs = sorted(processed_logs, key=lambda x: datetime.strptime(x["timestamp"], "%Y-%m-%d %H:%M"))
        return processed_logs
            

    def extract_and_save(self, save_path):
        result_dict = {}
        with open(self.raw_path) as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                sample_id = row['sample_id']
                generation = row['generation']
                generation = self.generation_postprocess(generation)
                logs = ast.literal_eval(generation)
                user_id, date_str = sample_id.split('_')
                if user_id not in result_dict.keys():
                    result_dict[user_id] = {}
                result_dict[user_id][date_str] = self.log_postprocess(sample_id, logs)
        save_path = os.path.join(self.save_dir, save_path)
        with open(save_path, 'w') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    model_name = 'qwen2.5-max'

    data_synthesis_root = '/data5/hzp/MemPAL/data_synthesis_v2/data'
    prompt_template_root = '/data5/hzp/MemPAL/data_synthesis_v2/prompt_template'
    background_data_dir = os.path.join(data_synthesis_root, 'background')
    background_file_path = os.path.join(background_data_dir, 'background.json')
    with open(background_file_path,'r', encoding='utf-8') as f:
        background_dict = json.load(f)
    experience_path = os.path.join(data_synthesis_root, 'experience', 'experience.json')
    with open(experience_path, 'r') as f:
        experience_dict = json.load(f)
    prompt_template_dir = os.path.join(prompt_template_root, 'log')
    save_dir = os.path.join(data_synthesis_root, 'log')

    logGeneration = LogGeneration(background_dict, experience_dict, prompt_template_dir, save_dir=save_dir)

    # system_prompt, user_prompt = logGeneration.get_prompt('0000_2024-03-24')
    # print('【system prompt】:\n{}'.format(system_prompt))
    # print('【user prompt】:\n{}'.format(user_prompt))

    logGeneration.sequential_generate()
    logGeneration.extract_and_save('log.json')