import os
import json
import requests
import csv
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from typing import List, Optional
import time
from copy import copy, deepcopy

class LLM_Proxy():

    def __init__(self):
        self.config = {"max_length": 10000}

    def llm_request(self, system_prompt, user_prompt, model_name='qwen2.5-max'):
        if model_name == "gpt4":
            return self.llm_request_gpt(system_prompt, user_prompt, model='gpt-4-0409')
        elif model_name == "gpt4o":
            return self.llm_request_gpt(system_prompt, user_prompt, model='gpt-4o-0513')
        elif model_name == "qwen_max":
            return self.llm_request_qwen(system_prompt, user_prompt, model='qwen_max')
        elif model_name == "qwen2.5-max":
            return self.llm_request_qwen(system_prompt, user_prompt, model='qwen2.5-max')
        raise RuntimeError("Invalid LLM.")

    def llm_request_gpt(self, system_prompt, user_prompt, model):
        api_key = "sk-..." # replace to your key
        client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1/")
        error_seg_list = [
            "Too many requests",
            "Service Unavailable",
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        success = True
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=8192
            )
            output = response.choices[0].message.content
        except Exception as ex:
            success = False
            output = ex
        if not isinstance(output, str):
            success = False
        else:
            for error_seg in error_seg_list:
                if error_seg in output:
                    success = False
        return success, output


    def llm_request_qwen(self, system_prompt, user_prompt, model):
        data = {
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "platformInput": {'model': model},
            "temperature": 0.7,
            "max_tokens": 8192
        }
        data = json.dumps(data).encode("UTF-8")
        headers = {}
        url = "..."
        response = ""
        
        try:
            response = requests.post(url=url, data=data, headers=headers)
            response = json.loads(response.content.decode("UTF-8"))
            if response['success']:
                response = response['data']['choices'][0]['message']['content']
                return True, response
            else:
                return False, response

        except Exception as ex:
            return False, ex


class LLM_Individual_Generation(LLM_Proxy):
    """
    带有错误处理的llm单次生成脚本，如果调用出错或生成内容不合法则重复生成直至成功。应用时需继承此类，重写get_prompt()函数和check_generation()函数。
    """
    def __init__(
        self,
        save_dir: str,
        err_file_name: str = 'err.csv', # 记录在当前轮次未能成功生成的sample和错误信息
        err_log_file_name: str = 'err.log', # 记录llm生成过程中的错误信息
        err_generation_file_name: str = 'err_generation.csv', # 记录check后出现错误的生成结果（便于分析生成效果）
        prompt_log_name: str = 'prompt.log', # 记录prompt（用来检查prompt是否正确）
        model_name: str = 'qwen_max', # llm名称
    ):
        super().__init__()
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.err_path = os.path.join(save_dir, err_file_name)
        self.err_log_path = os.path.join(save_dir, err_log_file_name)
        self.err_generation_path = os.path.join(save_dir, err_generation_file_name)
        self.prompt_log_path = os.path.join(save_dir, prompt_log_name)
        self.model_name = model_name
        self.tolerable_error_type_list = []


    def get_prompt(self, sample_id): # 需在子类中重写，实现根据指定的data_id生成对应的prompt
        system_prompt = 'You are a helpful assistant.'
        user_prompt = 'hello'
        return system_prompt, user_prompt


    def check_generation(self, sample_id, raw_generation): # 需在子类中重写，判断生成内容是否合法
        error_info = None
        return True, error_info


    def generation_postprocess(self, raw_generation_str):
        if '```json' in raw_generation_str:
            raw_generation_str = raw_generation_str.split('```json')[-1]
            if '```' in raw_generation_str:
                raw_generation_str = raw_generation_str.split('```')[0]
        raw_generation_str = raw_generation_str.replace('```', '').strip()
        generation_str = raw_generation_str.replace('\"\"\"', '').strip()
        return generation_str


    def record_prompt_log(self, sample_id, system_prompt, user_prompt, prompt_log_path):
        dump_str = '---------- {} ----------\n【system prompt】\n{}\n【user prompt】\n{}\n\n'.format(sample_id, system_prompt, user_prompt)
        with open(prompt_log_path, 'a') as prompt_log_file:
            prompt_log_file.write(dump_str)


    def generate(self, sample_id, sleep_time=None, retry_num=10, record_prompt=False):
        '''
        return：success, output
        '''
        success = False
        system_prompt, user_prompt = self.get_prompt(sample_id)
        if record_prompt:
            self.record_prompt_log(sample_id, system_prompt, user_prompt, self.prompt_log_path)

        cnt_err_num = 0
        api_error_type_list = ["限流", "Connection aborted.", "HTTPSConnectionPool"]
        while success == False: # 如果不成功就一直循环当前sample
            success, output = self.llm_request(system_prompt, user_prompt, self.model_name)
            if success: # 成功调用
                success, error_info = self.check_generation(sample_id, output)
                if success: # 成功调用且成功生成
                    return success, self.generation_postprocess(output)
                else:
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                    with open(self.err_log_path, 'a') as err_file:
                        err_file.write("{} | {} | {}\n".format(time_str, sample_id, error_info))
                    with open(self.err_generation_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([sample_id, output])

                    # 某些sample可能永远无法正常输出，为了防止生成过程卡住，需要跳过该sample，并将该sample记入err.csv中，留待后续单独处理。
                    cnt_error = True
                    for api_err_type_substring in api_error_type_list:
                        if api_err_type_substring in str(output):
                            cnt_error = False
                            break
                    if cnt_error:
                        cnt_err_num += 1
                        if cnt_err_num >= retry_num:
                            if error_info in self.tolerable_error_type_list:
                                with open(self.err_log_path, 'a') as err_file:
                                    err_file.write("{} | Ignore the error of {} | error type: {}\n".format(time_str, sample_id, error_info))
                                return True, self.generation_postprocess(output)
                            else:
                                with open(self.err_path, 'a') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([sample_id, output])
                                return False, None
            else:
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                with open(self.err_log_path, 'a') as err_file:
                    err_file.write("{} | {} | {}\n".format(time_str, sample_id, output))
                
                # 某些sample可能永远无法正常输出，为了防止生成过程卡住，需要跳过该sample，并将该sample记入err.csv中，留待后续单独处理。
                cnt_error = True
                for api_err_type_substring in api_error_type_list:
                    if api_err_type_substring in str(output):
                        cnt_error = False
                        break
                if cnt_error:
                    cnt_err_num += 1
                    if cnt_err_num >= retry_num:
                        with open(self.err_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([sample_id, output])
                        return False, None

            if sleep_time:
                time.sleep(sleep_time)



class LLM_Sequential_Generation(LLM_Proxy):
    """
    带有错误处理的llm生成脚本，顺序生成各个数据，如果调用出错或生成内容不合法则就地重复生成直至成功。应用时需继承此类，重写get_prompt()函数和check_generation()函数。
    """

    def __init__(
        self,
        save_dir: str,
        raw_file_name: str = 'raw.csv', # 存放llm初步生成结果
        err_file_name: str = 'err.csv', # 记录在当前轮次未能成功生成的sample和错误信息
        err_log_file_name: str = 'err.log', # 记录llm生成过程中的错误信息
        err_generation_file_name: str = 'err_generation.csv', # 记录check后出现错误的生成结果（便于分析生成效果）
        model_name: str = 'qwen_max', # llm名称
        sample_id_list: Optional[List[str]] = None # 生成sample的id列表
    ):
        super().__init__()
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.raw_path = os.path.join(save_dir, raw_file_name)
        self.err_path = os.path.join(save_dir, err_file_name)
        self.err_log_path = os.path.join(save_dir, err_log_file_name)
        self.err_generation_path = os.path.join(save_dir, err_generation_file_name)
        self.model_name = model_name
        self.sample_id_list = sample_id_list
        self.tolerable_error_type_list = [] # 用于`sequential_generate()`对generation进行check的过程中。如果某种生成错误类型难以避免，但又被认为是对输出没有过大的影响，可以在重复次数达到retry_num后直接忽略该种错误类型，将输出视为成功的输出

    def get_prompt(self, sample_id): # 需在子类中重写，实现根据指定的data_id生成对应的prompt
        system_prompt = 'You are a helpful assistant.'
        user_prompt = 'hello'
        return system_prompt, user_prompt

    def check_generation(self, sample_id, raw_generation): # 需在子类中重写，判断生成内容是否合法
        error_info = None
        return True, error_info

    def postprocess_for_iterative_generation(self, sample_id, success, generation):
        '''
        在迭代生成的情况下（后面sample生成的输入依赖于前面sample的输出）：
        可以重写该函数，在该函数中将当前sample的输出暂存，用于后续sample的输入
        '''
        return

    def sequential_generate(self, sleep_time=None, retry_num=10, continue_generate=False):
        '''
        以csv格式存储llm生成结果，保存到raw.csv中。
        列名：sample_id, generation
        如果调用失败，或生成内容不符合规范，则重新生成该条内容，直到生成成功。
        执行失败的信息记入err.log中。
        不符合规范的生成内容记入err.csv中，便于分析和检查生成效果。
        '''
        if not continue_generate:
            with open(self.raw_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'generation'])
            with open(self.err_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'generation'])
            with open(self.err_generation_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_id', 'generation'])
            err_log_file = open(self.err_log_path, 'w')
            err_log_file.close()

        for sample_id in tqdm(self.sample_id_list):
            success = False
            system_prompt, user_prompt = self.get_prompt(sample_id)

            # print("---sample_id---\t{}".format(sample_id))###
            # print(user_prompt)###
            # print('################################')###
            # print('\n\n\n')###

            cnt_err_num = 0
            api_error_type_list = ["限流", "Connection aborted.", "HTTPSConnectionPool"]
            while success == False: # 如果不成功就一直循环当前sample
                success, output = self.llm_request(system_prompt, user_prompt, self.model_name)
                if success: # 成功调用
                    success, error_info = self.check_generation(sample_id, output)
                    if success: # 成功调用且成功生成
                        with open(self.raw_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([sample_id, output])
                        self.postprocess_for_iterative_generation(sample_id, True, output)
                    else:
                        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                        with open(self.err_log_path, 'a') as err_file:
                            err_file.write("{} | {} | {}\n".format(time_str, sample_id, error_info))
                        with open(self.err_generation_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([sample_id, output])

                        # 某些sample可能永远无法正常输出，为了防止生成过程卡住，需要跳过该sample，并将该sample记入err.csv中，留待后续单独处理。
                        cnt_error = True
                        for api_err_type_substring in api_error_type_list:
                            if api_err_type_substring in str(output):
                                cnt_error = False
                                break
                        if cnt_error:
                            cnt_err_num += 1
                            if cnt_err_num >= retry_num:
                                if error_info in self.tolerable_error_type_list:
                                    with open(self.raw_path, 'a') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([sample_id, output])
                                    with open(self.err_log_path, 'a') as err_file:
                                        err_file.write("{} | Ignore the error of {} | error type: {}\n".format(time_str, sample_id, error_info))
                                    self.postprocess_for_iterative_generation(sample_id, True, output)
                                else:
                                    with open(self.err_path, 'a') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([sample_id, output])
                                    with open(self.raw_path, 'a') as f:
                                        writer = csv.writer(f)
                                        writer.writerow([sample_id, None])
                                    self.postprocess_for_iterative_generation(sample_id, False, None)
                                break
                else:
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                    with open(self.err_log_path, 'a') as err_file:
                        err_file.write("{} | {} | {}\n".format(time_str, sample_id, output))
                    
                    # 某些sample可能永远无法正常输出，为了防止生成过程卡住，需要跳过该sample，并将该sample记入err.csv中，留待后续单独处理。
                    cnt_error = True
                    for api_err_type_substring in api_error_type_list:
                        if api_err_type_substring in str(output):
                            cnt_error = False
                            break
                    if cnt_error:
                        cnt_err_num += 1
                        if cnt_err_num >= retry_num:
                            with open(self.err_path, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([sample_id, output])
                            with open(self.raw_path, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([sample_id, None])
                            self.postprocess_for_iterative_generation(sample_id, False, None)
                            break


                if sleep_time:
                    time.sleep(sleep_time)


    def generation_postprocess(self, raw_generation_str):
        if '```json' in raw_generation_str:
            raw_generation_str = raw_generation_str.split('```json')[-1]
            if '```' in raw_generation_str:
                raw_generation_str = raw_generation_str.split('```')[0]
        raw_generation_str = raw_generation_str.replace('```', '').strip()
        generation_str = raw_generation_str.replace('\"\"\"', '').strip()
        return generation_str
    
    

class User_Assistant_Interaction_Generation(LLM_Proxy):
    """
    带有错误处理的user-assistant多轮对话生成脚本，两个模型依次生成语句内容，如果调用出错或生成内容不合法则就地重复生成直至成功。
    应用时需继承此类，重写：get_user_prompt(), get_assistant_prompt(), check_user_generation(), check_assistant_generation(), update_interaction_context()函数。
    """
    def __init__(
        self,
        save_dir: str,
        raw_file_name: str = 'raw.csv', # 存放llm初步生成结果
        err_file_name: str = 'err.csv', # 记录在当前轮次未能成功生成的sample和错误信息
        err_log_file_name: str = 'err.log', # 记录llm生成过程中的错误信息
        err_generation_file_name: str = 'err_generation.csv', # 记录check后出现错误的生成结果（便于分析生成效果）
        user_prompt_log_name: str = 'user_llm_prompt.log', # 记录user-llm的prompt（用来检查prompt是否正确）
        assistant_prompt_log_name: str = 'assistant_llm_prompt.log', # 记录assistant-llm的prompt（用来检查prompt是否正确）
        user_model_name: str = 'qwen2.5-max', # llm名称
        assistant_model_name: str = 'qwen_max', # llm名称
        sample_id_list: Optional[List[str]] = None # 生成sample的id列表
    ):
        super().__init__()
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.raw_path = os.path.join(save_dir, raw_file_name)
        self.err_path = os.path.join(save_dir, err_file_name)
        self.err_log_path = os.path.join(save_dir, err_log_file_name)
        self.err_generation_path = os.path.join(save_dir, err_generation_file_name)
        self.user_prompt_log_path = os.path.join(save_dir, user_prompt_log_name)
        self.assistant_prompt_log_path = os.path.join(save_dir, assistant_prompt_log_name)
        self.user_model_name = user_model_name
        self.assistant_model_name = assistant_model_name
        self.sample_id_list = sample_id_list
        self.interaction_context_dict = {} # 每次成功生成后需要及时更新
        self.tolerable_error_type_list = [] # 用于`sequential_generate()`对generation进行check的过程中。如果某种生成错误类型难以避免，但又被认为是对输出没有过大的影响，可以在重复次数达到retry_num后直接忽略该种错误类型，将输出视为成功的输出


    def get_user_prompt(self, sample_id): # 需在子类中重写，实现根据指定的data_id生成对应的prompt
        system_prompt = 'You are a user.'
        user_prompt = 'hello'
        return system_prompt, user_prompt


    def get_assistant_prompt(self, sample_id): # 需在子类中重写，实现根据指定的data_id生成对应的prompt
        system_prompt = 'You are a helpful assistant.'
        user_prompt = 'hello'
        return system_prompt, user_prompt


    def check_user_generation(self, sample_id, raw_generation): # 需在子类中重写，判断生成内容是否合法
        error_info = None
        return True, error_info
    

    def check_assistant_generation(self, sample_id, raw_generation): # 需在子类中重写，判断生成内容是否合法
        error_info = None
        return True, error_info


    def update_interaction_context(self, role, sample_id, success, generation):
        '''
        需要重写该函数，将当前生成的语句更新到`self.interaction_context_dict`中
        '''
        return


    def record_prompt_log(self, sample_id, system_prompt, user_prompt, prompt_log_path):
        dump_str = '---------- {} ----------\n【system prompt】\n{}\n【user prompt】\n{}\n\n'.format(sample_id, system_prompt, user_prompt)
        with open(prompt_log_path, 'a') as prompt_log_file:
            prompt_log_file.write(dump_str)


    def create_log_files(self, record_prompt):
        '''
        如果需要保存其它中间数据或日志，则需重写该函数
        '''
        with open(self.raw_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'generation'])
        with open(self.err_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'generation'])
        with open(self.err_generation_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'generation'])
        err_log_file = open(self.err_log_path, 'w')
        err_log_file.close()
        if record_prompt:
            user_prompt_log_file = open(self.user_prompt_log_path, 'w')
            user_prompt_log_file.close()
            assistant_prompt_log_file = open(self.assistant_prompt_log_path, 'w')
            assistant_prompt_log_file.close()


    def assistant_process_before_generation(self, sample_id, record_prompt):
        '''
        如果assistant需要在生成前做一些分析步骤或做检索，则需重写该函数
        '''
        return


    def interaction_generate(self, sleep_time=None, retry_num=10, continue_generate=False, record_prompt=False):
        '''
        以csv格式存储llm生成结果，保存到raw.csv中。
        列名：sample_id, generation
        如果调用失败，或生成内容不符合规范，则重新生成该条内容，直到生成成功。
        执行失败的信息记入err.log中。
        不符合规范的生成内容记入err.csv中，便于分析和检查生成效果。
        '''
        if not continue_generate:
            self.create_log_files(record_prompt)

        for sample_id in tqdm(self.sample_id_list):
            for role in ['user', 'assistant']:
                role_sample_id = '{}_{}'.format(sample_id, role)
                success = False
                if role == 'user':
                    system_prompt, user_prompt = self.get_user_prompt(sample_id)
                    model_name = copy(self.user_model_name)
                    if record_prompt:
                        self.record_prompt_log(sample_id, system_prompt, user_prompt, self.user_prompt_log_path)
                else:
                    self.assistant_process_before_generation(sample_id, record_prompt) # assistant生成回复前要执行的步骤
                    system_prompt, user_prompt = self.get_assistant_prompt(sample_id)
                    model_name = copy(self.assistant_model_name)
                    if record_prompt:
                        self.record_prompt_log(sample_id, system_prompt, user_prompt, self.assistant_prompt_log_path)

                cnt_err_num = 0
                api_error_type_list = ["限流", "Connection aborted.", "HTTPSConnectionPool"]
                while success == False: # 如果不成功就一直循环当前sample
                    success, output = self.llm_request(system_prompt, user_prompt, model_name)
                    if success: # 成功调用
                        if role == 'user':
                            success, error_info = self.check_user_generation(sample_id, output)
                        else:
                            success, error_info = self.check_assistant_generation(sample_id, output)
                        if success: # 成功调用且成功生成
                            with open(self.raw_path, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([role_sample_id, output])
                            self.update_interaction_context(role, sample_id, True, output)
                        else:
                            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                            with open(self.err_log_path, 'a') as err_file:
                                err_file.write("{} | {} | {}\n".format(time_str, role_sample_id, error_info))
                            with open(self.err_generation_path, 'a') as f:
                                writer = csv.writer(f)
                                writer.writerow([role_sample_id, output])

                            # 某些sample可能永远无法正常输出，为了防止生成过程卡住，需要跳过该sample，并将该sample记入err.csv中，留待后续单独处理。
                            cnt_error = True
                            for api_err_type_substring in api_error_type_list:
                                if api_err_type_substring in str(output):
                                    cnt_error = False
                                    break
                            if cnt_error:
                                cnt_err_num += 1
                                if cnt_err_num >= retry_num:
                                    if error_info in self.tolerable_error_type_list:
                                        with open(self.raw_path, 'a') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([role_sample_id, output])
                                        with open(self.err_log_path, 'a') as err_file:
                                            err_file.write("{} | Ignore the error of {} | error type: {}\n".format(time_str, role_sample_id, error_info))
                                        self.update_interaction_context(role, sample_id, True, output)
                                    else:
                                        with open(self.err_path, 'a') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([role_sample_id, output])
                                        with open(self.raw_path, 'a') as f:
                                            writer = csv.writer(f)
                                            writer.writerow([role_sample_id, None])
                                        self.update_interaction_context(role, sample_id, False, None)
                                    break
                    
                    else:
                        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
                        with open(self.err_log_path, 'a') as err_file:
                            err_file.write("{} | {} | {}\n".format(time_str, role_sample_id, output))
                        
                        # 某些sample可能永远无法正常输出，为了防止生成过程卡住，需要跳过该sample，并将该sample记入err.csv中，留待后续单独处理。
                        cnt_error = True
                        for api_err_type_substring in api_error_type_list:
                            if api_err_type_substring in str(output):
                                cnt_error = False
                                break
                        if cnt_error:
                            cnt_err_num += 1
                            if cnt_err_num >= retry_num:
                                with open(self.err_path, 'a') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([role_sample_id, output])
                                with open(self.raw_path, 'a') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([role_sample_id, None])
                                self.update_interaction_context(role, sample_id, False, None)
                                break
                    
                    if sleep_time:
                        time.sleep(sleep_time)


    def generation_postprocess(self, raw_generation_str):
        if '```json' in raw_generation_str:
            raw_generation_str = raw_generation_str.split('```json')[-1]
            if '```' in raw_generation_str:
                raw_generation_str = raw_generation_str.split('```')[0]
        raw_generation_str = raw_generation_str.replace('```', '').strip()
        generation_str = raw_generation_str.replace('\"\"\"', '').strip()
        return generation_str