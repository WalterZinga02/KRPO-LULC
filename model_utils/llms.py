import json
import requests
from openai import OpenAI
import os
import time
import logging


class UTF8FileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)


def text_relcanon(query: str, docs: list[str]):
    data = {
        "model": "",
        "query": query,
        "return_documents": False,
        "top_n": 10,
        "documents": docs
    }
    json_data = json.dumps(data)
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post('http://127.0.0.1:your_port/', headers=headers, data=json_data)
        if response.status_code == 200:
            response_json = response.json()
            return [(rerank_res['index'], rerank_res['relevance_score']) for rerank_res in response_json['results']]
        else:
            print("text_relcanon error:", response.status_code, "retrying")
            return text_relcanon(query, docs)
    except:
        time.sleep(3)
        print("retry")
        return text_relcanon(query, docs)


class LLMInvoker:
    def __init__(self, choose_model="", llm_log_path=""):
        self.llm = OpenAI(
           api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.choose_model = choose_model

        self.llm_logger = logging.getLogger('Logger1')
        self.llm_log_handler = UTF8FileHandler(llm_log_path, encoding='utf-8')
        self.llm_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.llm_logger.addHandler(self.llm_log_handler)
        self.llm_logger.setLevel(logging.INFO)

    def _invoke(self, mess):
        self.llm_logger.info(f"callModel: {self.choose_model}")
        completion = self.llm.chat.completions.create(
            model=self.choose_model,
            messages=mess,
            temperature=0.2
        )
        self.llm_logger.info(f"callConsumption: {completion.usage}")
        return completion.choices[0].message.content

    def get_ans_with_retry(self, message, try_num=5):

        for attempt in range(try_num):
            try:
                res_ans = self._invoke(message)
                if isinstance(res_ans, str) and res_ans.strip():
                    return res_ans
            except Exception as e:
                self.llm_logger.warning(f"{attempt + 1}secondAttemptFailed: {e}")
                time.sleep(2 ** attempt)
        return None

    def llm_chat_response(self, prompt, sys_prompt=None, try_num=5):
        messages = [{'role': 'user', 'content': prompt}]
        if sys_prompt:
            messages.insert(0, {'role': 'system', 'content': sys_prompt})
        result_ans = self.get_ans_with_retry(messages, try_num)
        return result_ans
