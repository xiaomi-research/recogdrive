import os
import re
import time
import json
import requests


class GPTEvaluation:
    def __init__(self, 
                 api_key, 
                 temperature=0.0,
                 max_tokens=3000,
                 log_file=None):
        """GPT evaluation API.
        """
        
        self.api_key = api_key
        self.model = "gpt-3.5-turbo"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.content = []
        self.log_file = log_file
        self.log_data = {}
        self.judge_settings = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens}

        # Resume evaluation if log file exists
        log_dir = os.path.dirname(self.log_file)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        elif os.path.exists(self.log_file):
            self.resume_evaluation()

    def resume_evaluation(self):
        """Resumes evaluation from the last checkpoint in the log file.
        """
        with open(self.log_file, 'r', encoding='utf-8') as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict) or not isinstance(entry.get("gpt_score"), str):
                    continue
                if entry.get("judge_settings", self.judge_settings) != self.judge_settings:
                    continue
                key = tuple(entry.get(field) for field in ("scene_token", "frame_token", "question", "answer", "pred"))
                self.log_data[key] = entry["gpt_score"]

    def addTextPrompt(self, textPrompt: str):
        self.content.append({
            "type": "text",
            "text": textPrompt
        })

    def request_chatgpt(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": "\n".join([msg["text"] for msg in self.content])
                }
            ],
            "max_tokens": self.max_tokens,
        }
        for attempt in range(3):
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions",
                                         headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                break
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                status = exc.response.status_code if exc.response is not None else None
                if attempt == 2 or (status is not None and status != 429 and status < 500):
                    raise
                time.sleep(5)
        response_json = response.json()
        reply = response_json['choices'][0]['message']['content']
        total_tokens = response_json['usage']['total_tokens']
        return reply, total_tokens

    def prepare_chatgpt_message(self, prompt):
        system_message = "You are an evaluator who rates answers based on their closeness to the correct answer."
        self.addTextPrompt(system_message)
        self.addTextPrompt(prompt)

    def call_chatgpt(self, prompt):
        self.content = []  # Reset for each call.
        self.prepare_chatgpt_message(prompt)
        reply, total_tokens = self.request_chatgpt()
        return reply, total_tokens

    def extract_answer(self, reply):
        """Extracts an integer score from a line like "Total Score: <score>".
        """
        pattern = r"Total Score:\s*(\d{1,3})\b"
        match = re.search(pattern, reply)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"No valid score found in the reply:\n{reply}")

    def log(self, input_tuple):
        """
        Logs the provided prompt, GPT output, final score, and input data.
        The log entry automatically includes extra key-value pairs from extra_log_info.
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(input_tuple, separators=(',', ':')) + '\n')

    def forward(self, data):
        """
        Expects data as a tuple: ((question, predicted, GT, desc), prompt_template).
        It formats the prompt accordingly, calls GPT, extracts the score,
        logs the result (including an extra 'input_data' field), and returns the score.
        """
        (input_tuple, prompt_template) = data
        scene_token = input_tuple['scene_token']
        frame_token = input_tuple['frame_token']
        question = input_tuple['question']
        answer = input_tuple['answer']
        pred = input_tuple['pred']
        desc = input_tuple['desc']

        key = (scene_token, frame_token, question, answer, pred)
        if key in self.log_data:
            return self.extract_answer(self.log_data[key])

        prompt = prompt_template.format(GT=answer, PRED=pred, QUESTION=question, DESC=desc or "")
        reply, _ = self.call_chatgpt(prompt)
        final_score = self.extract_answer(reply)
        self.log({**input_tuple, 'gpt_score': reply, 'judge_settings': self.judge_settings})
        return final_score
