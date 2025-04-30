# -*- coding: utf-8 -*-
import os
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

extra_body = {
    'repetition_penalty': 1.1,
}
model_name = os.getenv('QWEN_MODEL_ID', 'qwen-max-2025-01-25')
def qwen_response(messages):
    client = OpenAI(
        # This is the default and can be omitted
        base_url=os.getenv('QWEN_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
        api_key=os.getenv('QWEN_API_KEY', 'sk-19833f4d600340668696111248147e45')
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        timeout=240,
        extra_body=extra_body
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    test_message = [{"role": "user", "content": "你好，介绍一下你自己"}]
    response = qwen_response(test_message)
    print(response)