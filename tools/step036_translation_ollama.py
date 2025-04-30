# -*- coding: utf-8 -*-
import json
import os
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def ollama_response(messages, model_name=None):
    """
    使用Ollama API进行翻译处理

    参数:
        messages: 与OpenAI格式兼容的消息列表
        model_name: Ollama模型名称，如果为None则从环境变量获取

    返回:
        翻译结果文本
    """
    model_name = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')

    # 获取Ollama API的URL
    base_url = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434/api')
    url = f"{base_url}/chat"

    # 准备请求数据
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False
    }

    try:
        logger.info(f"正在使用Ollama模型 {model_name} 进行翻译...")
        response = requests.post(url, json=payload, timeout=120)

        if response.status_code == 200:
            result = response.json()
            return result.get('message', {}).get('content', '')
        else:
            logger.error(f"请求Ollama API失败，状态码：{response.status_code}")
            logger.error(f"错误详情：{response.text}")
            raise Exception(f"请求Ollama API失败，状态码：{response.status_code}")
    except Exception as e:
        logger.error(f"与Ollama通信过程中发生错误: {str(e)}")
        raise


def ollama_stream_response(messages, model_name=None):
    """
    使用Ollama API进行流式翻译处理（适用于较长文本）

    参数:
        messages: 与OpenAI格式兼容的消息列表
        model_name: Ollama模型名称，如果为None则从环境变量获取

    返回:
        完整的翻译结果文本
    """
    if model_name is None:
        model_name = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')

    # 获取Ollama API的URL
    base_url = os.getenv('OLLAMA_API_BASE', 'http://localhost:11434/api')
    url = f"{base_url}/chat"

    # 准备请求数据
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True
    }

    try:
        logger.info(f"正在使用Ollama模型 {model_name} 进行流式翻译...")
        response = requests.post(url, json=payload, timeout=300, stream=True)

        if response.status_code == 200:
            # 收集流式响应中的所有结果
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line.decode('utf-8'))
                    if 'message' in line_data and 'content' in line_data['message']:
                        content = line_data['message']['content']
                        full_response += content

            return full_response
        else:
            logger.error(f"请求Ollama流式API失败，状态码：{response.status_code}")
            logger.error(f"错误详情：{response.text}")
            raise Exception(f"请求Ollama流式API失败，状态码：{response.status_code}")
    except Exception as e:
        logger.error(f"与Ollama流式通信过程中发生错误: {str(e)}")
        raise


if __name__ == '__main__':
    # 测试基本翻译功能
    test_message = [{"role": "user", "content": "你好，介绍一下你自己"}]
    response = ollama_response(test_message)
    print(f"基本响应:\n{response}\n")

    # 测试翻译功能
    translate_message = [
        {"role": "system", "content": "你是一个专业的翻译员，你需要将英语文本翻译成流畅自然的中文。"},
        {"role": "user",
         "content": "Translate this sentence to Chinese: 'The quick brown fox jumps over the lazy dog.'"}
    ]
    translation = ollama_response(translate_message)
    print(f"翻译结果:\n{translation}")

