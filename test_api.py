import requests
import json
import time
import os

# API 服务器地址
API_URL = "http://127.0.0.1:6006"


def test_do_everything():
    """测试一键处理功能"""

    # 请求参数
    payload = {
        "root_folder": "videos/test_api",
        "video_url": "/home/bmh/Linly-Dubbing/test/92_1745743557.mp4",
        "num_videos": 1,
        "resolution": "1080p",
        "demucs_model": "htdemucs_ft",
        "device": "auto",
        "shifts": 5,
        "asr_method": "FunASR",
        "whisper_model": "large",
        "batch_size": 32,
        "diarization": True,
        "whisper_min_speakers": None,
        "whisper_max_speakers": None,
        "translation_method": "阿里云-通义千问",
        "translation_target_language": "English",
        "tts_method": "xtts",
        "tts_target_language": "English",
        "voice": "zh-CN-XiaoxiaoNeural",
        "subtitles": True,
        "speed_up": 1.0,
        "fps": 30,
        "bgm_volume": 0.5,
        "video_volume": 1.0,
        "target_resolution": "720p",
        "max_workers": 1,
        "max_retries": 3
    }

    # 确保输出目录存在
    os.makedirs("videos/test_api2", exist_ok=True)

    # 发送请求
    print("正在发送一键处理请求...")
    response = requests.post(f"{API_URL}/api/do_everything", json=payload)

    # 检查响应
    if response.status_code == 200:
        result = response.json()
        print(f"请求成功: {result['message']}")

        # 轮询检查处理状态
        print("开始轮询检查处理状态...")
        folder = payload["root_folder"]

        # 记录上一次的状态，用于比较变化
        last_status = {
            "has_video": False,
            "has_audio": False,
            "has_vocals": False,
            "has_transcript": False,
            "has_translation": False,
            "has_tts": False,
            "has_combined": False
        }

        # 处理阶段描述
        stage_descriptions = {
            "has_video": "视频下载",
            "has_audio": "音频提取",
            "has_vocals": "人声分离",
            "has_transcript": "语音识别",
            "has_translation": "翻译处理",
            "has_tts": "语音合成",
            "has_combined": "视频合成"
        }

        for i in range(60):  # 最多等待60次，每次10秒
            time.sleep(10)
            status_response = requests.get(f"{API_URL}/api/get_status/{folder}")

            if status_response.status_code == 200:
                current_status = status_response.json()["processing_status"]

                # 检查状态变化并输出
                for key, value in current_status.items():
                    if value and not last_status[key]:
                        print(f"✅ {stage_descriptions[key]}阶段完成!")

                # 更新上一次状态
                last_status = current_status.copy()

                # 计算完成百分比
                completed_stages = sum(1 for v in current_status.values() if v)
                total_stages = len(current_status)
                progress_percent = (completed_stages / total_stages) * 100

                print(f"当前进度: {progress_percent:.1f}% - {completed_stages}/{total_stages}个阶段完成")

                # 检查是否全部完成
                if current_status["has_combined"]:
                    print("\n🎉 全部处理完成!")

                    # 获取视频列表
                    videos_response = requests.get(f"{API_URL}/api/list_videos", params={"folder": folder})
                    if videos_response.status_code == 200:
                        videos = videos_response.json()["videos"]
                        if videos:
                            print(f"生成的视频文件:")
                            for idx, video in enumerate(videos, 1):
                                print(f"  {idx}. {video}")
                        else:
                            print("未找到生成的视频")
                    break
            else:
                print(f"获取状态失败: {status_response.text}")
    else:
        print(f"请求失败: {response.status_code} - {response.text}")


if __name__ == "__main__":
    test_do_everything()