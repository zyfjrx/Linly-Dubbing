import os
from loguru import logger
model = None
from gradio_client import Client, handle_file
import shutil

# API_KEY = load_key("f5tts.302_api")
AUDIO_REFERS_DIR = "output/audio/refers"
UPLOADED_REFER_URL = None
GRADIO_CLIENT = Client("http://192.168.50.70:7860/")
NORMALIZED_REFERS_CACHE = {}

def _f5_tts(text: str, speaker_wav: str) -> bool:
    try:
        result = GRADIO_CLIENT.predict(
            ref_audio_input=handle_file(speaker_wav),
            ref_text_input=text,  # 使用相同的文本作为参考
            gen_text_input=text,
            remove_silence=False,
            cross_fade_duration_slider=0.15,
            nfe_slider=32,
            speed_slider=1,
            api_name="/basic_tts"
        )

        # 结果是一个元组，第一个元素是生成的音频文件路径
        generated_audio_path = result[0]

        # # 复制生成的音频文件到目标路径
        # shutil.copy2(generated_audio_path, save_path)
        # print(f"Audio file saved to {save_path}")
        return generated_audio_path

    except Exception as e:
        print(f"Request failed: {str(e)}")
        return False

def tts(text, output_path, speaker_wav):

    if os.path.exists(output_path):
        logger.info(f'TTS {text} 已存在')
        return
    for retry in range(3):
        try:

            generated_audio_path = _f5_tts(text, speaker_wav)
            shutil.copy2(generated_audio_path, output_path)
            logger.info(f'TTS {text}')
            break
        except Exception as e:
            logger.warning(f'TTS {text} 失败')
            logger.warning(e)


if __name__ == '__main__':
    # speaker_wav = r'videos/村长台钓加拿大/20240805 英文无字幕 阿里这小子在水城威尼斯发来问候/audio_vocals.wav'
    # os.makedirs('playground', exist_ok=True)
    # while True:
    #     text = input('请输入：')
    #     tts(text, f'playground/{text}.wav', speaker_wav)
    tts("阿里这小子在水城威尼斯发来问候","test.wav","/home/bmh/Linly-Dubbing/videos/leijun-test-sr-1.wav")
