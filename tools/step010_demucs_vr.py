import shutil
from demucs.api import Separator
import os
from loguru import logger
import time
from .utils import save_wav, normalize_wav
import torch
import gc

# 全局变量
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
separator = None
model_loaded = False  # 新增标志，跟踪模型是否已加载
current_model_config = {}  # 新增变量，存储当前加载模型的配置


def init_demucs():
    """
    初始化Demucs模型。
    如果模型已经初始化，直接返回而不重新加载。
    """
    global separator, model_loaded
    if not model_loaded:
        separator = load_model()
        model_loaded = True
    else:
        logger.info("Demucs模型已经加载，跳过初始化")


def load_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True,
               shifts: int = 5) -> Separator:
    """
    加载Demucs模型。
    如果相同配置的模型已加载，直接返回现有模型而不重新加载。
    """
    global separator, model_loaded, current_model_config

    if separator is not None:
        # 检查是否需要重新加载模型（配置不同）
        requested_config = {
            'model_name': model_name,
            'device': 'auto' if device == 'auto' else device,
            'shifts': shifts
        }

        if current_model_config == requested_config:
            logger.info(f'Demucs模型已加载且配置相同，重用现有模型')
            return separator
        else:
            logger.info(f'Demucs模型配置改变，需要重新加载')
            # 释放现有模型资源
            release_model()

    logger.info(f'加载Demucs模型: {model_name}')
    t_start = time.time()

    device_to_use = auto_device if device == 'auto' else device
    separator = Separator(model_name, device=device_to_use, progress=progress, shifts=shifts)

    # 存储当前模型配置
    current_model_config = {
        'model_name': model_name,
        'device': 'auto' if device == 'auto' else device,
        'shifts': shifts
    }

    model_loaded = True
    t_end = time.time()
    logger.info(f'Demucs模型加载完成，用时 {t_end - t_start:.2f} 秒')

    return separator


def release_model():
    """
    释放模型资源，避免内存泄漏
    """
    global separator, model_loaded, current_model_config

    if separator is not None:
        logger.info('正在释放Demucs模型资源...')
        # 删除引用
        separator = None
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_loaded = False
        current_model_config = {}
        logger.info('Demucs模型资源已释放')


def separate_audio(folder: str, model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True,
                   shifts: int = 5) -> None:
    """
    分离音频文件
    """
    global separator
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return None, None
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')

    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f'音频已分离: {folder}')
        return vocal_output_path, instruments_output_path

    logger.info(f'正在分离音频: {folder}')

    try:
        # 确保模型已加载并且配置正确
        if not model_loaded or current_model_config.get('model_name') != model_name or \
                (current_model_config.get('device') == 'auto') != (device == 'auto') or \
                current_model_config.get('shifts') != shifts:
            load_model(model_name, device, progress, shifts)

        t_start = time.time()

        try:
            origin, separated = separator.separate_audio_file(audio_path)
        except Exception as e:
            logger.error(f'音频分离出错: {e}')
            # 在发生错误时尝试重新加载模型一次
            release_model()
            load_model(model_name, device, progress, shifts)
            logger.info(f'已重新加载模型，重试分离...')
            origin, separated = separator.separate_audio_file(audio_path)

        t_end = time.time()
        logger.info(f'音频分离完成，用时 {t_end - t_start:.2f} 秒')

        vocals = separated['vocals'].numpy().T
        instruments = None
        for k, v in separated.items():
            if k == 'vocals':
                continue
            if instruments is None:
                instruments = v
            else:
                instruments += v
        instruments = instruments.numpy().T

        save_wav(vocals, vocal_output_path, sample_rate=44100)
        logger.info(f'已保存人声: {vocal_output_path}')

        save_wav(instruments, instruments_output_path, sample_rate=44100)
        logger.info(f'已保存伴奏: {instruments_output_path}')

        return vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'分离音频失败: {str(e)}')
        # 出现错误，释放模型资源并重新抛出异常
        release_model()
        raise


def extract_audio_from_video(folder: str) -> bool:
    """
    从视频中提取音频
    """
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        return False
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'音频已提取: {folder}')
        return True
    logger.info(f'正在从视频提取音频: {folder}')

    os.system(
        f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"')

    time.sleep(1)
    logger.info(f'音频提取完成: {folder}')
    return True


def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft", device: str = 'auto',
                                    progress: bool = True, shifts: int = 5) -> None:
    """
    分离文件夹下所有音频
    """
    global separator
    vocal_output_path, instruments_output_path = None, None

    try:
        for subdir, dirs, files in os.walk(root_folder):
            if 'download.mp4' not in files:
                continue
            if 'audio.wav' not in files:
                extract_audio_from_video(subdir)
            if 'audio_vocals.wav' not in files:
                vocal_output_path, instruments_output_path = separate_audio(subdir, model_name, device, progress,
                                                                            shifts)
            elif 'audio_vocals.wav' in files and 'audio_instruments.wav' in files:
                vocal_output_path = os.path.join(subdir, 'audio_vocals.wav')
                instruments_output_path = os.path.join(subdir, 'audio_instruments.wav')
                logger.info(f'音频已分离: {subdir}')

        logger.info(f'已完成所有音频分离: {root_folder}')
        return f'所有音频分离完成: {root_folder}', vocal_output_path, instruments_output_path

    except Exception as e:
        logger.error(f'分离音频过程中出错: {str(e)}')
        # 出现任何错误，释放模型资源
        release_model()
        raise


if __name__ == '__main__':
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)