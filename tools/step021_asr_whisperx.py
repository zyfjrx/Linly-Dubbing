import json
import time
import librosa
import numpy as np
import whisperx
import os
from loguru import logger
import torch
from dotenv import load_dotenv
load_dotenv()

whisper_model = None
diarize_model = None

align_model = None
language_code = None
align_metadata = None
DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}
DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
}
def init_whisperx():
    load_whisper_model()
    load_align_model()

def init_diarize():
    load_diarize_model()
    
def load_whisper_model(model_name='large', download_root = 'models/ASR/whisper', device='auto'):
    if model_name == 'large':
        pretrain_model = os.path.join(download_root,"faster-whisper-large-v3")
        model_name = 'large-v3' if not os.path.isdir(pretrain_model) else pretrain_model
    elif model_name == 'medium':
        pretrain_model = os.path.join(download_root,"faster-whisper-medium")
        model_name = 'medium' if not os.path.isdir(pretrain_model) else pretrain_model
    elif model_name == 'small':
        pretrain_model = os.path.join(download_root,"faster-whisper-small")
        model_name = 'small' if not os.path.isdir(pretrain_model) else pretrain_model
    elif model_name == 'tiny':
        pretrain_model = os.path.join(download_root,"faster-whisper-tiny")
        model_name = 'tiny' if not os.path.isdir(pretrain_model) else pretrain_model
    
    global whisper_model,model_type
    if whisper_model is not None and model_name == model_type:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Loading WhisperX model: {model_name}')
    t_start = time.time()
    model_type = model_name
    if device=='cpu':
        whisper_model = whisperx.load_model(model_type, download_root=download_root, device=device, compute_type='int8')
    else:
        whisper_model = whisperx.load_model(model_type, download_root=download_root, device=device)
    t_end = time.time()
    logger.info(f'Loaded WhisperX model: {model_type} in {t_end - t_start:.2f}s')

def load_align_model(language='en', device='auto', model_dir='models/ASR/whisper'):
    global align_model, language_code, align_metadata
    if align_model is not None and language_code == language:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    language_code = language
    t_start = time.time()
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language_code, device=device, model_dir = model_dir)
    t_end = time.time()
    logger.info(f'Loaded alignment model: {language_code} in {t_end - t_start:.2f}s')
    
def load_diarize_model(device='auto'):
    global diarize_model
    if diarize_model is not None:
        return
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_start = time.time()
    try:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv('HF_TOKEN'), device=device)
        t_end = time.time()
        logger.info(f'Loaded diarization model in {t_end - t_start:.2f}s')
    except Exception as e:
        t_end = time.time()
        logger.error(f"Failed to load diarization model in {t_end - t_start:.2f}s due to {str(e)}")
        logger.info("You have not set the HF_TOKEN, so the pyannote/speaker-diarization-3.1 model could not be downloaded.")
        logger.info("If you need to use the speaker diarization feature, please request access to the pyannote/speaker-diarization-3.1 model. Alternatively, you can choose not to enable this feature.")

def whisperx_transcribe_audio(wav_path, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32, diarization=True,min_speakers=None, max_speakers=None):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    load_whisper_model(model_name, download_root, device)
    rec_result = whisper_model.transcribe(wav_path, batch_size=batch_size)
    
    if rec_result['language'] == 'nn':
        logger.warning(f'No language detected in {wav_path}')
        return False
    
    load_align_model(rec_result['language'],device='auto', model_dir=download_root)
    rec_result = whisperx.align(rec_result['segments'], align_model, align_metadata,
                                wav_path, device, return_char_alignments=False)
    
    if diarization:
        load_diarize_model(device)
        if diarize_model:
            diarize_segments = diarize_model(wav_path,min_speakers=min_speakers, max_speakers=max_speakers)
            rec_result = whisperx.assign_word_speakers(diarize_segments, rec_result)
        else:
            logger.warning("Diarization model is not loaded, skipping speaker diarization")
        
    transcript = [{'start': segement['start'], 'end': segement['end'], 'text': segement['text'].strip(), 'speaker': segement.get('speaker', 'SPEAKER_00')} for segement in rec_result['segments']]
    return transcript


if __name__ == '__main__':
    # for root, dirs, files in os.walk("videos"):
    #     if 'audio_vocals.wav' in files:
    #         logger.info(f'Transcribing {os.path.join(root, "audio_vocals.wav")}')
    #         transcript = whisperx_transcribe_audio(os.path.join(root, "audio_vocals.wav"))
    #         print(transcript)
    #         break
    transcript = whisperx_transcribe_audio(model_name="medium",wav_path="/home/bmh/Linly-Dubbing/test/b0a9355ecd1564add481bd4f810e8892.mp4",download_root="/home/bmh/Linly-Dubbing/models/ASR/whisper")
    print(transcript)