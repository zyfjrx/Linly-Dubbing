import os
from typing import List, Optional, Union
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import tempfile
import shutil

from tools.step000_video_downloader import download_from_url
from tools.step010_demucs_vr import separate_all_audio_under_folder
from tools.step020_asr import transcribe_all_audio_under_folder
from tools.step030_translation import translate_all_transcript_under_folder
from tools.step040_tts import generate_all_wavs_under_folder
from tools.step050_synthesize_video import synthesize_all_video_under_folder
from tools.do_everything import do_everything
from tools.utils import SUPPORT_VOICE

app = FastAPI(title="Linly-Dubbing API", description="智能视频多语言AI配音/翻译工具 API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
os.makedirs("videos", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 定义请求模型
class VideoDownloadRequest(BaseModel):
    video_url: str
    output_folder: str = "videos"
    resolution: str = "1080p"
    num_videos: int = 5

class DemucsRequest(BaseModel):
    folder: str = "videos"
    model: str = "htdemucs_ft"
    device: str = "auto"
    show_progress: bool = True
    shifts: int = 5

class ASRRequest(BaseModel):
    folder: str = "videos"
    asr_method: str = "FunASR"
    whisper_model: str = "large"
    device: str = "auto"
    batch_size: int = 32
    diarization: bool = True
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

class TranslationRequest(BaseModel):
    folder: str = "videos"
    method: str = "LLM"
    target_language: str = "简体中文"

class TTSRequest(BaseModel):
    folder: str = "videos"
    method: str = "xtts"
    target_language: str = "中文"
    voice: str = "zh-CN-XiaoxiaoNeural"

class VideoSynthesisRequest(BaseModel):
    folder: str = "videos"
    subtitles: bool = True
    speed_up: float = 1.0
    fps: int = 30
    background_music: Optional[str] = None
    bgm_volume: float = 0.5
    video_volume: float = 1.0
    resolution: str = "1080p"

class DoEverythingRequest(BaseModel):
    root_folder: str = "videos"
    video_url: str
    num_videos: int = 5
    resolution: str = "1080p"
    demucs_model: str = "htdemucs_ft"
    device: str = "auto"
    shifts: int = 5
    asr_method: str = "FunASR"
    whisper_model: str = "large"
    batch_size: int = 32
    diarization: bool = True
    whisper_min_speakers: Optional[int] = None
    whisper_max_speakers: Optional[int] = None
    translation_method: str = "LLM"
    translation_target_language: str = "简体中文"
    tts_method: str = "xtts"
    tts_target_language: str = "中文"
    voice: str = "zh-CN-XiaoxiaoNeural"
    subtitles: bool = True
    speed_up: float = 1.0
    fps: int = 30
    bgm_volume: float = 0.5
    video_volume: float = 1.0
    target_resolution: str = "1080p"
    max_workers: int = 1
    max_retries: int = 3

# 后台任务处理
async def process_task(task_func, *args, **kwargs):
    return task_func(*args, **kwargs)

# API 路由
@app.get("/")
async def root():
    return {"message": "欢迎使用 Linly-Dubbing API"}

@app.post("/api/download")
async def api_download(request: VideoDownloadRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            process_task, 
            download_from_url, 
            request.video_url, 
            request.output_folder, 
            request.resolution, 
            request.num_videos
        )
        return {"status": "success", "message": "下载任务已启动"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/demucs")
async def api_demucs(request: DemucsRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            process_task, 
            separate_all_audio_under_folder, 
            request.folder, 
            request.model, 
            request.device, 
            request.show_progress, 
            request.shifts
        )
        return {"status": "success", "message": "人声分离任务已启动"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/asr")
async def api_asr(request: ASRRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            process_task, 
            transcribe_all_audio_under_folder, 
            request.folder, 
            request.asr_method, 
            request.whisper_model, 
            request.device, 
            request.batch_size, 
            request.diarization, 
            request.min_speakers, 
            request.max_speakers
        )
        return {"status": "success", "message": "语音识别任务已启动"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/translation")
async def api_translation(request: TranslationRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            process_task, 
            translate_all_transcript_under_folder, 
            request.folder, 
            request.method, 
            request.target_language
        )
        return {"status": "success", "message": "翻译任务已启动"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/tts")
async def api_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            process_task, 
            generate_all_wavs_under_folder, 
            request.folder, 
            request.method, 
            request.target_language, 
            request.voice
        )
        return {"status": "success", "message": "语音合成任务已启动"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/synthesize")
async def api_synthesize(request: VideoSynthesisRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            process_task, 
            synthesize_all_video_under_folder, 
            request.folder, 
            request.subtitles, 
            request.speed_up, 
            request.fps, 
            request.background_music, 
            request.bgm_volume, 
            request.video_volume, 
            request.resolution
        )
        return {"status": "success", "message": "视频合成任务已启动"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/do_everything")
async def api_do_everything(request: DoEverythingRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(
            process_task, 
            do_everything, 
            request.root_folder,
            request.video_url,
            request.num_videos,
            request.resolution,
            request.demucs_model,
            request.device,
            request.shifts,
            request.asr_method,
            request.whisper_model,
            request.batch_size,
            request.diarization,
            request.whisper_min_speakers,
            request.whisper_max_speakers,
            request.translation_method,
            request.translation_target_language,
            request.tts_method,
            request.tts_target_language,
            request.voice,
            request.subtitles,
            request.speed_up,
            request.fps,
            None,  # background_music
            request.bgm_volume,
            request.video_volume,
            request.target_resolution,
            request.max_workers,
            request.max_retries
        )
        return {"status": "success", "message": "一键处理任务已启动"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/upload_background_music")
async def upload_background_music(file: UploadFile = File(...)):
    try:
        # 保存上传的背景音乐
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        try:
            contents = await file.read()
            with open(temp_file.name, 'wb') as f:
                f.write(contents)
        finally:
            await file.close()
        
        # 移动到静态文件目录
        dest_path = os.path.join("static", "bgm", file.filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(temp_file.name, dest_path)
        
        return {"status": "success", "filename": file.filename, "path": dest_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/list_videos")
async def list_videos(folder: str = "videos"):
    try:
        videos = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    videos.append(os.path.join(root, file))
        return {"status": "success", "videos": videos}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/get_video/{video_path:path}")
async def get_video(video_path: str):
    try:
        if os.path.exists(video_path) and video_path.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            return FileResponse(video_path)
        else:
            return JSONResponse(status_code=404, content={"status": "error", "message": "视频文件不存在"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/api/get_voices")
async def get_voices():
    return {"status": "success", "voices": SUPPORT_VOICE}

@app.get("/api/get_status/{folder:path}")
async def get_status(folder: str):
    try:
        status = {
            "has_video": False,
            "has_audio": False,
            "has_vocals": False,
            "has_transcript": False,
            "has_translation": False,
            "has_tts": False,
            "has_combined": False
        }
        
        # 检查各个处理阶段的文件是否存在
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                    status["has_video"] = True
                if file == "audio.wav":
                    status["has_audio"] = True
                if file == "audio_vocals.wav":
                    status["has_vocals"] = True
                if file == "transcript.json":
                    status["has_transcript"] = True
                if file == "translation.json":
                    status["has_translation"] = True
                if file == "audio_tts.wav":
                    status["has_tts"] = True
                if file == "audio_combined.wav":
                    status["has_combined"] = True
        
        return {"status": "success", "processing_status": status}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=6006)