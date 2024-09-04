from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import whisper
import ffmpeg
import tempfile
import shutil
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("base")
pho_whisper = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small")

class TranscriptionResponse(BaseModel):
    transcription: str

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Define the destination path
        destination_file_path = f"uploads/{file.filename}"

        # Save the file
        with open(destination_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": file.filename, "content_type": file.content_type}

    except Exception as e:
        return JSONResponse(content={"message": str(e)}, status_code=500)
    


@app.post("/whisper", response_model=TranscriptionResponse)
async def w_transcribe(file: UploadFile = File(...)):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    
    temp_audio_path = tempfile.mktemp(suffix=".wav")
    ffmpeg.input(temp_video_path).output(temp_audio_path).run()

    
    audio = whisper.load_audio(temp_audio_path)
    result = model.transcribe(audio)

    
    os.remove(temp_video_path)
    os.remove(temp_audio_path)

    return TranscriptionResponse(transcription=result['text'])


@app.post("/phw", response_model=TranscriptionResponse)
async def phw_transcribe(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name

    
    temp_audio_path = tempfile.mktemp(suffix=".wav")
    ffmpeg.input(temp_video_path).output(temp_audio_path).run()


    with open(temp_audio_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        
    transcription = pho_whisper(audio_data)["text"]


    os.remove(temp_video_path)
    os.remove(temp_audio_path)

    return TranscriptionResponse(transcription=transcription)