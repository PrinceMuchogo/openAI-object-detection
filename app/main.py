from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import List
import io
import base64
import openai
import os
import cv2
import numpy as np

from app.openAi.detect_object import detect_object
from app.utils.upload_image import upload_image

# Set OpenAI key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Object Detection API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ObjectDetectionResponse(BaseModel):
    object_name: str
    confidence: float

class MultipleObjectDetectionResponse(BaseModel):
    detections: List[ObjectDetectionResponse]


@app.post("/detect")
async def detect(request: Request):
    try:
        # Read raw image bytes from request body
        image_bytes = await request.body()

        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Upload to image hosting service
        image_url = upload_image(image)

        # Run object detection using OpenAI Vision
        detection_result = detect_object(image_url)

        return {"result": detection_result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

@app.get("/")
def root():
    return {"message": "Object Detection API is running!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
