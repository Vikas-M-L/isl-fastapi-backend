from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import os

from predict_alphabet import predict_alphabet_from_base64
from predict_numbers import predict_number_from_base64


app = FastAPI(title="ISL Detector API", version="1.0.0")

# CORS origins: read from env FRONTEND_ORIGINS or default to localhost ports
origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def index():
    return {
        "name": "ISL Detector API (FastAPI)",
        "endpoints": [
            {"method": "POST", "path": "/detect/alphabet", "body": {"image": "dataUrl"}},
            {"method": "POST", "path": "/detect/number", "body": {"image": "dataUrl"}},
        ],
    }


@app.post("/detect/alphabet")
async def detect_alphabet(payload: dict = Body(...)):
    image = payload.get("image")
    if not image:
        return {"success": False, "message": "Missing 'image' field", "sign": None, "confidence": 0.0}
    return predict_alphabet_from_base64(image)


@app.post("/detect/number")
async def detect_number(payload: dict = Body(...)):
    image = payload.get("image")
    if not image:
        return {"success": False, "message": "Missing 'image' field", "sign": None, "confidence": 0.0}
    return predict_number_from_base64(image)
