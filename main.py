from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os

from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


class Request(BaseModel):
    image_base64: str
    question: str


@app.post("/answer-image")
def answer(req: Request):
    image_bytes = base64.b64decode(req.image_base64)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            ),
            req.question
            + "\nReturn ONLY the answer. "
              "If numeric, return only the number as a string. "
              "No units. No explanation."
        ],
    )

    return {"answer": response.text.strip()}