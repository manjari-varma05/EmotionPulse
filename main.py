
import os
import re
import time
import pickle
import numpy as np

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from google import genai

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import asyncio
from openai import AsyncOpenAI
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


load_dotenv()

# Model configuration
mod_path = "Artifacts/final_model_fixed.keras"
tokenizer_path = "Artifacts/tokenizer.pkl"
max_seq_len = 50

emotion_labels = [
    "sadness", "joy", "love", "anger", "fear", "surprise"
]

emotion_emojis = {
    "sadness": "😢",
    "joy": "😁",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😯"
}

# Load model and tokenizer once when FastAPI starts
dl_mod = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the BiGRU model and tokenizer...")

    dl_mod["BiGRU"] = load_model(mod_path, compile=False)

    with open(tokenizer_path, "rb") as file:
        dl_mod["Tokenizer"] = pickle.load(file)

    # Initialize the LLM clients once at startup.
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        dl_mod["GeminiClient"] = genai.Client(
            api_key=gemini_key,
            http_options={
                "timeout": 2500,
                "retry_options": {"attempts": 1},
            },
        ).aio

    if openai_key:
        dl_mod["OpenAIClient"] = AsyncOpenAI(
            api_key=openai_key,
            timeout=4.0,
            max_retries=0,
        )

    print("Model, tokenizer, and available LLM clients initialized.")

    try:
        yield
    finally:
        gemini = dl_mod.get("GeminiClient")
        if gemini:
            await gemini.aclose()

        openai_client = dl_mod.get("OpenAIClient")
        if openai_client:
            await openai_client.close()

        dl_mod.clear()



app = FastAPI(lifespan=lifespan)

# Frontend configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# Request and response schemas
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyse",
        json_schema_extra={
            "example": "I feel so happy and excited"
        }
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emot: str
    confidence: float
    all_probab: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# Text preprocessing
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = (
        dl_mod.get("BiGRU") is not None
        and dl_mod.get("Tokenizer") is not None
    )

    return HealthResponse(
        status="healthy" if loaded else "not ready",
        model_loaded=loaded
    )


# Shared prediction logic
def get_prediction(text: str) -> PredictionResponse:
    bigru_model = dl_mod.get("BiGRU")
    tokenizer = dl_mod.get("Tokenizer")

    if bigru_model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet."
        )

    cleaned_text = preprocess_text(text)

    tokenized_text = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=max_seq_len,
        padding="post",
        truncating="post"
    )

    probabilities = bigru_model.predict(
        padded_sequence,
        verbose=0
    )[0]

    top_index = int(np.argmax(probabilities))

    all_probabilities = {
        label: float(probability)
        for label, probability in zip(
            emotion_labels, probabilities
        )
    }

    print("Original text:", text)
    print("Cleaned text:", cleaned_text)
    print("Predicted probabilities:", all_probabilities)

    return PredictionResponse(
        text=text,
        predicted_emot=emotion_labels[top_index],
        confidence=float(probabilities[top_index]),
        all_probab=all_probabilities
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(text_inp: TextInput):
    return get_prediction(text_inp.text)



@app.post("/analyze-emotion")
async def analyze_emotion(text_inp: TextInput):
    # Stage 1: Predict emotion using your existing BiGRU.
    prediction = get_prediction(text_inp.text)

    # Stage 2: Prepare a personalized prompt.
    prompt = f"""
You are EmotionPulse, an empathetic emotion-analysis assistant.

User's original message:
{text_inp.text}

BiGRU predicted emotion: {prediction.predicted_emot}
BiGRU confidence: {prediction.confidence:.2%}

Instructions:
- Consider the user's actual words and their context.
- Treat the predicted emotion as a clue, not a certainty.
- Generate a personalized, warm, natural response.
- Acknowledge the specific situation described by the user.
- Do not use generic, repetitive responses.
- Do not invent personal details or diagnose the user.
- Keep the response concise and conversational.
- Return:
Likely emotion: ...
Explanation: ...
Supportive response: ...
"""

    errors = []

    # Stage 3: Try Gemini first, with a strict deadline.
    gemini = dl_mod.get("GeminiClient")

    if gemini:
        try:
            result = await asyncio.wait_for(
                gemini.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                ),
                timeout=2.7,
            )

            if result.text and result.text.strip():
                return {
                    "text": text_inp.text,
                    "bigru_emotion": prediction.predicted_emot,
                    "bigru_confidence": prediction.confidence,
                    "all_probab": prediction.all_probab,
                    "llm_analysis": result.text.strip(),
                    "llm_provider": "gemini",
                }

            errors.append("Gemini returned an empty response.")

        except Exception as e:
            errors.append(f"Gemini failed: {type(e).__name__}")
            print("Gemini request failed:", repr(e))

    # Stage 4: If Gemini fails or times out, use OpenAI.
    openai_client = dl_mod.get("OpenAIClient")

    if openai_client:
        try:
            result = await asyncio.wait_for(
                openai_client.responses.create(
                    model=os.getenv(
                        "OPENAI_MODEL", "gpt-4.1-mini"
                    ),
                    input=prompt,
                ),
                timeout=4.5,
            )

            if result.output_text and result.output_text.strip():
                return {
                    "text": text_inp.text,
                    "bigru_emotion": prediction.predicted_emot,
                    "bigru_confidence": prediction.confidence,
                    "all_probab": prediction.all_probab,
                    "llm_analysis": result.output_text.strip(),
                    "llm_provider": "openai",
                }

            errors.append("OpenAI returned an empty response.")

        except Exception as e:
            errors.append(f"OpenAI failed: {type(e).__name__}")
            print("OpenAI request failed:", repr(e))

    # Stage 5: Both LLM providers failed.
    # Do not pretend a personalized LLM response was generated.
    raise HTTPException(
        status_code=503,
        detail={
            "message": (
                "Personalized response generation is "
                "temporarily unavailable. Please try again."
            ),
            "bigru_emotion": prediction.predicted_emot,
            "bigru_confidence": prediction.confidence,
            "errors": errors,
        },
    )
