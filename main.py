
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

    dl_mod["BiGRU"] = load_model(mod_path)

    with open(tokenizer_path, "rb") as file:
        dl_mod["Tokenizer"] = pickle.load(file)

    print("Model and tokenizer loaded successfully.")

    yield

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
def analyze_emotion(text_inp: TextInput):
    # Stage 1: Get the prediction for THIS user's sentence
    prediction = get_prediction(text_inp.text)

    # Stage 2: Check Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing."
        )

    # Stage 3: Prepare the LLM prompt
    prompt = f"""
You are EmotionPulse, an empathetic emotion-analysis assistant.

User's sentence: {text_inp.text}

BiGRU predicted emotion: {prediction.predicted_emot}
BiGRU confidence: {prediction.confidence:.2%}

Instructions:
1. Interpret the sentence using its full context.
2. Consider whether the predicted emotion fits the wording.
3. If it does not fit, explain the likely emotion suggested
   by the sentence instead of blindly trusting the prediction.
4. Give a short, warm, natural, relatable response.
5. Do not invent personal details or claim certainty about feelings.

Return these three sections:
Likely emotion: ...
Explanation: ...
Supportive response: ...
"""

    # Stage 4: Call Gemini with limited retries
    try:
        client = genai.Client(api_key=api_key)
        response = None

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                if response.text:
                    break

                print("Gemini returned an empty response.")

            except Exception as e:
                error_text = str(e).upper()

                is_temporary = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                )

                if not is_temporary:
                    raise

                print(
                    f"Gemini temporarily unavailable. "
                    f"Attempt {attempt + 1}/3: {e}"
                )

                if attempt < 2:
                    time.sleep(2 * (attempt + 1))

        # Stage 5: Keep the app usable if Gemini fails
        if response is None or not response.text:
            return {
                "text": text_inp.text,
                "bigru_emotion": prediction.predicted_emot,
                "bigru_confidence": prediction.confidence,
                "all_probab": prediction.all_probab,
                "llm_analysis": (
                    "Gemini is temporarily unavailable. "
                    "Your BiGRU prediction is ready, but the "
                    "supportive response could not be generated. "
                    "Please try again later."
                )
            }

        # Stage 6: Return BiGRU results and Gemini response
        return {
            "text": text_inp.text,
            "bigru_emotion": prediction.predicted_emot,
            "bigru_confidence": prediction.confidence,
            "all_probab": prediction.all_probab,
            "llm_analysis": response.text
        }

    except HTTPException:
        raise

    except Exception as e:
        print("GEMINI ERROR:", repr(e))

        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini request failed. Check the backend "
                "terminal for the exact error."
            )
        )
