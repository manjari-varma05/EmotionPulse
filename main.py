from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
import re
from pydantic import BaseModel,Field
from keras.models import load_model
import pickle
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
app=FastAPI(
  
)

@app.get('/')
def greet():
  return {"Hello"}

mod_path="Artifacts/BiGRU_model_fixed.keras"

tokenizer_path="Artifacts/tokenizer.pkl"

max_seq_len=50

emotion_labels=['sadness','joy','love','anger','fear','surprise']



emotion_emojis = {
    "sadness": "😢",
    "joy": "😁",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😯"
}


def preprocess_text(text:str)->str:
  text=text.lower()
  text=re.sub(r"'","",text)
  text=re.sub(r"[^a-z0-9\s]"," ", text)
  text=re.sub(r"\s+"," ",text).strip()
  return text

class TextInput(BaseModel):
  text : str=Field(...,
                   min_length=1,
                   max_length=2000,
                   description='The sentence to analyse',
                   json_schema_extra={'example':"I feel so happy and excited"}
                   )
class PredictionResponse(BaseModel):
  text: str
  predicted_emot: str
  confidence: float
  all_probab: dict[str,float]

class HealthResponse(BaseModel):
  status: str
  model_loaded: bool

"""
Load the model and tokenizer once the server starts up
"""
dl_mod={}

@asynccontextmanager
async def lifespan(app: FastAPI):
  print("Loading the model and tokenizer...")
  dl_mod['BiGRU']=load_model(mod_path)
  with open(tokenizer_path,'rb') as file:
    dl_mod['Tokenizer']=pickle.load(file)
  print("Models are loaded successfully")
  yield
  dl_mod.clear()
app=FastAPI(
  lifespan=lifespan
)

"""
Mount the static files to the Fastapi app
Enable CORS(Cross origin resource sharing) to allow requests from sifferent origins
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static',StaticFiles(directory="static"),name="static")

@app.get('/',include_in_schema=False)
def serve_ui():
  return FileResponse('static/index.html')

@app.get('/health',response_model=HealthResponse)
def health():
  return HealthResponse(status='healthy',model_loaded=bool(dl_mod))

@app.post('/predict',response_model=PredictionResponse)
def predict(text_inp: TextInput):
  #clean the text
  BiGRU_mod=dl_mod.get('BiGRU')
  tokenizer_mod=dl_mod.get('Tokenizer')
  if BiGRU_mod is None or tokenizer_mod is None:
    
    raise HTTPException(status_code=503,detail='Model is not loaded yet')
  cleaned_text=preprocess_text(text_inp.text)

  #tokenize the text
  tokenized_text=tokenizer_mod.texts_to_sequences([cleaned_text])
  


  padded_sequence = pad_sequences(
      tokenized_text,
      maxlen=max_seq_len,
      padding="post",
      truncating="post"
  )
  print("Original text:", text_inp.text)
  print("Cleaned text:", cleaned_text)
  print("Tokenized text:", tokenized_text)
  print("Padded shape:", padded_sequence.shape)
  
  
  probabilities = BiGRU_mod.predict(padded_sequence)[0]
  print("Predicted probabilities:", dict(zip(emotion_labels, probabilities)))
  
  top_emotion_index = int(np.argmax(probabilities))  # 4

  all_probabilities = {
      label: float(prob)
      for prob, label in zip(probabilities,emotion_labels)
  }
  test_sentences = [
        "i cant believe i finally got the offer",
        "i am happy",
        "i feel loved",
        "i hate you"
    ]

  for text in test_sentences:
        cleaned_text = preprocess_text(text)

        tokenized_text = tokenizer_mod.texts_to_sequences(
            [cleaned_text]
        )

        padded_sequence = pad_sequences(
            tokenized_text,
            maxlen=max_seq_len,
            padding="post",
            truncating="post"
        )

        probabilities = BiGRU_mod.predict(padded_sequence, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))

        print(f"\nText: {text}")
        print(f"Predicted emotion: {emotion_labels[predicted_index]}")
        print(f"Confidence: {probabilities[predicted_index]:.2%}")

  return PredictionResponse(
      text=text_inp.text,
      predicted_emot=emotion_labels[top_emotion_index],
      confidence=float(probabilities[top_emotion_index]),
      all_probab=all_probabilities
  )


    
        



