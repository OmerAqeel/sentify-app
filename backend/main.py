from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React app URL
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the sentiment analysis pipeline using a BERT model
model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")


class TextInput(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_sentiment(input: TextInput):
    prediction = model(input.text)[0]  # Get the first (and only) result
    result = {
        "sentiment": prediction['label'],
        "confidence": round(prediction['score'] * 100, 2)
    }
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
