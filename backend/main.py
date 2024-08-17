from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import pandas as pd

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

@app.post("/analyze_csv")
async def analyze_sentiment(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    sentiments = []

    for feedback in df['Feedback']:
        prediction = model(feedback)[0]
        result = {
            "feedback": feedback,
            "sentiment": prediction['label'],
            "confidence": round(prediction['score'] * 100, 2)
        }
        sentiments.append(result)
    
     # Add the sentiments to the DataFrame or return as a list
    df['sentiment'] = [s['sentiment'] for s in sentiments]
    df['confidence'] = [s['confidence'] for s in sentiments]
    
    # Example: Return summary statistics for each product (assuming 'product' column exists)
    product_summary = df.groupby('product').agg({
        'sentiment': lambda x: x.value_counts().idxmax(),  # Most common sentiment
        'confidence': 'mean'  # Average confidence
    }).reset_index()

    return product_summary.to_dict(orient='records')




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
