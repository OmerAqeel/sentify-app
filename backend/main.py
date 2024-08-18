from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

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
async def analyze_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Perform sentiment analysis on each feedback
    sentiments = [model(feedback)[0] for feedback in df['Feedback']]
    
    # Pass the DataFrame and sentiments to the analysis function
    result = analyze_sentiment_trends(df, sentiments)
    
    return result


def analyze_sentiment_trends(df: pd.DataFrame, sentiments: list) -> dict:
    """
    Analyze sentiment trends over time for each product in the given DataFrame.
    
    Parameters:
    - df: DataFrame containing 'Product', 'Feedback', and 'Date' columns.
    - sentiments: List of sentiment analysis results corresponding to each feedback.
    
    Returns:
    - A dictionary containing the sentiment summary and a base64-encoded plot image.
    """
    # Convert the Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Add the sentiments to the DataFrame
    df['Sentiment'] = [s['label'] for s in sentiments]
    df['Sentiment_Score'] = df['Sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0})
    
    # Group by product and date to calculate the average sentiment score
    sentiment_over_time = df.groupby(['Product', pd.Grouper(key='Date', freq='D')]).agg({
        'Sentiment_Score': 'mean'
    }).reset_index()

    # Visualization: Sentiment Trend Over Time by Product
    plt.figure(figsize=(10, 6))
    for product in sentiment_over_time['Product'].unique():
        product_data = sentiment_over_time[sentiment_over_time['Product'] == product]
        plt.plot(product_data['Date'], product_data['Sentiment_Score'], label=product)

    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.title('Sentiment Trend Over Time by Product')
    plt.legend()
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Return the sentiment summary and the plot as a base64 string
    return {
        "product_summary": sentiment_over_time.to_dict(orient='records'),
        "plot_image": image_base64
    }


     

   


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
