from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
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
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")

# Load a text generation model (e.g., GPT-3/4)
recommendation_model = pipeline("text-generation", model="./gpt2-local") # Use gpt-2 for local purposes

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_sentiment(input: TextInput):
    prediction = sentiment_model(input.text)[0]  # Get the first (and only) result
    result = {
        "sentiment": prediction['label'],
        "confidence": round(prediction['score'] * 100, 2)
    }
    return result

@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Perform sentiment analysis on each feedback
    sentiments = [sentiment_model(feedback)[0] for feedback in df['Feedback']]
    
    # Filter negative feedbacks
    df['Sentiment'] = [s['label'] for s in sentiments]
    negative_feedbacks = df[df['Sentiment'] == 'NEGATIVE']['Feedback'].tolist()

    # Generate dynamic recommendations based on the negative feedbacks
    recommendations = generate_dynamic_recommendations(negative_feedbacks)
    
    # Pass the DataFrame and sentiments to the analysis function
    sentiment_result = analyze_sentiment_counts(df, sentiments)
    
    return {
        **sentiment_result,  # Include sentiment counts and plot
        "recommendations": recommendations  # Include dynamic recommendations
    }

def analyze_sentiment_counts(df: pd.DataFrame, sentiments: list) -> dict:
    """
    Analyze sentiment counts for each product in the given DataFrame.
    
    Parameters:
    - df: DataFrame containing 'Product', 'Feedback', and 'Date' columns.
    - sentiments: List of sentiment analysis results corresponding to each feedback.
    
    Returns:
    - A dictionary containing the sentiment summary and a base64-encoded bar chart image.
    """
    # Add the sentiments to the DataFrame
    df['Sentiment'] = [s['label'] for s in sentiments]

    # Count the number of positive, negative, and neutral sentiments for each product
    sentiment_counts = df.groupby(['Product', 'Sentiment']).size().unstack(fill_value=0)

    # Visualization: Sentiment Counts by Product
    sentiment_counts.plot(kind='bar', stacked=False, figsize=(8, 4))

    plt.xlabel('Product')
    plt.ylabel('Count of Feedbacks')
    plt.title('Sentiment Counts by Product')
    plt.xticks(rotation=45)
    plt.legend(title="Sentiment")
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Return the sentiment summary and the plot as a base64 string
    return {
        "product_summary": sentiment_counts.to_dict(),
        "plot_image": image_base64
    }

def generate_dynamic_recommendations(feedbacks):
    """
    Generate dynamic recommendations based on negative feedbacks using a text generation model.
    """
    recommendations = []
    for feedback in feedbacks:
        # Prompt the text generation model
        prompt = f"Based on the following feedback: '{feedback}', what improvements would you suggest?"
        suggestion = recommendation_model(prompt, max_length=50)[0]['generated_text']
        recommendations.append({
            "feedback": feedback,
            "recommendation": suggestion
        })
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
