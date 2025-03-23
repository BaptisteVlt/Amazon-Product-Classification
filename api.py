from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import joblib
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model, tokenizer, and label encoder
model = AutoModelForSequenceClassification.from_pretrained("./final_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("./final_model")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize the text classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Define the input schema
class ProductInput(BaseModel):
    description: str

# Initialize FastAPI
app = FastAPI()

# Preprocessing function same as during training
def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    return text

# Inference endpoint
@app.post("/predict")
async def predict(product: ProductInput):
    try:
        # Preprocess the input
        cleaned_description = preprocess_text(product.description)

        # Predict the category
        prediction = classifier(cleaned_description)
        #Get the label name
        predicted_label = label_encoder.inverse_transform([int(prediction[0]["label"].split("_")[-1])])

        return {"main_cat": predicted_label[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}