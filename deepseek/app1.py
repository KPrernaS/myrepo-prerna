import re
import hashlib
import os
from docx import Document
import PyPDF2
import email
from email import policy
from email.parser import BytesParser
import spacy
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import yaml
from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")  # Replace with your MongoDB connection string
db = client["email_classifier"]
hashes_collection = db["email_hashes"]

# Global variable to store the configuration
config: Dict[str, Any] = {}

# Hugging Face API settings
HF_API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B"  # Replace with your preferred model
HF_API_TOKEN = "HFAPITOKEN"  # Replace with your Hugging Face API token

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load the YAML configuration
def load_config():
    global config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found. Please create it.")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

# Get the current configuration
def get_config():
    return config

# Load the configuration at startup
load_config()

# Extract text from .docx files
def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a .docx file.
    """
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Extract text from .pdf files
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a .pdf file.
    """
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages])

# Extract text from email files
def extract_text_from_email(email_content: bytes) -> str:
    """
    Extract text from an email file (.eml).
    """
    msg = BytesParser(policy=policy.default).parsebytes(email_content)
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_payload(decode=True).decode()
    else:
        text = msg.get_payload(decode=True).decode()
    return text

# Clean the text
def clean_text(text: str) -> str:
    """
    Clean the text by removing unwanted elements.
    """
    # Remove email signatures and disclaimers
    text = re.sub(r"(\n.*){2,}.*(signature|disclaimer).*", "", text, flags=re.IGNORECASE)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra spaces and line breaks
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Normalize the text
def normalize_text(text: str) -> str:
    """
    Normalize the text by converting to lowercase and removing stopwords.
    """
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Preprocess the text
def preprocess_text(text: str) -> str:
    """
    Preprocess the text by cleaning and normalizing it.
    """
    text = clean_text(text)
    text = normalize_text(text)
    return text

# Classify request types using Hugging Face Inference API
def classify_request_type(text: str) -> str:
    """
    Classify the request type using Hugging Face Inference API.
    """
    # Define the prompt
    prompt = f"""
    Classify the following email into one of these request types:
    - Money Movement – Inbound
    - Adjustment
    - Fee Payment

    Email: {text}

    Request Type:
    """
    # Make API call to Hugging Face
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to classify request type")
    # Extract the request type from the response
    request_type = response.json()[0]["generated_text"].split("Request Type:")[-1].strip()
    return request_type

# Extract required information using Hugging Face Inference API
def extract_required_info(text: str, required_info: List[str]) -> Dict[str, List[str]]:
    """
    Extract required information using Hugging Face Inference API.
    """
    extracted_info = {}
    for info in required_info:
        # Define the prompt
        prompt = f"""
        Extract the {info} from the following email:

        Email: {text}

        {info}:
        """
        # Make API call to Hugging Face
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {"inputs": prompt}
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to extract {info}")
        # Extract the information from the response
        extracted_info[info] = [response.json()[0]["generated_text"].split(f"{info}:")[-1].strip()]
    return extracted_info

# Check for duplicate emails using MongoDB
def check_duplicate(text: str) -> bool:
    email_hash = hashlib.md5(text.encode()).hexdigest()
    # Check if the hash exists in MongoDB
    if hashes_collection.find_one({"hash": email_hash}):
        return True
    # If not, insert the hash into MongoDB
    hashes_collection.insert_one({"hash": email_hash})
    return False

# Predict priority (dummy implementation)
def predict_priority(text: str) -> str:
    # Dummy logic (replace with actual implementation)
    if "urgent" in text.lower():
        return "High"
    return "Medium"

# Main function to process email
def process_email(text: str) -> Dict[str, Any]:
    # Preprocess text
    text = preprocess_text(text)

    # Classify request type using Hugging Face Inference API
    request_type = classify_request_type(text)

    # Extract required information using Hugging Face Inference API
    required_info = next((req["required_info"] for req in config["request_types"] if req["type"] == request_type), [])
    extracted_info = extract_required_info(text, required_info)

    # Check for duplicates
    is_duplicate = check_duplicate(text)

    # Predict priority
    priority = predict_priority(text)

    # Generate output
    return {
        "request_type": request_type,
        "extracted_info": extracted_info,
        "is_duplicate": "Yes" if is_duplicate else "No",
        "priority": priority
    }

# Process a single file
def process_file(file: UploadFile, file_type: str) -> Dict[str, Any]:
    # Save the uploaded file temporarily
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Extract text based on file type
    if file_type == "docx":
        text = extract_text_from_docx(file_path)
    elif file_type == "pdf":
        text = extract_text_from_pdf(file_path)
    elif file_type == "email":
        with open(file_path, "rb") as file:
            email_content = file.read()
        text = extract_text_from_email(email_content)
    else:
        raise ValueError("Unsupported file type")

    # Preprocess the text
    text = preprocess_text(text)

    # Process the email
    return process_email(text)

# FastAPI endpoint for single file classification
class ClassificationRequest(BaseModel):
    file_type: str

@app.post("/classify")
async def classify(file: UploadFile = File(...), file_type: str = "docx"):
    # Validate file type
    if file_type not in ["docx", "pdf", "email"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Process the file
    output = process_file(file, file_type)

    return output

# FastAPI endpoint for bulk classification
@app.post("/bulk_classify")
async def bulk_classify(files: List[UploadFile] = File(...), file_type: str = "docx"):
    # Validate file type
    if file_type not in ["docx", "pdf", "email"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Process files in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file, file_type) for file in files]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"error": str(e)})

    return {"results": results}

# FastAPI endpoint to reload configuration
@app.post("/reload_config")
async def reload_config():
    """
    Reload the YAML configuration file.
    """
    try:
        load_config()
        return {"message": "Configuration reloaded successfully"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)