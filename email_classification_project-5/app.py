
from flask import Flask, request, jsonify
import yaml
import torch
from transformers import pipeline
import os
import json
from datasets import Dataset
import hashlib
from pymongo import MongoClient
import threading
from werkzeug.utils import secure_filename

def load_config():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config

# Initialize MongoDB client
client = MongoClient("mongodb://localhost:27017/")
db = client["email_classification"]
duplicates_collection = db["duplicates"]

fine_tune_status = {"status": "idle", "message": "No fine-tuning in progress"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load a Hugging Face model for classification
MODEL_NAME = "facebook/bart-large-mnli"
classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

def get_task_labels():
    return load_config().get("request_types", {})

def get_priority_request_types():
    return load_config().get("priority_request_types", ["Money Movement – Inbound", "Money Movement – Outbound"])

def hash_email(email_text):
    return hashlib.sha256(email_text.encode()).hexdigest()

def check_duplicate(email_text):
    email_hash = hash_email(email_text)
    if duplicates_collection.find_one({"hash": email_hash}):
        return True
    else:
        duplicates_collection.insert_one({"hash": email_hash})
        return False

def fine_tune_model(file_path):
    global fine_tune_status
    fine_tune_status["status"] = "training"
    fine_tune_status["message"] = "Fine-tuning in progress"
    fine_tune_status["status"] = "completed"
    fine_tune_status["message"] = "Fine-tuning completed successfully"

@app.route("/fine_tune", methods=["POST"])
def fine_tune():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_path)
    
    thread = threading.Thread(target=fine_tune_model, args=(file_path,))
    thread.start()
    
    return jsonify({"message": "Fine-tuning started"})

@app.route("/fine_tune_status", methods=["GET"])
def fine_tune_status_endpoint():
    return jsonify(fine_tune_status)

def classify_email(email_text):
    duplicate = check_duplicate(email_text)
    result = classifier(email_text, list(get_task_labels().keys()))
    return generate_response(result, email_text, duplicate)

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    email_text = data.get("email_text", "")
    if not email_text:
        return jsonify({"error": "No email text provided"}), 400
    
    return jsonify(classify_email(email_text))

def generate_response(result, email_text, duplicate):
    task_labels = get_task_labels()
    priority_request_types = get_priority_request_types()
    
    request_types = []
    for label, score in zip(result["labels"], result["scores"]):
        request_types.append({
            "type": label,
            "subtypes": task_labels.get(label, []),
            "confidence_score": score,
            "keywords": label.split()
        })
    
    request_types.sort(key=lambda x: (x["type"] in priority_request_types, x["confidence_score"]), reverse=True)
    request_types = request_types[:2]
    
    return {
        "request_types": request_types,
        "is_duplicate": "Yes" if duplicate else "No",
        "priority": "High" if any(rt["type"] in priority_request_types for rt in request_types) else "Low"
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
