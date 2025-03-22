
from flask import Flask, request, jsonify
import yaml
import spacy
import hashlib
import openai
import os
import re
from email import message_from_string
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
import json
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient

def load_config():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client.email_classification
duplicates_collection = db.duplicates

# Create an index on the hash field for faster lookup
duplicates_collection.create_index("email_hash", unique=True)

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def classify_with_openai(text):
    response = openai.Completion.create(
        model="fine-tuned-model-id",
        prompt=f"Classify the following email:\n{text}\nOutput: ",
        max_tokens=500
    )
    return response['choices'][0]['text'].strip()

def determine_priority(classifications):
    priority_types = ["Money Movement – Inbound", "Money Movement – Outbound"]
    for cls in classifications:
        if any(priority_type in cls for priority_type in priority_types):
            return "High"
    return "Low"

def extract_required_info(classifications, config):
    result = {}
    confidence_score = 0.85

    for cls in classifications:
        main_type = cls.split(",")[0].split(":")[-1].strip()
        subtype = cls.split(",")[1].split(":")[-1].strip() if "," in cls else ""
        
        if main_type in config.get('request_types', {}):
            required_info = config['request_types'][main_type].get('required_info', [])
            if main_type not in result:
                result[main_type] = {
                    "confidence_score": confidence_score,
                    "priority": determine_priority([cls]),
                    "required_info": required_info,
                    "subtypes": []
                }
            if subtype:
                result[main_type]["subtypes"].append(subtype)
    
    return result

def parse_email(email_content):
    msg = message_from_string(email_content)
    if msg.is_multipart():
        parts = [part.get_payload(decode=True).decode(errors='ignore') for part in msg.walk() if part.get_content_type() == 'text/plain']
        email_body = '\n'.join(parts)
    else:
        email_body = msg.get_payload(decode=True).decode(errors='ignore')
    return email_body

def parse_pdf(file):
    return extract_pdf_text(file)

def parse_docx(file):
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def preprocess_text(text):
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def classify_email(text, config):
    email_hash = hashlib.md5(text.encode()).hexdigest()
    is_duplicate = "No"
    
    if duplicates_collection.find_one({"email_hash": email_hash}):
        is_duplicate = "Yes"
    else:
        duplicates_collection.insert_one({"email_hash": email_hash})
    
    cleaned_text = preprocess_text(text)
    openai_result = classify_with_openai(cleaned_text)
    classifications = openai_result.splitlines()
    structured_output = extract_required_info(classifications, config)
    entities = extract_entities(cleaned_text)
    
    return {
        "is_duplicate": is_duplicate,
        "requests": structured_output,
        "entities": entities
    }

def classify_file(file, config):
    if file.filename.endswith('.pdf'):
        text = parse_pdf(file)
    elif file.filename.endswith('.docx'):
        text = parse_docx(file)
    else:
        email_content = file.read().decode('utf-8')
        text = parse_email(email_content)
    
    return classify_email(text, config)

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    files = request.files.getlist('files')
    config = load_config()
    
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(classify_file, file, config) for file in files]
        for future in futures:
            results.append(future.result())
    
    return jsonify({"batch_results": results})

@app.route('/fine_tune', methods=['POST'])
def fine_tune():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    dataset_path = f"./{file.filename}"
    file.save(dataset_path)
    
    openai_file = openai.File.create(file=open(dataset_path), purpose="fine-tune")
    file_id = openai_file.id
    
    fine_tune_job = openai.FineTune.create(training_file=file_id, model="gpt-3.5-turbo")
    return jsonify({"fine_tune_job_id": fine_tune_job.id})

@app.route('/fine_tune_status', methods=['GET'])
def fine_tune_status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({"error": "No job ID provided"}), 400
    
    status = openai.FineTune.retrieve(id=job_id)
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
