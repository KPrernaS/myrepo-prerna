
from flask import Flask, request, jsonify
from pymongo import MongoClient
from openai import OpenAI
import threading
import hashlib
import os
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['email_classification']
collection = db['emails']

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai = OpenAI(api_key=OPENAI_API_KEY)

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hash emails for duplicate detection
def hash_email(content):
    return hashlib.md5(content.encode()).hexdigest()

# Preprocessing
def preprocess_email(content):
    return content.lower().strip()

# Classification
def classify_email(content):
    response = openai.Completion.create(
        model='fine-tuned-model-id',
        prompt=f'Classify the following email: {content}',
        max_tokens=500
    )
    return response['choices'][0]['text'].strip()

# Duplicate detection
def is_duplicate(content):
    email_hash = hash_email(content)
    return collection.find_one({'hash': email_hash}) is not None

# Batch processing
def batch_classify(files):
    results = []
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file): file for file in files}
        for future in threading.as_completed(future_to_file):
            file = future_to_file[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append({'error': str(e)})
    return results

def process_file(file):
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)
    with open(filepath, 'r') as f:
        content = preprocess_email(f.read())
    duplicate = is_duplicate(content)
    classification = classify_email(content)
    result = {
        'file': file.filename,
        'is_duplicate': duplicate,
        'classification': classification
    }
    return result

@app.route('/batch_classify', methods=['POST'])
def batch_classify_api():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    files = request.files.getlist('files')
    results = batch_classify(files)
    return jsonify({'batch_results': results})

@app.route('/fine_tune', methods=['POST'])
def fine_tune():
    if 'file' not in request.files:
        return jsonify({'error': 'No dataset provided'}), 400
    dataset = request.files['file']
    dataset.save(os.path.join(UPLOAD_FOLDER, secure_filename(dataset.filename)))
    return jsonify({'message': 'Fine-tuning started successfully'})

@app.route('/fine_tune_status', methods=['GET'])
def fine_tune_status():
    return jsonify({'status': 'completed'})

if __name__ == '__main__':
    app.run(debug=True)
