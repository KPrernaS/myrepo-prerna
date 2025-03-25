from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
import logging
from config_manager import ConfigManager
from database import DatabaseManager
from file_handlers import FileHandler
from classifiers import Classifier
from utils import FileUtils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components
config = ConfigManager()
db = DatabaseManager()
file_handler = FileHandler()
classifier = Classifier(api_key="openaikey")

def process_file(file):
    try:
        file_type = FileUtils.get_file_type(file.filename)
        
        # Extract text based on file type
        if file_type == "docx":
            text = file_handler.extract_text_from_docx(file)
        elif file_type == "pdf":
            text = file_handler.extract_text_from_pdf(file)
        elif file_type == "eml":
            text = file_handler.extract_text_from_eml(file)
        else:  # text
            text = file_handler.extract_text_from_text(file)
        
        text = classifier.normalize_content(text)
        content_hash = db.generate_hash(text)
        
        if db.is_duplicate(content_hash):
            return [{
                "Request Type": "Duplicate",
                "Sub Request Type": "Duplicate",
                "Confidence Score": 1.0,
                "Decision Words": "",
                "Required Info": {},
                "Duplicate Flag": True,
                "Priority Flag": "Low"
            }]
        
        db.store_hash(content_hash)
        classifications = classifier.classify_text(text, config.request_types)
        classifications = classifier.enforce_priority(classifications)
        return classifier.merge_duplicate_requests(classifications)
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        classifications = process_file(request.files["file"])
        return jsonify(classifications)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500

@app.route("/bulk_classify", methods=["POST"])
def bulk_classify():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in files]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"error": str(e)})

    return jsonify(results)

@app.route("/update_config", methods=["POST"])
def update_config():
    try:
        new_config = request.json
        if not new_config:
            return jsonify({"error": "No configuration provided"}), 400

        config.update_config(new_config)
        return jsonify({"message": "Configuration updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7100, debug=True)