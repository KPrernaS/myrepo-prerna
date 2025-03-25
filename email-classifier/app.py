from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
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

# Configure Swagger
app.config['SWAGGER'] = {
    'title': 'Email Classifier API',
    'uiversion': 3,
    'description': 'API for classifying email content and attachments',
    'version': '1.0.0'
}
swagger = Swagger(app)

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


@app.route('/classify', methods=['POST'])
@swag_from({
    'tags': ['Classification'],
    'description': 'Classify a single file (PDF, DOCX, EML, or plain text)',
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'File to classify (PDF, DOCX, EML, or plain text)'
        }
    ],
    'responses': {
        200: {
            'description': 'Successful classification',
            'examples': {
                'application/json': [
                    {
                        "Request Type": "Money Movement – Outbound",
                        "Sub Request Type": "Timebound",
                        "Confidence Score": 0.95,
                        "Decision Words": "process payment, Account #12345",
                        "Required Info": {
                            "date": "7 November 2023",
                            "account_number": "Account #12345"
                        },
                        "Duplicate Flag": False,
                        "Priority Flag": "High"
                    }
                ]
            }
        },
        400: {
            'description': 'Bad request (missing file or invalid file type)'
        },
        500: {
            'description': 'Internal server error'
        }
    }
})
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

@app.route('/bulk_classify', methods=['POST'])
@swag_from({
    'tags': ['Classification'],
    'description': 'Classify multiple files in bulk',
    'consumes': ['multipart/form-data'],
    'parameters': [
         {
            'name': 'files',
            'in': 'formData',
            'type': 'array',
            'items': {
                'type': 'file',
                'description': 'File to upload'
            },
            'required': True,
            'description': 'List of files to classify'
        }
    ],
    'responses': {
        200: {
            'description': 'Successful classification of all files',
            'examples': {
                'application/json': [
                    [
                        {
                            "Request Type": "Account Maintenance",
                            "Sub Request Type": "Update Details",
                            "Confidence Score": 0.85,
                            "Decision Words": "change address, update contact",
                            "Required Info": {
                                "new_address": "123 Main St"
                            },
                            "Duplicate Flag": False,
                            "Priority Flag": "Medium"
                        }
                    ],
                    [
                        {
                            "Request Type": "Duplicate",
                            "Sub Request Type": "Duplicate",
                            "Confidence Score": 1.0,
                            "Decision Words": "",
                            "Required Info": {},
                            "Duplicate Flag": True,
                            "Priority Flag": "Low"
                        }
                    ]
                ]
            }
        },
        400: {
            'description': 'Bad request (missing files)'
        },
        500: {
            'description': 'Internal server error'
        }
    }
})
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

@app.route('/update_config', methods=['POST'])
@swag_from({
    'tags': ['Configuration'],
    'description': 'Update the classification configuration',
    'consumes': ['application/json'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'request_types': {
                        'type': 'object',
                        'description': 'Hierarchy of request types and sub-types',
                        'example': {
                            "Money Movement – Outbound": ["Timebound", "Recurring"],
                            "Account Maintenance": ["Update Details", "Close Account"]
                        }
                    }
                },
                'required': ['request_types']
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Configuration updated successfully',
            'examples': {
                'application/json': {
                    "message": "Configuration updated successfully"
                }
            }
        },
        400: {
            'description': 'Bad request (invalid configuration)'
        },
        500: {
            'description': 'Internal server error'
        }
    }
})
def update_config():
    try:
        new_config = request.json
        if not new_config:
            return jsonify({"error": "No configuration provided"}), 400

        config.update_config(new_config)
        return jsonify({"message": "Configuration updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return """
    <h1>Email Classifier API</h1>
    <p>Visit <a href="/apidocs">/apidocs</a> for Swagger documentation</p>
    """
# Add this Swagger definition at the app level (not inside a route)
app.config['SWAGGER']['template'] = {
    'swagger': '2.0',
    'info': {
        'title': 'Email Classifier API',
        'description': 'API for classifying email content and attachments',
        'version': '1.0.0'
    },
    'definitions': {
        'ClassificationResult': {
            'type': 'object',
            'properties': {
                'Request Type': {'type': 'string'},
                'Sub Request Type': {'type': 'string'},
                'Confidence Score': {'type': 'number', 'format': 'float'},
                'Decision Words': {'type': 'string'},
                'Required Info': {'type': 'object'},
                'Duplicate Flag': {'type': 'boolean'},
                'Priority Flag': {'type': 'string', 'enum': ['Low', 'Medium', 'High']}
            }
        }
    }
}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7100, debug=True)