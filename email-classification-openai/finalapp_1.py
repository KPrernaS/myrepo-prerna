import re
import openai
import docx
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
import pymongo
import hashlib
from flask import Flask, request, jsonify
import os
import json
import mimetypes
from werkzeug.utils import secure_filename
import logging
import yaml
from concurrent.futures import ThreadPoolExecutor

# Load configuration from YAML file
CONFIG_FILE = "path_to_config/config.yml"  # Path to the config file
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Config file not found at: {CONFIG_FILE}")

with open(CONFIG_FILE, "r") as file:
    config = yaml.safe_load(file)

# Debug: Print the loaded config
print("Loaded config:", config)

# Ensure the config contains the required keys
if not config or "request_types" not in config:
    raise ValueError("Invalid config file: 'request_types' key is missing or empty.")



# Extract request types and sub-request types from the config
REQUEST_TYPES = config["request_types"]

# Set your OpenAI API key
openai.api_key = "OPEN_API_KEY"


# MongoDB connection
try:
    client = pymongo.MongoClient("localhost:27017")
    db = client["email_classification"]
    collection = db["request_hashes"]
except Exception as e:
    logging.error("Failed to connect to MongoDB: %s", e)
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to reload the configuration
def reload_config():
    """
    Reload the configuration from the config.yml file.
    """
    global config, REQUEST_TYPES
    with open(CONFIG_FILE, "r") as config_file:
        config = yaml.safe_load(config_file)
    REQUEST_TYPES = config["request_types"]
    logger.info("Configuration reloaded successfully.")

# Function to normalize content
def normalize_content(content):
    """
    Normalize the content by removing extra spaces, line breaks, and leading/trailing whitespace.
    """
    # Remove extra spaces and line breaks
    content = re.sub(r"\s+", " ", content).strip()
    return content

# Function to generate a hash for the content
def generate_hash(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# Function to check for duplicates
def is_duplicate(content_hash):
    return collection.find_one({"hash": content_hash}) is not None

# Function to store a hash in MongoDB
def store_hash(content_hash):
    collection.insert_one({"hash": content_hash})

# Function to extract text from DOCX
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error("Failed to extract text from DOCX: %s", e)
        raise

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        return extract_text(file)
    except Exception as e:
        logger.error("Failed to extract text from PDF: %s", e)
        raise

# Function to merge duplicate entries
def merge_duplicate_requests(classifications):
    """
    Merge duplicate entries based on exact matches of request type and sub-request type.
    """
    merged_classifications = {}
    for classification in classifications:
        key = (classification["Request Type"], classification["Sub Request Type"])
        if key in merged_classifications:
            # Merge the required info
            existing_info = merged_classifications[key]["Required Info"]
            new_info = classification["Required Info"]
            # Update existing info with new info (overwrite if keys overlap)
            existing_info.update(new_info)
            # Combine decision words
            existing_words = set(merged_classifications[key]["Decision Words"].split(", "))
            new_words = set(classification["Decision Words"].split(", "))
            merged_classifications[key]["Decision Words"] = ", ".join(existing_words.union(new_words))
            # Keep the higher confidence score
            if classification["Confidence Score"] > merged_classifications[key]["Confidence Score"]:
                merged_classifications[key]["Confidence Score"] = classification["Confidence Score"]
            # Set duplicate flag to True if either entry is a duplicate
            if classification["Duplicate Flag"]:
                merged_classifications[key]["Duplicate Flag"] = True
            # Set priority flag to the higher priority
            if classification["Priority Flag"] == "High":
                merged_classifications[key]["Priority Flag"] = "High"
        else:
            # Add the classification to the merged dictionary
            merged_classifications[key] = classification
    return list(merged_classifications.values())

# Function to generate the prompt dynamically
def generate_prompt(text, request_types):
    """
    Generate the prompt dynamically based on the hierarchical structure of request types and sub-request types.
    """
    request_type_list = []
    for request_type, sub_types in request_types.items():
        if sub_types:
            request_type_list.append(f"{request_type}: {', '.join(sub_types)}")
        else:
            request_type_list.append(request_type)
    request_types_str = "\n".join([f"- {rt}" for rt in request_type_list])

    prompt = f"""
    Analyze the following email content and identify all individual requests within it:
    Email Content: {text}

    For each request, provide the following details:
    1. Request Type: Choose from the following:
    {request_types_str}
    2. Sub Request Type: Choose from the sub-types listed under the selected Request Type.
    3. Confidence Score: A score between 0 and 1 indicating your confidence in the classification.
    4. Decision Words: Key phrases or words from the content that influenced your decision.
    5. Required Info: Extract important information as key-value pairs.
    6. Duplicate Flag: Indicate whether this request is a duplicate (True or False).
    7. Priority Flag: Indicate the priority level (Low, Medium, High). Money Movement – Inbound and Money Movement – Outbound requests should always be given High priority.

    **Important Instructions:**
    - If the email contains repetitive or duplicate requests, combine them into a single entry.
    - Do not generate multiple entries for the same request type and sub-request type.
    - If no valid requests are found, return an empty list.

    Return the output as a JSON array of request classifications. Example:
    [
      {{
        "Request Type": "Money Movement – Outbound",
        "Sub Request Type": "Timebound",
        "Confidence Score": 0.95,
        "Decision Words": "process payment, Account #12345, 2023-10-15",
        "Required Info": {{
          "date": "7 November 2023",
          "account_number": "Account #12345",
          "sender_email": "john.doe@example.com",
          "global_principal_balance": "$1,000,000.00"
        }},
        "Duplicate Flag": false,
        "Priority Flag": "High"
      }}
    ]
    """
    return prompt

# Function to classify text using OpenAI API
def classify_text(text, request_types):
    """
    Use OpenAI to analyze the email content and identify multiple requests directly.
    """
    prompt = generate_prompt(text, request_types)
    
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",  # Use the recommended model
            prompt=prompt,
            max_tokens=500,  # Increase max_tokens to handle larger outputs
            temperature=0.3
        )
        logger.info("OpenAI API Response: %s", response.choices[0].text.strip())
        
        # Parse the response as JSON
        classifications = json.loads(response.choices[0].text.strip())
        if not isinstance(classifications, list):  # Ensure the response is a list
            logger.error("Model response is not a list.")
            return []
        # Enforce priority for Money Movement requests
        classifications = enforce_priority_for_multiple_requests(classifications)
        # Merge duplicate entries
        classifications = merge_duplicate_requests(classifications)
        return classifications
    except json.JSONDecodeError as e:
        logger.error("Failed to parse classification as JSON: %s", e)
        return []
    except Exception as e:
        logger.error("OpenAI API call failed: %s", e)
        return []

def enforce_priority_for_multiple_requests(classifications):
    """
    Enforce priority for Money Movement requests in a list of classifications.
    """
    for classification in classifications:
        if classification.get("Request Type") in ["Money Movement – Inbound", "Money Movement – Outbound"]:
            classification["Priority Flag"] = "High"
    return classifications

# Function to classify a file with auto-detected file type
def classify_file(file):
    try:
        # Get the file name and extension
        filename = secure_filename(file.filename)
        file_type = mimetypes.guess_type(filename)[0]

        # Map MIME types to supported file types
        if file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            file_type = "docx"
        elif file_type == "application/pdf":
            file_type = "pdf"
        elif file_type == "text/plain":
            file_type = "email"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Extract text based on file type
        if file_type == "docx":
            text = extract_text_from_docx(file)
        elif file_type == "pdf":
            text = extract_text_from_pdf(file)
        elif file_type == "email":
            # For emails, assume the file contains the email body as plain text
            text = file.read().decode("utf-8")
        else:
            raise ValueError("Unsupported file type. Use 'docx', 'pdf', or 'email'.")

        # Normalize the content
        text = normalize_content(text)
        logger.info(f"Normalized content: {text}")  # Log normalized content

        # Generate hash for the content
        content_hash = generate_hash(text)
        logger.info(f"Generated hash: {content_hash}")  # Log the hash

        # Check for duplicates
        if is_duplicate(content_hash):
            logger.info("Duplicate detected!")  # Log if duplicate is detected
            return [{
                "Request Type": "Duplicate",
                "Sub Request Type": "Duplicate",
                "Confidence Score": 1.0,
                "Decision Words": "",
                "Required Info": {},
                "Duplicate Flag": True,
                "Priority Flag": "Low"
            }]

        # Store the hash in MongoDB
        store_hash(content_hash)
        logger.info("Hash stored in MongoDB.")  # Confirm hash is stored

        # Classify the email content (including multiple requests)
        classifications = classify_text(text, REQUEST_TYPES)  # Pass REQUEST_TYPES here
        logger.info(f"Classified requests: {classifications}")  # Log classified requests

        return classifications
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise

# Flask API
app = Flask(__name__)

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        classifications = classify_file(file)
        return jsonify(classifications)
    except ValueError as e:
        logger.error(f"Unsupported file type: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/bulk_classify", methods=["POST"])
def bulk_classify():
    """
    Endpoint to classify multiple files (PDFs, DOCXs, or emails) in bulk.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    # Use ThreadPoolExecutor to process files in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(classify_file, file) for file in files]
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing file: {e}")
                results.append({"error": str(e)})

    return jsonify(results)

@app.route("/update_config", methods=["POST"])
def update_config():
    """
    Endpoint to update the config.yml file on the fly.
    """
    try:
        # Get the new configuration from the request body
        new_config = request.json
        if not new_config:
            return jsonify({"error": "No configuration provided"}), 400

        # Validate the new configuration
        if "request_types" not in new_config:
            return jsonify({"error": "Missing 'request_types' in configuration"}), 400

        # Write the new configuration to the config.yml file
        with open(CONFIG_FILE, "w") as config_file:
            yaml.safe_dump(new_config, config_file)

        # Reload the configuration in memory
        reload_config()

        return jsonify({"message": "Configuration updated successfully"})
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7100, debug=True)