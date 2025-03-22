# Email Classification API

## Overview
This project provides an **email classification API** using a **Hugging Face zero-shot classification model**. It accepts input in various formats (PDF, DOCX, EML, MSG) and classifies emails based on predefined request types stored in a YAML configuration file.

## Features
- Supports **single and bulk classification**.
- Accepts **PDF, DOCX, EML, and MSG** as input.
- Uses **zero-shot classification** for request type identification.
- Stores **request types and required info** in a configurable YAML file.
- Implements **duplicate detection** using MongoDB.
- Supports **dynamic updates** to request types.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- MongoDB (for duplicate detection)
- Pip and Virtualenv

### Setup Instructions
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd email-classification-api
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up MongoDB and update `config.yml` with connection details.
5. Start the API:
   ```sh
   python app.py
   ```

## API Endpoints
### 1. **Classify a Single Email**
**Endpoint:** `POST /classify`
- **Input:** Raw text or file (PDF, DOCX, EML, MSG)
- **Output:** JSON with classified request type, sub-request type, and confidence score.

Example:
```json
{
  "request_types": [
    {"type": "Money Movement – Inbound", "subtypes": ["Principal"], "confidence_score": 0.94}
  ],
  "is_duplicate": "No",
  "priority": "High"
}
```

### 2. **Bulk Classification**
**Endpoint:** `POST /bulk_classify`
- Accepts multiple files for parallel classification.

### 3. **Update Request Types**
**Endpoint:** `POST /update_request_types`
- Updates request types dynamically from `config.yml`.

### 4. **Fine-Tuning Endpoints** (Future Scope)
- `/fine_tune`
- `/fine_tune_status`

## Configuration
Modify `config.yml` to update request types and required info mappings.

## Deployment
- Can be run locally or deployed to a cloud service.
- For Docker support, create a `Dockerfile` and use:
  ```sh
  docker build -t email-classifier .
  docker run -p 5000:5000 email-classifier
  ```

## Future Enhancements
- Implement fine-tuning for improved accuracy.
- Add support for multi-threaded processing.

---
For any issues, feel free to reach out!
