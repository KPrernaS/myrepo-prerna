# Email Classifier API

## Overview
A robust API for classifying email content and attachments using AI, with support for PDF, DOCX, EML, and plain text files.

## Technologies Used
- Python
- Flask
- OpenAI

## License
MIT

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Multi-format Support:** Process PDFs, Word docs, EML files, and plain text.
- **Bulk Processing:** Classify multiple files in parallel.
- **Duplicate Detection:** Identifies duplicate requests using content hashing.
- **Dynamic Configuration:** Update request types without restarting.
- **Swagger UI:** Interactive API documentation.
- **Priority Handling:** Automatic high-priority flagging for financial requests.

## Quick Start

### Prerequisites
- Python 3.7+
- MongoDB (local or remote)
- OpenAI API key

### Installation

Clone the repository:
```bash
git clone https://github.com/your-repo/email-classifier-api.git
cd email-classifier-api
```

Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create configuration:
```bash
cp config.example.yml config.yml
```
Edit `config.yml` with your request types.

### Running the API
```bash
python app.py
```
The API will be available at `http://localhost:7100` with Swagger UI at `/apidocs`.

## API Documentation

### Endpoints
| Endpoint        | Method | Description                        |
|---------------|--------|--------------------------------|
| /classify      | POST   | Classify a single file         |
| /bulk_classify | POST   | Classify multiple files        |
| /update_config | POST   | Update request type configuration |

### Example Requests

**Classify Single File:**
```bash
curl -X POST -F "file=@email.pdf" http://localhost:7100/classify
```

**Bulk Classify:**
```bash
curl -X POST \
  -F "files=@file1.pdf" \
  -F "files=@file2.docx" \
  http://localhost:7100/bulk_classify
```

**Update Configuration:**
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"request_types": {"Urgent": ["Payment", "Update"]}}' \
  http://localhost:7100/update_config
```

## Configuration

### `config.yml`
```yaml
request_types:
  "Money Movement":
    - "Outbound"
    - "Inbound"
  "Account Maintenance":
    - "Update Details"
    - "Close Account"
```

### Environment Variables
Set these in your environment or `.env` file:
```
OPENAI_API_KEY=your_api_key_here
MONGO_URI=mongodb://localhost:27017
```

## Testing
Run the test suite with:
```bash
pip install -r requirements-test.txt
pytest --cov=.
```
Test coverage includes:
- API endpoints
- File processing
- Classification logic
- Configuration management

## Deployment

### Production Setup

#### Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b :7100 app:app
```

#### Docker:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 7100
CMD ["gunicorn", "--bind", "0.0.0.0:7100", "app:app"]
```

#### NGINX (Sample Config):
```nginx
server {
    listen 80;
    server_name api.example.com;
    
    location / {
        proxy_pass http://localhost:7100;
        proxy_set_header Host $host;
    }
}
```

## Contributing
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Note:** Replace `your-repo` and other placeholder values with your actual project information.

