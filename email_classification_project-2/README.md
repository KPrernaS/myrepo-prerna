
# Email Classification Project

## Features
- Batch classification of emails (EML, PDF, DOCX)
- Duplicate detection using MongoDB
- Fine-tuning with OpenAI GPT
- Dockerized for easy deployment

## Setup
1. Install Docker
2. Build the Docker image:
```
docker build -t email-classifier .
```
3. Run the Docker container:
```
docker run -p 5000:5000 email-classifier
```
4. Use the API endpoints:
- `/batch_classify` for batch processing
- `/fine_tune` for fine-tuning
- `/fine_tune_status` to check fine-tuning status
