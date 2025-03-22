# Email Classifier Project (Full Package)

## Features
- DistilBERT-based multi-label classification
- Predicts Request Type + Sub Request Type
- Dynamic info extraction per request type (configurable)
- MongoDB storage with duplicate detection
- Priority flagging for Money Movement
- Rolling logs with adjustable log levels

## How to Use
1. Run `model.py` to train and predict.
2. Update `config/extraction_config.json` for custom extraction fields.
3. Place trained model in `distilbert-email-classifier/` if pre-trained.
