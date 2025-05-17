# Duplicate Question Detector

This web application detects if two questions are duplicates of each other using a BERT-based model.

## Features

- Simple web interface with two input boxes for questions
- Real-time duplicate detection
- Confidence score for predictions
- Example questions for demonstration
- Available as both a Flask web app and a Streamlit app
- Command-line interface for quick testing

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the pre-trained model or use your own BERT model

## Usage

### Flask Web App

Run the Flask application:

```bash
python flask_app.py
```

Then open your browser and go to `http://localhost:5000`

### Streamlit App

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser.

### Command Line Interface

You can also use the CLI for quick testing:

```bash
python cli.py --question1 "What are the benefits of drinking green tea?" --question2 "Why should I drink green tea regularly?"
```

## Docker

You can also run the application using Docker:

```bash
# Build the Docker image
docker build -t duplicate-question-detector .

# Run the container
docker run -p 5000:5000 duplicate-question-detector
```

Then open your browser and go to `http://localhost:5000`

## Model

The application uses a fine-tuned BERT model for sequence classification. The model was trained on the Quora Question Pairs dataset to detect whether two questions are duplicates of each other.

## Files

- `app.py`: Streamlit web application
- `flask_app.py`: Flask web application
- `model.py`: Model class for duplicate question detection
- `cli.py`: Command-line interface
- `requirements.txt`: Dependencies
- `Dockerfile`: Docker configuration
- `templates/index.html`: HTML template for the Flask app

## License

MIT
