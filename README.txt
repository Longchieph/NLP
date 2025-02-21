# Text and Document Translator

## Overview

This is a Flask-based web application that allows users to translate text and documents (DOCX, PDF) between different languages using the MarianMT translation model from Hugging Face. The system supports multiple languages and can process both text input and file uploads.

## Features

- Translate text between different languages.
- Upload and translate DOCX and PDF documents.
- Download translated files.
- Supports various language pairs using the Helsinki-NLP MarianMT models.

## Installation & Setup

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

### Install Dependencies

1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <project_directory>
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv  # Create virtual environment
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```
3. Install required Python libraries:
   ```sh
   pip install -r requirements.txt
   ```

### Required Packages

The project relies on the following libraries:

```sh
pip install flask transformers torch sentencepiece werkzeug python-docx pymupdf
```

## Running the Application

1. Ensure all dependencies are installed.
2. Run the Flask application:
   ```sh
   python app.py
   ```
3. Open a web browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## Model and Data Setup

This project uses **Helsinki-NLP/opus-mt** models from Hugging Face for translation. The model is automatically downloaded when the application is first run.

To manually download models:

```sh
from transformers import MarianMTModel, MarianTokenizer
model_name = "Helsinki-NLP/opus-mt-en-vi"  # Change based on language pair
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)
```

## Folder Structure

```
project-directory/
│-- app.py                   # Main Flask application
│-- requirements.txt         # Dependencies list
│-- uploads/                 # Directory for uploaded files
│-- templates/
│   └── index.html           # HTML template
│-- static/
│   └── styles.css           # CSS for styling
```

## Usage

1. **Translate Text**
   - Enter text in the text box.
   - Select source and target languages.
   - Click "Translate" to get the result.
2. **Translate Documents**
   - Upload a `.docx` or `.pdf` file.
   - Select source and target languages.
   - Click "Translate" and download the translated file.

## References

- Hugging Face MarianMT: [https://huggingface.co/Helsinki-NLP](https://huggingface.co/Helsinki-NLP)
- Flask Documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
- PyTorch Installation Guide: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- SentencePiece Library: [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

## License

This project is licensed under the MIT License.

