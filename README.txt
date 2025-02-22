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

This project uses pre-trained MarianMT models from Hugging Face for translation. These models, specifically designed for machine translation, support a wide range of language pairs and are known for their efficiency. The model is automatically downloaded when the application is first run and stored in the Hugging Face transformers cache directory. Subsequent runs will load the model from this local cache.

To select a model for a specific language pair, visit the Helsinki-NLP organization on Hugging Face (https://huggingface.co/Helsinki-NLP). Models are named in the format `opus-mt-<source_language>-<target_language>`. For example, to translate from French ('fr') to German ('de'), use `Helsinki-NLP/opus-mt-fr-de`. Modify the `model_name` variable in the following code to change the model.

To manually download models:

```python
import torch
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-vi"  # Change based on language pair
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# If using GPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

## Folder Structure

```
project-directory/
│-- app.py # Main Flask application
│-- uploads/ # Directory to store uploaded files
│-- templates/
│   └── index.html # HTML template
│-- static/
│   └── styles.css # CSS for styling
│-- evaluation.py # Translation evaluation script
│-- evaluationBi.py # Bi-directional evaluation script
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

- Hugging Face MarianMT: https://huggingface.co/Helsinki-NLP
- Flask Documentation: https://flask.palletsprojects.com/
- PyTorch Installation Guide: https://pytorch.org/get-started/locally/
- SentencePiece Library: https://github.com/google/sentencepiece
- sacreBLEU: https://github.com/mjpost/sacrebleu
- rouge-score: https://github.com/google-research/google-research/tree/master/rouge
- NLTK: https://www.nltk.org/
- Hugging Face Datasets: https://huggingface.co/docs/datasets/
## License

This project is licensed under the MIT License.

