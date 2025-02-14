import os
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from transformers import MarianMTModel, MarianTokenizer
from docx import Document
import fitz  # PyMuPDF

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load MarianMT model for translation
def load_model(src_lang, dest_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_word_file(filepath, model, tokenizer):
    doc = Document(filepath)
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            paragraph.text = translate_text(paragraph.text, model, tokenizer)
    translated_path = os.path.join(app.config["UPLOAD_FOLDER"], "translated_" + os.path.basename(filepath))
    doc.save(translated_path)
    return translated_path

def translate_pdf_file(filepath, model, tokenizer):
    pdf_document = fitz.open(filepath)
    translated_path = os.path.join(app.config["UPLOAD_FOLDER"], "translated_" + os.path.basename(filepath))
    translated_pdf = fitz.open()
    
    for page in pdf_document:
        text = page.get_text()
        if text.strip():
            translated_text = translate_text(text, model, tokenizer)
            translated_page = translated_pdf.new_page(width=page.rect.width, height=page.rect.height)
            translated_page.insert_text(page.rect.topleft, translated_text, fontsize=12)
    
    translated_pdf.save(translated_path)
    pdf_document.close()
    return translated_path

@app.route('/', methods=['GET', 'POST'])
def translate():
    translated_text = None
    languages = [
        {"code": "en", "name": "English"},
        {"code": "vi", "name": "Vietnamese"},
        # Bạn có thể thêm các ngôn ngữ khác ở đây
    ]
    
    if request.method == 'POST':
        src_lang = request.form['src_lang']
        dest_lang = request.form['dest_lang']
        model, tokenizer = load_model(src_lang, dest_lang)
        
        if 'text' in request.form:  # Translate text
            text = request.form['text']
            translated_text = translate_text(text, model, tokenizer)
        
        if 'file' in request.files:
            file = request.files['file']
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                
                if filename.endswith(".docx"):
                    translated_file = translate_word_file(filepath, model, tokenizer)
                elif filename.endswith(".pdf"):
                    translated_file = translate_pdf_file(filepath, model, tokenizer)
                else:
                    return "Unsupported file format. Please upload a .docx or .pdf file."
                
                return send_file(translated_file, as_attachment=True)
    
    return render_template('index.html', translated_text=translated_text, languages=languages)

if __name__ == '__main__':
    app.run(debug=True)
