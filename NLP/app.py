import os
import torch
from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from transformers import MarianMTModel, MarianTokenizer
from docx import Document
import fitz  # PyMuPDF

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Lưu cache để tránh load model nhiều lần
models_cache = {}

def get_model(src_lang, dest_lang):
    """Tải model dịch với caching để tránh load lại nhiều lần."""
    key = f"{src_lang}-{dest_lang}"
    if key not in models_cache:
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{dest_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        models_cache[key] = (model, tokenizer)
    return models_cache[key]

def translate_text(text, model, tokenizer):
    """Dịch văn bản sử dụng mô hình MarianMT."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():  # Tăng tốc dịch
        translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_word_file(filepath, model, tokenizer):
    """Dịch nội dung file Word (.docx)."""
    doc = Document(filepath)
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            paragraph.text = translate_text(paragraph.text, model, tokenizer)
    translated_path = os.path.join(app.config["UPLOAD_FOLDER"], "translated_" + os.path.basename(filepath))
    doc.save(translated_path)
    return translated_path

def translate_pdf_file(filepath, model, tokenizer):
    """Dịch nội dung file PDF."""
    pdf_document = fitz.open(filepath)
    translated_pdf = fitz.open()
    translated_path = os.path.join(app.config["UPLOAD_FOLDER"], "translated_" + os.path.basename(filepath))

    for page in pdf_document:
        text = page.get_text()
        if text.strip():
            translated_text = translate_text(text, model, tokenizer)
            translated_page = translated_pdf.new_page(width=page.rect.width, height=page.rect.height)
            translated_page.insert_text(page.rect.topleft, translated_text, fontsize=12)
    
    translated_pdf.save(translated_path)
    pdf_document.close()
    return translated_path

# Kiểm tra định dạng file hợp lệ
ALLOWED_EXTENSIONS = {'docx', 'pdf'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def translate():
    translated_text = None
    error_msg = None
    translated_file = None
    languages = [
        {"code": "en", "name": "English"},
        {"code": "vi", "name": "Vietnamese"},
    ]
    
    if request.method == 'POST':
        src_lang = request.form.get('src_lang')
        dest_lang = request.form.get('dest_lang')

        if not src_lang or not dest_lang:
            error_msg = "Please select both source and target languages."
        else:
            model, tokenizer = get_model(src_lang, dest_lang)

            # Dịch văn bản
            if 'text' in request.form and request.form['text'].strip():
                text = request.form['text']
                translated_text = translate_text(text, model, tokenizer)

            # Dịch tài liệu
            if 'file' in request.files:
                file = request.files['file']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file.save(filepath)

                    if filename.endswith(".docx"):
                        translated_file = translate_word_file(filepath, model, tokenizer)
                    elif filename.endswith(".pdf"):
                        translated_file = translate_pdf_file(filepath, model, tokenizer)
                    return redirect(url_for('download_translated_file', filename=os.path.basename(translated_file)))
                else:
                    error_msg = "Unsupported file format. Please upload a .docx or .pdf file."

    return render_template('index.html', translated_text=translated_text, translated_file=translated_file, languages=languages, error_msg=error_msg)

@app.route('/download/<filename>')
def download_translated_file(filename):
    """Tải xuống file dịch sau khi xử lý."""
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
