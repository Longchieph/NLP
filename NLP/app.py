import os
from flask import Flask, render_template, request, send_file
import requests
import json
from werkzeug.utils import secure_filename
from docx import Document  # Thư viện đọc/ghi file Word
import fitz  # PyMuPDF: Đọc/ghi file PDF

app = Flask(__name__)

# Thư mục lưu file tạm thời
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Danh sách ngôn ngữ hỗ trợ
LANGUAGES = [
    {"code": "en", "name": "English"},
    {"code": "vi", "name": "Vietnamese"},
    {"code": "fr", "name": "French"},
    {"code": "es", "name": "Spanish"},
    {"code": "de", "name": "German"},
    {"code": "zh-CN", "name": "Chinese (Simplified)"},
    {"code": "zh-TW", "name": "Chinese (Traditional)"},
    {"code": "ja", "name": "Japanese"},
    {"code": "ko", "name": "Korean"},
    {"code": "ar", "name": "Arabic"},
    {"code": "ru", "name": "Russian"},
    {"code": "it", "name": "Italian"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "nl", "name": "Dutch"},
    {"code": "tr", "name": "Turkish"},
    {"code": "pl", "name": "Polish"},
    {"code": "th", "name": "Thai"},
    {"code": "id", "name": "Indonesian"},
    {"code": "ms", "name": "Malay"},
    {"code": "hi", "name": "Hindi"},
    {"code": "bn", "name": "Bengali"},
    {"code": "ur", "name": "Urdu"},
    {"code": "he", "name": "Hebrew"},
    {"code": "sv", "name": "Swedish"},
    {"code": "no", "name": "Norwegian"},
    {"code": "da", "name": "Danish"},
    {"code": "fi", "name": "Finnish"},
    {"code": "cs", "name": "Czech"},
    {"code": "ro", "name": "Romanian"},
    {"code": "el", "name": "Greek"},
    {"code": "hu", "name": "Hungarian"},
    {"code": "sk", "name": "Slovak"},
    {"code": "bg", "name": "Bulgarian"},
    {"code": "uk", "name": "Ukrainian"},
    {"code": "sr", "name": "Serbian"},
    {"code": "hr", "name": "Croatian"},
    {"code": "sl", "name": "Slovenian"},
    {"code": "et", "name": "Estonian"},
    {"code": "lv", "name": "Latvian"},
    {"code": "lt", "name": "Lithuanian"}
]

# Hàm dịch văn bản
def translate_text(text, src_lang, dest_lang):
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src_lang}&tl={dest_lang}&dt=t&q={text}"
    response = requests.get(url)
    try:
        data = json.loads(response.text)
        return data[0][0][0]
    except (json.JSONDecodeError, IndexError, KeyError):
        return "Translation failed."

# Hàm xử lý file Word
def translate_word_file(filepath, src_lang, dest_lang):
    doc = Document(filepath)
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            paragraph.text = translate_text(paragraph.text, src_lang, dest_lang)
    translated_path = os.path.join(app.config["UPLOAD_FOLDER"], "translated_" + os.path.basename(filepath))
    doc.save(translated_path)
    return translated_path

# Hàm xử lý file PDF
def translate_pdf_file(filepath, src_lang, dest_lang):
    pdf_document = fitz.open(filepath)
    translated_path = os.path.join(app.config["UPLOAD_FOLDER"], "translated_" + os.path.basename(filepath))
    translated_pdf = fitz.open()
    for page in pdf_document:
        text = page.get_text()
        if text.strip():
            translated_text = translate_text(text, src_lang, dest_lang)
            translated_page = translated_pdf.new_page(width=page.rect.width, height=page.rect.height)
            translated_page.insert_text(page.rect.topleft, translated_text, fontsize=12)
    translated_pdf.save(translated_path)
    pdf_document.close()
    return translated_path

@app.route('/', methods=['GET', 'POST'])
def translate():
    translated_text = None

    if request.method == 'POST':
        if 'text' in request.form:  # Trường hợp dịch văn bản
            text = request.form['text']
            src_lang = request.form['src_lang']
            dest_lang = request.form['dest_lang']
            translated_text = translate_text(text, src_lang, dest_lang)

        if 'file' in request.files:  # Trường hợp dịch file
            file = request.files['file']
            if file:
                src_lang = request.form['src_lang']
                dest_lang = request.form['dest_lang']
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)

                # Xử lý dịch file theo định dạng
                if filename.endswith(".docx"):
                    translated_file = translate_word_file(filepath, src_lang, dest_lang)
                elif filename.endswith(".pdf"):
                    translated_file = translate_pdf_file(filepath, src_lang, dest_lang)
                else:
                    return "Unsupported file format. Please upload a .docx or .pdf file."

                return send_file(translated_file, as_attachment=True)

    return render_template('index.html', languages=LANGUAGES, translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)