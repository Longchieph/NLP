import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
from nltk.tokenize import word_tokenize

# Tải dữ liệu cần thiết
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')  # Cần để dùng word_tokenize
nltk.download('averaged_perceptron_tagger')  # Bổ sung để hỗ trợ tokenization tốt hơn
nltk.download('universal_tagset')  # Bổ sung nếu cần
# Câu gốc từ người dùng nhập trên web
reference = "I have a pen, I have an apple"  

# Câu đã được dịch từ ứng dụng web
translated = "Tôi có một cây bút, tôi có một quả táo"  

# 🔹 Tiền xử lý: Tokenization chính xác hơn
ref_tokens = word_tokenize(reference.lower())  # Chuyển chữ thường + Tokenization
trans_tokens = word_tokenize(translated.lower())  

# ✅ 1️⃣ Tính BLEU Score
bleu = sacrebleu.sentence_bleu(" ".join(trans_tokens), [" ".join(ref_tokens)])
print("BLEU Score:", bleu.score)

# ✅ 2️⃣ Tính ROUGE Score
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = scorer.score(" ".join(ref_tokens), " ".join(trans_tokens))
print("ROUGE Scores:", rouge_scores)

# ✅ 3️⃣ Tính METEOR Score (Fix lỗi bằng tokenized input)
meteor = meteor_score([ref_tokens], trans_tokens)
print("METEOR Score:", meteor)
