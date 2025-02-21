import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import MarianMTModel, MarianTokenizer
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
import json
from datetime import datetime
import os

class BiDirectionalTranslationEvaluator:
    def __init__(self):
        """Khởi tạo evaluator cho cả 2 chiều dịch"""
        # Khởi tạo cho chiều Anh-Việt
        self.en_vi_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        self.en_vi_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        
        # Khởi tạo cho chiều Việt-Anh
        self.vi_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
        self.vi_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
        
        # Khởi tạo ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Đảm bảo NLTK data được tải
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        
    def translate_text(self, text, direction="en-vi"):
        """Dịch một câu văn bản theo hướng chỉ định"""
        if direction == "en-vi":
            model = self.en_vi_model
            tokenizer = self.en_vi_tokenizer
        else:  # vi-en
            model = self.vi_en_model
            tokenizer = self.vi_en_tokenizer
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def calculate_metrics(self, reference, hypothesis):
        """Tính toán các metric đánh giá cho một cặp câu và trả về chi tiết ROUGE cũng như METEOR"""
        # Tokenize
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        # BLEU
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score
        
        # ROUGE (trả về chi tiết gồm precision, recall, fmeasure)
        rouge_scores = self.rouge_scorer.score(reference, hypothesis)
        rouge_dict = {
            'rouge1_f': rouge_scores['rouge1'].fmeasure,
            'rouge2_f': rouge_scores['rouge2'].fmeasure,
            'rougeL_f': rouge_scores['rougeL'].fmeasure
        }
        
        # METEOR
        meteor = meteor_score([ref_tokens], hyp_tokens)
        
        return {
            'bleu': bleu,
            **rouge_dict,
            'meteor': meteor,
            'rouge_details': rouge_scores  # chi tiết ROUGE
        }

    def evaluate_translations(self, texts, references, direction):
        """Đánh giá dịch cho một batch văn bản"""
        results = []
        
        for text, ref in tqdm(zip(texts, references), total=len(texts), 
                                desc=f"Evaluating {direction} translations"):
            # Dịch văn bản
            translated = self.translate_text(text, direction)
            
            # Tính metric
            metrics = self.calculate_metrics(ref, translated)
            
            results.append({
                'source': text,
                'translation': translated,
                'reference': ref,
                **metrics
            })
            
        return pd.DataFrame(results)

    def load_and_prepare_dataset(self, num_samples=1000):
        """Tải và chuẩn bị dataset từ file CSV Kaggle (English-Vietnamese)
           Giả sử tệp CSV có 2 cột: cột thứ nhất là tiếng Anh, cột thứ hai là tiếng Việt.
        """
        try:
            # Đọc dữ liệu từ file CSV với encoding 'utf-16'
            dataset_path = r"D:\English - Vietnamese.csv\English - Vietnamese.csv"
            kaggle_df = pd.read_csv(dataset_path, encoding='utf-16', on_bad_lines='skip')
            
            # In ra tên các cột để kiểm tra (chỉ dùng cho debug)
            print("Columns in CSV:", kaggle_df.columns.tolist())
            
            # Lấy 1000 mẫu đầu tiên
            kaggle_df = kaggle_df.head(num_samples)
            
            # Giả sử rằng cột đầu tiên chứa tiếng Anh và cột thứ hai chứa tiếng Việt.
            english_texts = kaggle_df.iloc[:, 0].astype(str).tolist()
            vietnamese_texts = kaggle_df.iloc[:, 1].astype(str).tolist()
            
            return english_texts, vietnamese_texts
            
        except FileNotFoundError:
            raise FileNotFoundError(
                "Không tìm thấy file dataset. Vui lòng kiểm tra lại đường dẫn tới file CSV."
            )

    def evaluate_bidirectional(self, num_samples=1000):
        """Đánh giá cả 2 chiều dịch và lưu lại kết quả"""
        # Tải dataset
        english_texts, vietnamese_texts = self.load_and_prepare_dataset(num_samples)
        
        # Đánh giá Anh-Việt
        print("\nEvaluating English to Vietnamese translations...")
        en_vi_results = self.evaluate_translations(
            english_texts, vietnamese_texts, "en-vi"
        )
        
        # Đánh giá Việt-Anh
        print("\nEvaluating Vietnamese to English translations...")
        vi_en_results = self.evaluate_translations(
            vietnamese_texts, english_texts, "vi-en"
        )
        
        # Tính trung bình các metric (chỉ tính giá trị đã tính toán, không bao gồm chi tiết ROUGE)
        metrics = ['bleu', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'meteor']
        en_vi_summary = {f"en_vi_{m}": en_vi_results[m].mean() for m in metrics}
        vi_en_summary = {f"vi_en_{m}": vi_en_results[m].mean() for m in metrics}
        
        # Lưu kết quả
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("evaluation_results", exist_ok=True)
        
        en_vi_results.to_csv(f"evaluation_results/en_vi_results_{timestamp}.csv", index=False)
        vi_en_results.to_csv(f"evaluation_results/vi_en_results_{timestamp}.csv", index=False)
        
        summary = {
            "config": {
                "timestamp": timestamp,
                "num_samples": num_samples,
                "dataset": "English-Vietnamese Kaggle"
            },
            "metrics": {**en_vi_summary, **vi_en_summary}
        }
        
        with open(f"evaluation_results/summary_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        return summary

def main():
    print("Bắt đầu đánh giá...")
    # Khởi tạo evaluator
    evaluator = BiDirectionalTranslationEvaluator()
    
    # Thực hiện đánh giá tổng quát
    summary = evaluator.evaluate_bidirectional(num_samples=1000)
    
    # In kết quả tổng hợp
    print("\nEvaluation Summary:")
    print("\nEnglish to Vietnamese:")
    for metric, value in summary['metrics'].items():
        if metric.startswith('en_vi'):
            print(f"{metric}: {value:.4f}")
            
    print("\nVietnamese to English:")
    for metric, value in summary['metrics'].items():
        if metric.startswith('vi_en'):
            print(f"{metric}: {value:.4f}")
            
    # In kết quả chi tiết cho một cặp câu mẫu
    print("\nDetailed Metrics for a sample sentence:")
    english_texts, vietnamese_texts = evaluator.load_and_prepare_dataset(num_samples=1000)
    sample_en = english_texts[0]
    sample_vi = vietnamese_texts[0]
    sample_translation = evaluator.translate_text(sample_en, "en-vi")
    sample_metrics = evaluator.calculate_metrics(sample_vi, sample_translation)
    print("ROUGE Scores:", sample_metrics.get('rouge_details'))
    print("METEOR Score:", sample_metrics['meteor'])

if __name__ == "__main__":
    main()
