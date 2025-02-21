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
from datasets import load_dataset

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
        
        # ROUGE (tính fmeasure cho các loại ROUGE)
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
            translated = self.translate_text(text, direction)
            metrics = self.calculate_metrics(ref, translated)
            results.append({
                'source': text,
                'translation': translated,
                'reference': ref,
                **metrics
            })
        return pd.DataFrame(results)

    def load_and_prepare_dataset(self, num_samples=1000):
        """
        Tải và chuẩn bị dataset từ HuggingFace: harouzie/vi_en-translation.
        Code này kiểm tra các tên cột (column_names) của dataset và lấy dữ liệu từ:
          - Nếu có key "translation": sử dụng item['translation']['en'] và item['translation']['vi']
          - Nếu không, nếu có key "en" và "vi": sử dụng item['en'] và item['vi']
          - Nếu không, nếu có key "English" và "Vietnamese": sử dụng item['English'] và item['Vietnamese']
        Nếu không tìm thấy, báo lỗi.
        """
        try:
            dataset = load_dataset("harouzie/vi_en-translation", split="train")
        except Exception as e:
            raise ValueError(f"Không thể tải dataset harouzie/vi_en-translation: {e}")
        
        print("Dataset columns:", dataset.column_names)
        dataset = dataset.select(range(num_samples))
        english_texts = []
        vietnamese_texts = []
        
        if 'translation' in dataset.column_names:
            for item in dataset:
                en_text = item['translation'].get('en')
                vi_text = item['translation'].get('vi')
                if en_text and vi_text:
                    english_texts.append(en_text)
                    vietnamese_texts.append(vi_text)
        elif 'en' in dataset.column_names and 'vi' in dataset.column_names:
            for item in dataset:
                en_text = item.get('en')
                vi_text = item.get('vi')
                if en_text and vi_text:
                    english_texts.append(en_text)
                    vietnamese_texts.append(vi_text)
        elif 'English' in dataset.column_names and 'Vietnamese' in dataset.column_names:
            for item in dataset:
                en_text = item.get('English')
                vi_text = item.get('Vietnamese')
                if en_text and vi_text:
                    english_texts.append(en_text)
                    vietnamese_texts.append(vi_text)
        else:
            raise ValueError("Dataset không chứa các key mong đợi: 'translation' hoặc ('en' và 'vi') hoặc ('English' và 'Vietnamese').")
            
        print(f"Số mẫu lấy được: {len(english_texts)}")
        return english_texts, vietnamese_texts

    def evaluate_bidirectional(self, num_samples=1000):
        """Đánh giá cả 2 chiều dịch và lưu lại kết quả"""
        english_texts, vietnamese_texts = self.load_and_prepare_dataset(num_samples)
        
        print("\nEvaluating English to Vietnamese translations...")
        en_vi_results = self.evaluate_translations(english_texts, vietnamese_texts, "en-vi")
        
        print("\nEvaluating Vietnamese to English translations...")
        vi_en_results = self.evaluate_translations(vietnamese_texts, english_texts, "vi-en")
        
        metrics = ['bleu', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'meteor']
        en_vi_summary = {f"en_vi_{m}": en_vi_results[m].mean() for m in metrics}
        vi_en_summary = {f"vi_en_{m}": vi_en_results[m].mean() for m in metrics}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("evaluation_results", exist_ok=True)
        en_vi_results.to_csv(f"evaluation_results/en_vi_results_{timestamp}.csv", index=False)
        vi_en_results.to_csv(f"evaluation_results/vi_en_results_{timestamp}.csv", index=False)
        
        summary = {
            "config": {
                "timestamp": timestamp,
                "num_samples": num_samples,
                "dataset": "harouzie/vi_en-translation"
            },
            "metrics": {**en_vi_summary, **vi_en_summary}
        }
        
        with open(f"evaluation_results/summary_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        return summary

def main():
    print("Bắt đầu đánh giá...")
    evaluator = BiDirectionalTranslationEvaluator()
    
    summary = evaluator.evaluate_bidirectional(num_samples=1000)
    
    print("\nEvaluation Summary:")
    print("\nEnglish to Vietnamese:")
    for metric, value in summary['metrics'].items():
        if metric.startswith('en_vi'):
            print(f"{metric}: {value:.4f}")
            
    print("\nVietnamese to English:")
    for metric, value in summary['metrics'].items():
        if metric.startswith('vi_en'):
            print(f"{metric}: {value:.4f}")
            
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
