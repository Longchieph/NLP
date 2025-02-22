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
        fine_tuned_en_vi = r"D:\Year3\NLP\NLP\NLP\results\fine-tuned-en-vi"
        fine_tuned_vi_en = r"D:\Year3\NLP\NLP\NLP\results\fine-tuned-vi-en"

        self.en_vi_tokenizer = MarianTokenizer.from_pretrained(fine_tuned_en_vi)
        self.en_vi_model = MarianMTModel.from_pretrained(fine_tuned_en_vi)

        self.vi_en_tokenizer = MarianTokenizer.from_pretrained(fine_tuned_vi_en)
        self.vi_en_model = MarianMTModel.from_pretrained(fine_tuned_vi_en)

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)

    def load_and_prepare_dataset(self, num_samples=1000):
        """Tải và chuẩn bị dataset từ Hugging Face"""
        try:
            dataset = load_dataset("harouzie/vi_en-translation")
            df = pd.DataFrame(dataset['test'])

            # Kiểm tra cột và đổi tên cho đúng
            print("Columns in dataset:", df.columns.tolist())
            if 'English' in df.columns and 'Vietnamese' in df.columns:
                df.rename(columns={'English': 'en', 'Vietnamese': 'vi'}, inplace=True)
            elif 'en' not in df.columns or 'vi' not in df.columns:
                raise ValueError("Dataset không có cột 'English' hoặc 'Vietnamese'.")

            df = df.head(num_samples)
            english_texts = df['en'].astype(str).tolist()
            vietnamese_texts = df['vi'].astype(str).tolist()
            return english_texts, vietnamese_texts
        except Exception as e:
            print("Lỗi khi tải dataset:", str(e))
            return [], []

    def translate_text(self, text, direction="en-vi"):
        """Dịch một câu văn bản"""
        if direction == "en-vi":
            model = self.en_vi_model
            tokenizer = self.en_vi_tokenizer
        else:
            model = self.vi_en_model
            tokenizer = self.vi_en_tokenizer
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)

    def calculate_metrics(self, reference, hypothesis):
        """Tính toán BLEU, ROUGE, METEOR"""
        ref_tokens = nltk.word_tokenize(reference.lower())
        hyp_tokens = nltk.word_tokenize(hypothesis.lower())
        
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score
        rouge_scores = self.rouge_scorer.score(reference, hypothesis)
        
        rouge_dict = {
            'rouge1_f': rouge_scores['rouge1'].fmeasure,
            'rouge2_f': rouge_scores['rouge2'].fmeasure,
            'rougeL_f': rouge_scores['rougeL'].fmeasure
        }
        meteor = meteor_score([ref_tokens], hyp_tokens)
        
        return {'bleu': bleu, **rouge_dict, 'meteor': meteor, 'rouge_details': rouge_scores}

    def evaluate_translations(self, texts, references, direction):
        """Đánh giá bản dịch"""
        results = []
        for text, ref in tqdm(zip(texts, references), total=len(texts), desc=f"Evaluating {direction} translations"):
            translated = self.translate_text(text, direction)
            metrics = self.calculate_metrics(ref, translated)
            results.append({'source': text, 'translation': translated, 'reference': ref, **metrics})
        return pd.DataFrame(results)

    def evaluate_bidirectional(self, num_samples=1000):
        """Đánh giá cả 2 chiều dịch"""
        english_texts, vietnamese_texts = self.load_and_prepare_dataset(num_samples)
        if not english_texts or not vietnamese_texts:
            print("Dataset bị lỗi hoặc rỗng! Dừng quá trình đánh giá.")
            return None

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

        summary = {"config": {"timestamp": timestamp, "num_samples": num_samples, "dataset": "harouzie/vi_en-translation"},
                   "metrics": {**en_vi_summary, **vi_en_summary}}

        with open(f"evaluation_results/summary_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

def main():
    print("Bắt đầu đánh giá...")
    evaluator = BiDirectionalTranslationEvaluator()
    summary = evaluator.evaluate_bidirectional(num_samples=1000)

    if summary:
        print("\nEvaluation Summary:")
        for metric, value in summary['metrics'].items():
            print(f"{metric}: {value:.4f}")

        english_texts, vietnamese_texts = evaluator.load_and_prepare_dataset(num_samples=1000)
        sample_en = english_texts[0]
        sample_vi = vietnamese_texts[0]
        sample_translation = evaluator.translate_text(sample_en, "en-vi")
        sample_metrics = evaluator.calculate_metrics(sample_vi, sample_translation)

        print("\nDetailed Metrics for a sample sentence:")
        print("ROUGE Scores:", sample_metrics.get('rouge_details'))
        print("METEOR Score:", sample_metrics['meteor'])

if __name__ == "__main__":
    main()
