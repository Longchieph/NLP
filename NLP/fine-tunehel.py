import os
import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import evaluate

# 1. Cài đặt tên model và tokenizer
model_name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 2. Load bộ dữ liệu từ Hugging Face (thainq107/iwslt2015-en-vi)
dataset = load_dataset("thainq107/iwslt2015-en-vi")

# 3. Thiết lập một số tham số cho việc token hóa
max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    if "en" not in examples or "vi" not in examples:
        return {}
    
    inputs = [text if text is not None else "" for text in examples["en"]]
    targets = [text if text is not None else "" for text in examples["vi"]]
    
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


# 4. Tokenize toàn bộ dataset
# Lưu ý: batched=True để tăng tốc, bạn có thể chỉnh num_proc nếu muốn tận dụng đa luồng
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 5. Định nghĩa metric sacreBLEU để đánh giá
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    # Loại bỏ khoảng trắng thừa ở đầu/cuối
    preds = [pred.strip() for pred in preds]
    # sacreBLEU yêu cầu tham chiếu là danh sách danh sách
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    """
    Hàm tính BLEU:
    - decode các token dự đoán và nhãn.
    - post-process để đưa về format sacreBLEU yêu cầu.
    - tính BLEU và trả về kết quả.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Giải mã token thành chuỗi
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Xử lý token -100 thành pad_token_id
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Chuẩn hóa chuỗi trước khi tính BLEU
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    return {"bleu": result["score"]}

# 6. Cấu hình các tham số huấn luyện
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Đánh giá theo epoch thay vì từng step nhỏ
    save_strategy="epoch",        # Lưu model sau mỗi epoch
    learning_rate=3e-5,           # Có thể tăng nhẹ learning rate nếu cần
    per_device_train_batch_size=32,  # Tăng batch size nếu GPU có VRAM đủ
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Giúp mô phỏng batch lớn hơn nếu bộ nhớ bị giới hạn
    weight_decay=0.01,
    num_train_epochs=5,  # Huấn luyện lâu hơn một chút
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=50,  # Ghi log thường xuyên hơn
)


# 7. Tạo data collator để tự động padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 8. Tách dataset cho train/validation (nếu có)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# 9. Khởi tạo Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# 10. Thực hiện fine-tuning
trainer.train(resume_from_checkpoint="./results/checkpoint-17000")

# 11. Lưu model sau khi train xong
trainer.save_model("./results/fine-tuned-en-vi")
