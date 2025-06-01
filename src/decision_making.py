# decision_making.py

import os
import csv
import torch
import pandas as pd
from datetime import datetime
from transformers import BertTokenizerFast
from model import SentimentClassifier  # Ваш wrapper-класс для BERT

# Path к сохранённому чекпоинту
CHECKPOINT_PATH = 'checkpoints/best_model_bert3cls.pth'
# Словарь idx→label
IDX2LABEL = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# 1) Загружаем модель и токенизатор (единожды, при старте)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', local_files_only=True)

# Инициализируем SentimentClassifier и грузим веса
model = SentimentClassifier(pretrained_model_name='bert-base-uncased', num_labels=3)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
# Если вы сохраняли state_dict bert_model, а не весь wrapper:
model.bert_model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# 2) Функция для inference и принятия решения
def make_decision(text: str) -> tuple[str, str]:
    """
    На вход даётся raw текст (str). Функция возвращает кортеж (label, action):
    - label ∈ {'Negative','Neutral','Positive'}
    - action ∈ {'ALERT','MONITOR','NO_ACTION'}
    """
    # 2.1) Токенизируем текст
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 2.2) Прогон через модель
    with torch.no_grad():
        outputs = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # shape [1, 3]
        pred_idx = int(torch.argmax(logits, dim=1).item())

    label = IDX2LABEL[pred_idx]

    # 2.3) Правила (Decision-Making Logic)
    if label == 'Negative':
        action = 'ALERT'      # Срочное действие
    elif label == 'Neutral':
        action = 'MONITOR'    # Логируем, но без экстренной реакции
    else:  # label == 'Positive'
        action = 'NO_ACTION'  # Ничего не делаем

    return label, action

# 3) Функция для логирования (дописать в CSV)
def log_decision(text: str, label: str, action: str, csv_path: str = 'decision_logs.csv'):
    """
    Добавляем строку в CSV: timestamp, text, label, action.
    Если CSV не существует – создаём и пишем header.
    """
    header = ['timestamp', 'text', 'label', 'action']
    row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), text, label, action]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


# 4) Пример CLI: прочитать ввод пользователя, принять решение и залогировать
if __name__ == '__main__':
    print("=== Decision-Making CLI ===")
    print("Вводите текст (или 'exit' для выхода):")
    while True:
        txt = input("> ")
        if txt.lower() in ('exit', 'quit'):
            break

        lbl, act = make_decision(txt)
        log_decision(txt, lbl, act)
        print(f"[Decision] Label={lbl}, Action={act} → записано в decision_logs.csv")
