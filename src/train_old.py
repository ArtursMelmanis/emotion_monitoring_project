# src/train.py

import os
import argparse
import csv
import pickle
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, classification_report

from data_prep import prepare_data, create_dataloaders
from model import SentimentClassifier


def load_tokenizer_and_model(
    model_name: str,
    num_labels: int,
    local_only: bool,
    device: torch.device
):
    """
    Пытаемся загрузить токенизатор и модель 'model_name' из кэша.
    Если не найдено и local_only=False — скачиваем из интернета.
    Затем переносим модель на device.
    """
    from transformers import BertTokenizerFast, BertForSequenceClassification

    # --- Tokenizer ---
    try:
        print("[INFO] Попытка загрузить токенизатор из локального кэша…")
        tokenizer = BertTokenizerFast.from_pretrained(
            model_name,
            local_files_only=True
        )
        print("[OK] Токенизатор загружен из кэша.")
    except Exception:
        if local_only:
            raise RuntimeError(
                f"[ERROR] Токенизатор '{model_name}' не найден в локальном кэше, а указан --local_only.\n"
                "Запустите один раз без --local_only, чтобы скачать модель:\n"
                "    python src/train.py\n"
            )
        print("[INFO] Токенизатор не найден, скачиваем из интернета (~30 MB)…")
        tokenizer = BertTokenizerFast.from_pretrained(
            model_name,
            local_files_only=False
        )
        print("[OK] Токенизатор скачан и сохранён в кэш.")

    # --- Model ---
    try:
        print("[INFO] Попытка загрузить модель из локального кэша…")
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            local_files_only=True
        )
        print("[OK] Модель загружена из кэша.")
    except Exception:
        if local_only:
            raise RuntimeError(
                f"[ERROR] Модель '{model_name}' не найдена в локальном кэше, а указан --local_only.\n"
                "Запустите один раз без --local_only:\n"
                "    python src/train.py\n"
            )
        print("[INFO] Модель не найдена, скачиваем из интернета (~440 MB)…")
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            local_files_only=False
        )
        print("[OK] Модель скачана и сохранена в кэш.")

    # --- Переносим модель на device ---
    model.to(device)  # type: ignore
    return tokenizer, model


def train(
    resume_from: Optional[str] = None,
    local_only: bool = False,
    epochs: int = 5
):
    """
    Trains (or resumes training) BERT-классификатор. 
    resume_from: путь к checkpoint для продолжения обучения (или None)
    local_only: если указано, токенизатор и модель берутся только из локального кэша
    epochs: число эпох обучения (начальное значение — default=5, можно переопределить)
    """
    # 1) Подготовка данных (train/val/test split + сохранение CSV)
    raw_csv = 'data/raw/sentimentdataset.csv'
    processed_dir = 'data/processed/'
    train_df, val_df, test_df, label2idx = prepare_data(
        csv_path=raw_csv,
        test_size=0.10,
        val_size=0.05,
        random_state=42,
        processed_dir=processed_dir
    )
    print(f"[DATA] train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
    print(f"[DATA] label2idx: {label2idx}")  # {'Negative':0, 'Neutral':1, 'Positive':2}

    # 2) Выбираем устройство (GPU, если доступен)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using device: {device}")

    # 3) Загружаем tokenizer и BERT-модель (с учётом local_only)
    model_name = 'bert-base-uncased'
    num_labels = len(label2idx)
    tokenizer, bert_model = load_tokenizer_and_model(
        model_name=model_name,
        num_labels=num_labels,
        local_only=local_only,
        device=device
    )

    # 4) Resume-логика: если указан resume_from и файл существует, загружаем checkpoint
    optimizer = None
    scheduler = None
    start_epoch = 1

    if resume_from is not None:
        if os.path.exists(resume_from):
            print(f"[RESUME] Loading checkpoint from: {resume_from}")
            ckpt = torch.load(resume_from, map_location=device)
            bert_model.load_state_dict(ckpt['model_state_dict'])

            # Воссоздаём optimizer и scheduler
            optimizer = AdamW(bert_model.parameters(), lr=2e-5, eps=1e-8)
            total_steps = len(train_df) // 16 * epochs  # условно, потом переопределим
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'epoch' in ckpt:
                start_epoch = ckpt['epoch'] + 1
                print(f"[RESUME] Will continue from epoch {start_epoch}")
        else:
            print(f"[WARNING] Checkpoint {resume_from} not found → training from scratch.")

    # 5) Создаём DataLoader’ы (train/val/test)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        max_len=64,
        batch_size=16,
        num_workers=0,
        pin_memory=False
    )
    print(f"[LOADERS] train_batches={len(train_loader)} | val_batches={len(val_loader)} | test_batches={len(test_loader)}")

    # 6) Настраиваем Loss, Optimizer и Scheduler (если не восстановились из ckpt)
    criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = AdamW(bert_model.parameters(), lr=2e-5, eps=1e-8)

    if scheduler is None:
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

    # 7) Подготовка к логированию и чекпоинтам
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    log_path = 'logs/train_log.csv'

    if start_epoch == 1:
        # Если начинаем с нуля — создаём новый лог-файл
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
    else:
        # Если дообучаемся — дописываем в существующий лог
        print(f"[LOG] Appending to existing log: {log_path}")

    best_val_acc = 0.0

    # 8) ЦИКЛ ОБУЧЕНИЯ (теперь от start_epoch до epochs)
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        bert_model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

            # Печатаем прогресс каждые 10 батчей
            if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                print(f"  [Epoch {epoch}] batch {batch_idx}/{len(train_loader)} processed")

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total

        # — Валидация в конце эпохи —
        bert_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item() * input_ids.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"[TRAIN] loss={train_loss:.4f} | acc={train_acc:.4f}")
        print(f"[VAL]   loss={val_loss:.4f} | acc={val_acc:.4f}")

        # Запись в лог (append)
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc])

        # Сохраняем checkpoint, если улучшилась валидационная точность
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_dict = {
                'model_state_dict': bert_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }
            checkpoint_path = 'checkpoints/best_model_bert3cls.pth'
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"[CHECKPOINT] Сохранён новый лучший checkpoint → {checkpoint_path}")

    print("\n===== Training finished =====")
    print(f"[INFO] Best val accuracy = {best_val_acc:.4f}")

    # 9) Оценка на тестовом наборе и сохранение Confusion Matrix
    print("\n=== Оценка на тестовом наборе ===")
    checkpoint_path = 'checkpoints/best_model_bert3cls.pth'
    ckpt = torch.load(checkpoint_path, map_location=device)
    bert_model.load_state_dict(ckpt['model_state_dict'])
    bert_model.eval()

    all_preds = []
    all_labels = []
    test_loss = 0.0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits

            test_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            test_total += labels.size(0)

    test_loss = test_loss / test_total
    test_acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
    print(f"[TEST] loss={test_loss:.4f} | acc={test_acc:.4f}")

    idx2label = {v: k for k, v in label2idx.items()}
    target_names = [idx2label[i] for i in sorted(idx2label)]
    print("\n===== Classification Report =====")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs('logs', exist_ok=True)
    with open('logs/confusion_matrix.pkl', 'wb') as f:
        pickle.dump(cm, f)
    print("[INFO] Сохранена Confusion Matrix → logs/confusion_matrix.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or resume BERT sentiment classifier")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (or None to train from scratch)"
    )
    parser.add_argument(
        "--local_only",
        action="store_true",
        help="If set, load model/tokenizer only from local cache"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,      # ← по умолчанию 10 эпох, можно указать своё число при запуске
        help="Number of epochs to train (default: 10)"
    )
    args = parser.parse_args()
    train(
        resume_from=args.resume_from,
        local_only=args.local_only,
        epochs=args.epochs
    )
