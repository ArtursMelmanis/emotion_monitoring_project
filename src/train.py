# src/train.py

import os
import argparse
import csv
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from data_prep import prepare_data, create_dataloaders


def load_tokenizer_and_model(model_name, num_labels, local_only, device):
    from transformers import BertTokenizerFast, BertForSequenceClassification, BertConfig

    # Настроим config с пробросом dropout = 0.3
    config = BertConfig.from_pretrained(model_name, num_labels=num_labels)
    config.hidden_dropout_prob = 0.3
    config.attention_probs_dropout_prob = 0.3

    # Попытка загрузки локально
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_name, local_files_only=True)
    except Exception:
        if local_only:
            raise RuntimeError("Tok not in cache, а указан --local_only")
        tokenizer = BertTokenizerFast.from_pretrained(model_name)

    try:
        model = BertForSequenceClassification.from_pretrained(model_name, config=config, local_files_only=True)
    except Exception:
        if local_only:
            raise RuntimeError("Model not in cache, а указан --local_only")
        model = BertForSequenceClassification.from_pretrained(model_name, config=config)

    model.to(device)
    return tokenizer, model


def train(resume_from=None, local_only=False, epochs=5):
    # 1) Данные
    train_df, val_df, test_df, label2idx = prepare_data(
        csv_path='data/raw/sentimentdataset.csv',
        test_size=0.10, val_size=0.05, random_state=42, processed_dir='data/processed/'
    )
    print(f"[DATA] tr={len(train_df)} | val={len(val_df)} | test={len(test_df)}  labels={label2idx}")

    # 2) Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")

    # 3) Tokenizer + Model (с dropout=0.3)
    model_name = 'bert-base-uncased'
    num_labels = len(label2idx)
    tokenizer, bert_model = load_tokenizer_and_model(
        model_name, num_labels, local_only, device
    )

    # 4) Resume logic
    optimizer = None
    scheduler = None
    start_epoch = 1
    saved_scheduler_state = None

    if resume_from and os.path.exists(resume_from):
        print(f"[RESUME] loading checkpoint {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        bert_model.load_state_dict(ckpt['model_state_dict'])

        optimizer = AdamW(bert_model.parameters(), lr=5e-6, eps=1e-8, weight_decay=0.01)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            saved_scheduler_state = ckpt['scheduler_state_dict']
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            print(f"[RESUME] start_epoch = {start_epoch}")
    elif resume_from:
        print(f"[WARN] {resume_from} not found → from scratch")

    # 5) DataLoader (batch_size=32)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        tokenizer=tokenizer,
        max_len=64,
        batch_size=32,     # увеличили с 16 до 32
        num_workers=2,
        pin_memory=True
    )
    print(f"[LOADERS] train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # 6.1) Class weights
    train_labels = train_df['label_idx'].values
    cw = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 6.2) Optimizer
    if optimizer is None:
        optimizer = AdamW(bert_model.parameters(), lr=1e-5, eps=1e-8, weight_decay=0.01)

    # 6.3) Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    if saved_scheduler_state:
        scheduler.load_state_dict(saved_scheduler_state)

    # 7) Checkpoints & Logging
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    log_path = 'logs/train_log.csv'
    if start_epoch == 1:
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch','train_loss','val_loss','train_acc','val_acc'])
    else:
        print(f"[LOG] append to {log_path}")

    best_val_acc = 0.0
    patience = 3
    stale = 0

    # 8) Train loop
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        bert_model.train()
        running_loss, running_correct, running_total = 0, 0, 0

        for i, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = bert_model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if i % 10 == 0 or i == len(train_loader):
                print(f"  [Epoch{epoch}] batch {i}/{len(train_loader)}")

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # Validation
        bert_model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = bert_model(input_ids, attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item() * input_ids.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"[TRAIN] loss={train_loss:.4f} acc={train_acc:.4f}")
        print(f"[VAL]   loss={val_loss:.4f} acc={val_acc:.4f}")

        # Early stopping logic
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, train_acc, val_acc])

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            stale = 0
            ckpt = {
                'model_state_dict': bert_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(ckpt, 'checkpoints/best_model_bert3cls.pth')
            print(f"[CHECKPOINT] saved best at epoch {epoch}")
        else:
            stale += 1
            if stale >= patience:
                print(f"[EARLY STOP] no improvement for {patience} epochs → stop")
                break

    print(f"\n===== Done training. Best val_acc = {best_val_acc:.4f} =====")

    # 9) Final evaluation on test set
    print("\n=== Test evaluation ===")
    ckpt = torch.load('checkpoints/best_model_bert3cls.pth', map_location=device)
    bert_model.load_state_dict(ckpt['model_state_dict'])
    bert_model.eval()

    all_preds, all_labels = [], []
    test_loss, test_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = bert_model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            test_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            test_total += labels.size(0)

    test_loss = test_loss / test_total
    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    idx2label = {v: k for k, v in label2idx.items()}
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=[idx2label[i] for i in sorted(idx2label)]))

    cm = confusion_matrix(all_labels, all_preds)
    with open('logs/confusion_matrix.pkl', 'wb') as f:
        pickle.dump(cm, f)
    print("[INFO] Confusion matrix saved → logs/confusion_matrix.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or resume BERT sentiment classifier")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint to resume training from (or None to train from scratch)")
    parser.add_argument("--local_only", action="store_true",
                        help="Load model/tokenizer only from локального кэша")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs (default: 10)")
    args = parser.parse_args()
    train(resume_from=args.resume_from, local_only=args.local_only, epochs=args.epochs)
