# src/evaluate.py

import os
import argparse
import pickle

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data_prep import prepare_data, create_dataloaders
from model import SentimentClassifier


def plot_train_val_metrics(log_csv_path: str = 'logs/train_log.csv', save_dir: str = 'logs/'):
    """
    Строит и сохраняет:
      1) train_loss vs val_loss (loss_curve.png)
      2) train_acc vs val_acc   (accuracy_curve.png)
    из лог-файла 'log_csv_path'.
    """
    if not os.path.exists(log_csv_path):
        print(f"[WARNING] Лог-файл не найден: {log_csv_path}. Пропускаем построение графиков.")
        return

    df = pd.read_csv(log_csv_path)
    epochs = df['epoch'].tolist()
    train_loss = df['train_loss'].tolist()
    val_loss = df['val_loss'].tolist()
    train_acc = df['train_acc'].tolist()
    val_acc = df['val_acc'].tolist()

    # 1) Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='o', label='Train Loss')
    plt.plot(epochs, val_loss, marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    loss_plot_path = os.path.join(save_dir, 'loss_curve.png')
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print(f"[PLOT] Сохранён график Loss → {loss_plot_path}")

    # 2) Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker='o', label='Train Acc')
    plt.plot(epochs, val_acc, marker='o', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs. Validation Accuracy')
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(save_dir, 'accuracy_curve.png')
    plt.tight_layout()
    plt.savefig(acc_plot_path, dpi=200)
    plt.close()
    print(f"[PLOT] Сохранён график Accuracy → {acc_plot_path}")


def plot_confusion_matrix(cm_pkl_path: str = 'logs/confusion_matrix.pkl',
                          class_names: list = ['Negative', 'Neutral', 'Positive'],
                          save_dir: str = 'logs/'):
    """
    Загружает pickle-файл матрицы ошибок и строит heatmap,
    сохраняет как 'confusion_matrix.png'.
    """
    if not os.path.exists(cm_pkl_path):
        print(f"[WARNING] Pickle-файл с Confusion Matrix не найден: {cm_pkl_path}. Пропускаем этот шаг.")
        return

    with open(cm_pkl_path, 'rb') as f:
        cm = pickle.load(f)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    cm_plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"[PLOT] Сохранён Confusion Matrix → {cm_plot_path}")


def evaluate(skip_plots: bool = False):
    """
    1) Если нет data/processed/test.csv → вызывает prepare_data
    2) Загружает test_loader
    3) Загружает checkpoint, делает inference на тесте
    4) Выводит classification_report, сохраняет confusion_matrix.pkl
    5) (Опционально) Строит графики loss/acc и heatmap CM
    """
    raw_csv = 'data/raw/sentimentdataset.csv'
    processed_dir = 'data/processed/'

    if not os.path.exists(os.path.join(processed_dir, 'test.csv')):
        print("[WARNING] data/processed/test.csv не найден. Пересоздаём разбиение…")
        prepare_data(
            csv_path=raw_csv,
            test_size=0.10,
            val_size=0.05,
            random_state=42,
            processed_dir=processed_dir
        )

    # Читаем CSV-ки
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(processed_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'))

    # 2) Загружаем tokenizer и создаём test_loader
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', local_files_only=False)

    _, _, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        max_len=64,
        batch_size=16,
        num_workers=0,
        pin_memory=False
    )

    # 3) Загружаем checkpoint и делаем inference
    _, _, _, label2idx = prepare_data(
        csv_path=raw_csv,
        test_size=0.10,
        val_size=0.05,
        random_state=42,
        processed_dir=processed_dir
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentClassifier(pretrained_model_name='bert-base-uncased', num_labels=len(label2idx))

    checkpoint_path = 'checkpoints/best_model_bert3cls.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.bert_model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"[MODEL] Loaded checkpoint → {checkpoint_path} on {device}")

    all_preds = []
    all_labels = []
    test_loss = 0.0
    test_total = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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

    # 4) Classification Report
    idx2label = {v: k for k, v in label2idx.items()}
    target_names = [idx2label[i] for i in sorted(idx2label)]
    print("\n===== Classification Report =====")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 5) Сохраняем Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs('logs', exist_ok=True)
    with open('logs/confusion_matrix.pkl', 'wb') as f:
        pickle.dump(cm, f)
    print("[INFO] Confusion matrix saved → logs/confusion_matrix.pkl")

    # 6) Строим графики (если skip_plots=False)
    if not skip_plots:
        print("\n=== Building train/val loss & acc curves ===")
        plot_train_val_metrics(log_csv_path='logs/train_log.csv', save_dir='logs/')
        print("\n=== Building Confusion Matrix heatmap ===")
        plot_confusion_matrix(cm_pkl_path='logs/confusion_matrix.pkl',
                              class_names=[idx2label[i] for i in sorted(idx2label)],
                              save_dir='logs/')
    else:
        print("[INFO] Пропущено построение графиков (--skip_plots)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BERT sentiment classifier and plot metrics")
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Если указано, графики не будут строиться."
    )
    args = parser.parse_args()
    evaluate(skip_plots=args.skip_plots)
