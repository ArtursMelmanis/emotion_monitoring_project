import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_train_val_metrics(log_csv_path: str = 'logs/train_log.csv', save_dir: str = 'logs/'):
    """
    Строит два отдельных графика:
      1) train_loss vs val_loss по эпохам
      2) train_acc vs val_acc по эпохам
    Сохраняет их в папке save_dir как 'loss_curve.png' и 'accuracy_curve.png'.
    """
    if not os.path.exists(log_csv_path):
        raise FileNotFoundError(f"Лог-файл не найден: {log_csv_path}")

    df = pd.read_csv(log_csv_path)
    epochs = df['epoch'].tolist()
    train_loss = df['train_loss'].tolist()
    val_loss = df['val_loss'].tolist()
    train_acc = df['train_acc'].tolist()
    val_acc = df['val_acc'].tolist()

    # 1) График Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='o', label='Train Loss')
    plt.plot(epochs, val_loss, marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(save_dir, 'loss_curve.png')
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()
    print(f"[PLOT] Сохранён график Loss → {loss_plot_path}")

    # 2) График Accuracy
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
    Загружает pickle-файл с матрицей ошибок (confusion matrix) и 
    строит heatmap, сохраняет как 'confusion_matrix.png'.
    """
    if not os.path.exists(cm_pkl_path):
        raise FileNotFoundError(f"Pickle-файл с CM не найден: {cm_pkl_path}")

    with open(cm_pkl_path, 'rb') as f:
        cm = pickle.load(f)

    fig, ax = plt.subplots(figsize=(6, 5))
    # Вместо plt.cm.Blues пишем просто 'Blues', чтобы Pylance не ругался
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Подписи на осях
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # Подписи над ячейками
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

    cm_plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"[PLOT] Сохранён Confusion Matrix → {cm_plot_path}")


if __name__ == "__main__":
    print("=== Plotting Train/Val Metrics ===")
    plot_train_val_metrics()

    print("\n=== Plotting Confusion Matrix ===")
    plot_confusion_matrix(class_names=['Negative', 'Neutral', 'Positive'])
