# ====== gui_app.py ======

import os
import csv
import torch
import torch.nn.functional as F
from datetime import datetime

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")

from PIL import Image, ImageTk
from transformers import BertTokenizerFast
from model import SentimentClassifier

# Для bar‐chart / line‐chart
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ====== Пути и константы ======
CHECKPOINT_PATH       = 'checkpoints/best_model_bert3cls.pth'
DECISION_CSV          = 'decision_logs.csv'
LOGS_DIR              = 'logs'

LOSS_CURVE_PATH       = os.path.join(LOGS_DIR, 'loss_curve.png')
ACCURACY_CURVE_PATH   = os.path.join(LOGS_DIR, 'accuracy_curve.png')
CONFUSION_MATRIX_PATH = os.path.join(LOGS_DIR, 'confusion_matrix.png')

IDX2LABEL = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


# ====== Загрузка модели и токенизатора ======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', local_files_only=True)
except Exception:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', local_files_only=False)

model = SentimentClassifier(pretrained_model_name='bert-base-uncased', num_labels=3)
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.bert_model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()


# ====== Функции Decision-Making ======

def make_decision(text: str) -> tuple[str, str, float]:
    """
    Возвращает (label, action, confidence),
    где confidence – вероятность предсказанного класса (0.0–1.0).
    """
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits               # shape = [1, num_labels]
        probs = F.softmax(logits, dim=1)      # shape = [1, num_labels]

    # 3) Берём индекс предсказанного класса и явно конвертируем его в int
    pred_idx_raw = torch.argmax(probs, dim=1).item()
    pred_idx = int(pred_idx_raw)  # теперь Pylance точно знает, что это int

    # 4) Получаем строковый label и percent‐confidence
    label = IDX2LABEL[pred_idx]  # здесь используется ваш словарь IDX2LABEL
    confidence_raw = probs[0][pred_idx].item()
    confidence = float(confidence_raw)  # теперь это чистый float

    # 5) Решаем, какое действие («ALERT», «MONITOR» или «NO_ACTION») в зависимости от метки
    if label == "Negative":
        action = "ALERT"
    elif label == "Neutral":
        action = "MONITOR"
    else:
        action = "NO_ACTION"

    return label, action, confidence


def log_decision(text: str, label: str, action: str, confidence: float):
    """
    Записывает в DECISION_CSV строку:
    Timestamp, text, label, action, confidence
    """
    header = ["Timestamp", "text", "label", "action", "confidence"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        text,
        label,
        action,
        f"{confidence:.4f}"
    ]

    # Узнаем, есть ли в DECISION_CSV указание папки:
    directory = os.path.dirname(DECISION_CSV)
    if directory:
        # Если directory != "", создаём вложенные папки (при их отсутствии)
        os.makedirs(directory, exist_ok=True)

    # Если файла нет — создаём его и пишем header
    if not os.path.exists(DECISION_CSV):
        with open(DECISION_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Дозаписываем новую строку
    with open(DECISION_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ====== Основное GUI-приложение ======

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion Monitoring System")
        self.geometry("1000x600")

        # Инициализируем ID таймеров (чтобы Pylance знал, что они существуют)
        self.auto_refresh_id = None
        self.dist_refresh_id = None

        # Notebook (вкладки)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка 1: Decision-Making
        self.tab1 = tk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Decision-Making")

        # Вкладка 2: Statistics
        self.tab2 = tk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Statistics")

        # Вкладка 3: Monitoring – Recent Predictions
        self.tab3 = tk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Monitoring: Recent")

        # Вкладка 4: Monitoring – Confidence over Time
        self.tab4 = tk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="Monitoring: Distribution")

        # Инициализируем каждый таб
        self._init_tab1_decision()
        self._init_tab2_statistics()
        self._init_tab3_monitoring_recent()
        self._init_tab4_monitoring_distribution()

        # При закрытии отменяем все живые таймеры
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ---------------------- Вкладка 1: Decision-Making ----------------------
    def _init_tab1_decision(self):
        frame_input = tk.Frame(self.tab1)
        frame_input.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(frame_input, text="Enter text to classify:", font=("Arial", 10)).pack(side=tk.LEFT)
        self.text_entry = tk.Entry(frame_input, width=60)
        self.text_entry.pack(side=tk.LEFT, padx=5)

        classify_btn = tk.Button(frame_input, text="Classify", command=self.on_classify)
        classify_btn.pack(side=tk.LEFT, padx=5)

        self.result_label = tk.Label(self.tab1, text="Result: —", font=("Arial", 10, "italic"))
        self.result_label.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=5)

        tk.Label(self.tab1, text="Decision Logs:", font=("Arial", 12, "bold")).pack(side=tk.TOP, anchor=tk.W, padx=10)

        # Здесь оставляем только 4 колонки: timestamp, text, label, action (игнорируем confidence)
        columns = ("timestamp", "text", "label", "action")
        self.tree1 = ttk.Treeview(self.tab1, columns=columns, show="headings", height=12)
        for col in columns:
            self.tree1.heading(col, text=col.capitalize())
            width = 350 if col == "text" else 120
            self.tree1.column(col, anchor=tk.W, width=width)
        self.tree1.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=5)

        frame_buttons = tk.Frame(self.tab1)
        frame_buttons.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        refresh_btn = tk.Button(frame_buttons, text="Update log", command=self.load_logs_tab1)
        refresh_btn.pack(side=tk.LEFT)

        clear_btn = tk.Button(frame_buttons, text="Clear log", command=self.clear_logs_tab1)
        clear_btn.pack(side=tk.LEFT, padx=5)

        export_btn = tk.Button(frame_buttons, text="Export logs", command=self.export_logs_tab1)
        export_btn.pack(side=tk.LEFT, padx=5)

        self.load_logs_tab1()

    def on_classify(self):
        text = self.text_entry.get().strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text for classification.")
            return

        # <-- Здесь мы распаковываем ТРИ значения, потому что make_decision их возвращает
        label, action, confidence = make_decision(text)

        color = "red" if action == "ALERT" else ("orange" if action == "MONITOR" else "green")
        self.result_label.config(
            text=f"Label = {label}    →    Action = {action}    →    Confidence = {confidence*100:.1f}%",
            fg=color
        )

        # <-- Передаём в лог теперь четыре поля: text, label, action, confidence
        log_decision(text, label, action, confidence)

        self.load_logs_tab1()
        self.update_recent_chart()
        self.update_distribution_tab4()

    def load_logs_tab1(self):
        for row_id in self.tree1.get_children():
            self.tree1.delete(row_id)
        if not os.path.exists(DECISION_CSV):
            return

        df = pd.read_csv(DECISION_CSV)
        for row in df.itertuples(index=False):
            # row = (Timestamp, text, label, action, confidence)
            # но в таблицу Decision Logs мы сохраняем первые 4: row.Timestamp, row.text, row.label, row.action
            self.tree1.insert(
                "",
                tk.END,
                values=(row.Timestamp, row.text, row.label, row.action)
            )

    def clear_logs_tab1(self):
        if os.path.exists(DECISION_CSV):
            os.remove(DECISION_CSV)
        self.load_logs_tab1()
        # Обновим графики
        self.update_recent_chart()
        self.update_distribution_tab4()

    def export_logs_tab1(self):
        if os.path.exists(DECISION_CSV):
            df = pd.read_csv(DECISION_CSV)
            df.to_csv("decision_logs_export.csv", index=False)
            messagebox.showinfo("Export", "Logs exported to decision_logs_export.csv")
        else:
            messagebox.showwarning("Warning", "The file decision_logs.csv was not found.")

    # ---------------------- Вкладка 2: Statistics ----------------------
    def _init_tab2_statistics(self):
        frame_top = tk.Frame(self.tab2)
        frame_top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(frame_top, text="Statistics (Loss & Accuracy)", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        stats_refresh_btn = tk.Button(frame_top, text="Refresh", command=self.load_stats_tab2)
        stats_refresh_btn.pack(side=tk.RIGHT)

        self.stats_canvas = tk.Frame(self.tab2)
        self.stats_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.load_stats_tab2()

    def load_stats_tab2(self):
        # Очищаем старые виджеты
        for widget in self.stats_canvas.winfo_children():
            widget.destroy()

        if not os.path.exists(LOSS_CURVE_PATH) or not os.path.exists(ACCURACY_CURVE_PATH):
            tk.Label(self.stats_canvas, text="No charts found.\nRun evaluate.py", fg="red").pack(padx=20, pady=20)
            return

        try:
            img1 = Image.open(LOSS_CURVE_PATH)
            img2 = Image.open(ACCURACY_CURVE_PATH)
        except Exception as e:
            tk.Label(self.stats_canvas, text=f"Failed to open images:\n{e}", fg="red").pack(padx=20, pady=20)
            return

        # Выбор ресэмплинга в зависимости от версии Pillow
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.ANTIALIAS  # type: ignore[reportAttributeAccess]

        img1 = img1.resize((400, 300), resample_filter)
        img2 = img2.resize((400, 300), resample_filter)

        photo1 = ImageTk.PhotoImage(img1)
        photo2 = ImageTk.PhotoImage(img2)

        self.stats_canvas.loss_img = photo1  # type: ignore[reportAttributeAccess]
        self.stats_canvas.acc_img = photo2   # type: ignore[reportAttributeAccess]

        frm = tk.Frame(self.stats_canvas)
        frm.pack(fill=tk.BOTH, expand=True)

        lbl1 = tk.Label(frm, text="Train vs. Validation Loss", font=("Arial", 11, "bold"))
        lbl1.pack(side=tk.TOP, anchor=tk.W)
        canvas1 = tk.Label(frm, image=photo1)
        canvas1.pack(side=tk.LEFT, padx=5, pady=5)

        lbl2 = tk.Label(frm, text="Train vs. Validation Accuracy", font=("Arial", 11, "bold"))
        lbl2.pack(side=tk.TOP, anchor=tk.E)
        canvas2 = tk.Label(frm, image=photo2)
        canvas2.pack(side=tk.RIGHT, padx=5, pady=5)

    # ---------------------- Вкладка 3: Monitoring – Recent Predictions ----------------------
    def _init_tab3_monitoring_recent(self):
        tk.Label(self.tab3,
                 text="Recent Predictions (percentage composition of the last N)",
                 font=("Arial", 12, "bold")
        ).pack(side=tk.TOP, anchor=tk.W, padx=10, pady=5)

        # Frame для графика
        self.recent_chart_frame = tk.Frame(self.tab3)
        self.recent_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.recent_fig = Figure(figsize=(5, 3), dpi=100)
        self.recent_ax = self.recent_fig.add_subplot(111)
        self.recent_ax.set_title("Distribution of labels in the last N predictions")
        self.recent_ax.set_ylabel("Percent (%)")
        self.recent_ax.set_ylim(0, 100)

        self.recent_canvas = FigureCanvasTkAgg(self.recent_fig, master=self.recent_chart_frame)
        self.recent_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        refresh_btn = ttk.Button(self.tab3, text="Update chart", command=self.update_recent_chart)
        refresh_btn.pack(side=tk.BOTTOM, pady=5)

        self.update_recent_chart()

    def update_recent_chart(self):
        """
        Читает последние N записей из DECISION_CSV и рисует bar‐chart
        с процентным распределением меток.
        """
        if not os.path.exists(DECISION_CSV):
            self.recent_ax.clear()
            self.recent_ax.set_title("No data to draw")
            self.recent_canvas.draw()
            return

        df = pd.read_csv(DECISION_CSV)
        N = 20
        last_df = df.tail(N)

        counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
        for lbl in last_df['label']:
            if lbl in counts:
                counts[lbl] += 1

        total = len(last_df)
        labels = ['Negative', 'Neutral', 'Positive']
        percents = []
        for key in labels:
            cnt = counts[key]
            pct = (cnt / total * 100) if total > 0 else 0
            percents.append(pct)

        self.recent_ax.clear()
        self.recent_ax.set_title(f"Percentage of tags for the last {total} posts")
        self.recent_ax.set_ylabel("Percent (%)")
        self.recent_ax.set_ylim(0, 100)

        bar_colors = ['tab:red', 'tab:orange', 'tab:green']
        bars = self.recent_ax.bar(labels, percents, color=bar_colors)
        # Если Pylance снова жалуется, напишите ниже # type: ignore
        self.recent_ax.bar_label(
            bars,
            labels=[f"{p:.1f}%" for p in percents],  # type: ignore[reportArgumentType]
            padding=3,
            fontsize=9
        )

        self.recent_canvas.draw()
        # Сохранить ID таймера, чтобы потом отменить, когда окно закроется
        self.auto_refresh_id = self.after(5000, self.update_recent_chart)

    # ---------------------- Обработчик закрытия окна ----------------------
    def on_closing(self):
        # Отменяем таймер вкладки “Recent”, если он есть
        if self.auto_refresh_id is not None:
            try:
                self.after_cancel(self.auto_refresh_id)
            except Exception:
                pass

        # Отменяем таймер вкладки “Distribution”, если он есть
        if self.dist_refresh_id is not None:
            try:
                self.after_cancel(self.dist_refresh_id)
            except Exception:
                pass

        self.destroy()

    # ---------------------- Вкладка 4: Monitoring – Confidence over Time ----------------------
    def _init_tab4_monitoring_distribution(self):
        tk.Label(self.tab4, text="Monitoring: Confidence over Time", font=("Arial", 12, "bold")).pack(
            side=tk.TOP, anchor=tk.W, padx=10, pady=5
        )

        self.dist_chart_frame = tk.Frame(self.tab4)
        self.dist_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.dist_fig = Figure(figsize=(6, 3), dpi=100)
        self.dist_ax = self.dist_fig.add_subplot(111)
        self.dist_ax.set_title("Confidence (probability) of the last N predictions")
        self.dist_ax.set_xlabel("Index (1 = oldest of last N)")
        self.dist_ax.set_ylabel("Confidence (%)")
        self.dist_ax.set_ylim(0, 100)

        self.dist_canvas = FigureCanvasTkAgg(self.dist_fig, master=self.dist_chart_frame)
        self.dist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        refresh_btn2 = ttk.Button(self.tab4, text="Update chart", command=self.update_distribution_tab4)
        refresh_btn2.pack(side=tk.BOTTOM, pady=5)

        self.update_distribution_tab4()

    def update_distribution_tab4(self):
        """
        Читает последние N строк из DECISION_CSV и строит line-chart
        с confidence (%) для каждого предсказания.
        """
        if not os.path.exists(DECISION_CSV):
            self.dist_ax.clear()
            self.dist_ax.set_title("No data to plot")
            self.dist_canvas.draw()
            return

        df = pd.read_csv(DECISION_CSV)
        N = 20
        last_df = df.tail(N)

        confidences = []
        for val in last_df['confidence']:
            try:
                conf = float(val) * 100  # если в CSV сохранено 0.0–1.0
            except Exception:
                conf = float(val)
            confidences.append(conf)

        total = len(confidences)
        xs = list(range(1, total + 1))

        self.dist_ax.clear()
        self.dist_ax.set_title(f"Confidence of the last {total} predictions")
        self.dist_ax.set_xlabel("Index (1 = oldest)")
        self.dist_ax.set_ylabel("Confidence (%)")
        self.dist_ax.set_ylim(0, 100)

        # Рисуем линию: xs по оси X, confidences по оси Y
        self.dist_ax.plot(xs, confidences, marker='o', linestyle='-', color='tab:blue')
        for x, y in zip(xs, confidences):
            self.dist_ax.text(x, y + 1, f"{y:.1f}%", fontsize=7, ha='center')

        self.dist_canvas.draw()
        # Сохранить ID таймера, чтобы потом отменить
        self.dist_refresh_id = self.after(5000, self.update_distribution_tab4)


if __name__ == "__main__":
    app = App()
    app.mainloop()
