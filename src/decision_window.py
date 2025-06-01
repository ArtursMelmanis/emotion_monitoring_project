# decision_window.py

import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

CSV_PATH = 'decision_logs.csv'

class DecisionWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Decision-Making Logs")
        self.geometry("800x400")

        # Кнопка обновить
        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        refresh_btn = tk.Button(btn_frame, text="Refresh", command=self.load_logs)
        refresh_btn.pack(side=tk.LEFT)

        clear_btn = tk.Button(btn_frame, text="Clear Log", command=self.clear_logs)
        clear_btn.pack(side=tk.LEFT, padx=10)

        export_btn = tk.Button(btn_frame, text="Export to CSV", command=self.export_csv)
        export_btn.pack(side=tk.LEFT)

        # Табличка (Treeview)
        columns = ("timestamp", "text", "label", "action")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, anchor=tk.W, width=180 if col=="text" else 100)

        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # При старте сразу загрузим
        self.load_logs()

    def load_logs(self):
        """Считываем весь CSV и заполняем таблицу заново."""
        # Сначала очищаем старые записи:
        for row in self.tree.get_children():
            self.tree.delete(row)

        if not os.path.exists(CSV_PATH):
            return

        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            values = (row['timestamp'], row['text'], row['label'], row['action'])
            self.tree.insert("", tk.END, values=values)

    def clear_logs(self):
        """Очистить CSV-файл и таблицу."""
        if os.path.exists(CSV_PATH):
            os.remove(CSV_PATH)
        self.load_logs()

    def export_csv(self):
        """Просто продублируем текущий CSV куда-нибудь (например, decision_logs_export.csv)."""
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df.to_csv("decision_logs_export.csv", index=False)
            messagebox.showinfo("Export", "Logs exported to decision_logs_export.csv")
        else:
            messagebox.showwarning("Warning", "No decision_logs.csv to export.")


if __name__ == "__main__":
    app = DecisionWindow()
    app.mainloop()
