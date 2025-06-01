import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Путь к файлу
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_dir, "data", "raw", "sentimentdataset.csv")

# Загрузка и фильтрация
df = pd.read_csv(data_path)
df["Sentiment"] = df["Sentiment"].str.strip()
df = df[df["Sentiment"].isin(["Positive", "Neutral", "Negative"])]

# Подсчёт
sentiment_counts = df["Sentiment"].value_counts()

# Построение графика
plt.figure(figsize=(6, 4))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Примеры
print("\nSample entries from the dataset:")
print(df[["Text", "Sentiment"]].sample(5, random_state=42))
