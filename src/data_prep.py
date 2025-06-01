import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizerFast

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def map_to_polarity(raw_label: str) -> str:
    label = raw_label.strip().lower()
    if label == 'positive':
        return 'Positive'
    if label == 'negative':
        return 'Negative'
    if label == 'neutral':
        return 'Neutral'
    
    positive_emotions = {
        'joy', 'happiness', 'happy', 'contentment', 'gratitude', 'awe',
        'admiration', 'amusement', 'adoration', 'affection', 'enjoyment',
        'excited', 'excitement', 'hopeful', 'hope', 'serenity', 'euphoria',
        'inspiration', 'relief', 'surprise'
    }
    negative_emotions = {
        'sadness', 'sad', 'despair', 'grief', 'anger', 'disgust',
        'fear', 'loneliness', 'embarrassed', 'disappointed', 'confusion'
    }

    if label in positive_emotions:
        return 'Positive'
    if label in negative_emotions:
        return 'Negative'
    return 'Neutral'


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer: BertTokenizerFast, max_len: int):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_data(
    csv_path: str,
    test_size: float = 0.10,
    val_size: float = 0.05,
    random_state: int = 42,
    processed_dir: str = 'data/processed/'
):
    df_full = pd.read_csv(csv_path)

    #  Убираем лишние "Unnamed" колонки, если они есть
    cols = df_full.columns.tolist()
    if cols[0].startswith('Unnamed'):
        df_full = df_full.drop(columns=[cols[0]])
        cols = df_full.columns.tolist()
    if cols[0] == 'Unnamed: 0':
        df_full = df_full.drop(columns=[cols[0]])
        cols = df_full.columns.tolist()

    # Проверяем наличие нужных колонок
    if 'Text' not in df_full.columns or 'Sentiment' not in df_full.columns:
        raise KeyError(f"Ожидались колонки 'Text' и 'Sentiment', но найдены: {df_full.columns.tolist()}")

    df = df_full[['Text', 'Sentiment']].dropna().reset_index(drop=True)
    df.columns = ['text', 'emotion']  # Переименуем для удобства

    # 1) Преобразуем raw emotion в polarity (три класса)
    df['polarity'] = df['emotion'].apply(map_to_polarity)

    # 2) Кодируем три метки: Negative=0, Neutral=1, Positive=2
    unique_labels = ['Negative', 'Neutral', 'Positive']
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    df['label_idx'] = df['polarity'].map(label2idx)

    # 3) Чистим текст
    df['clean_text'] = df['text'].apply(clean_text)

    # 4) Разбиваем на train+val и test (stratify по label_idx)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label_idx']
    )
    # Разбиваем train_val на train и val
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_df['label_idx']
    )

    # 5) Сохраняем CSV'ки в папку processed_dir
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)

    return train_df, val_df, test_df, label2idx


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: BertTokenizerFast,
    max_len: int = 64,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False
):
    train_texts = train_df['clean_text'].to_numpy()
    train_labels = train_df['label_idx'].to_numpy()
    val_texts = val_df['clean_text'].to_numpy()
    val_labels = val_df['label_idx'].to_numpy()
    test_texts = test_df['clean_text'].to_numpy()
    test_labels = test_df['label_idx'].to_numpy()

    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_len)
    test_dataset = EmotionDataset(test_texts, test_labels, tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
