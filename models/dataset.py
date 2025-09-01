import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import DistilBertTokenizerFast
from sklearn.preprocessing import LabelEncoder


def prepare_datasets(X_train, X_test, y_train, y_test):
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Example: Convert pandas series to a list
    train_texts = X_train['text'].tolist()
    test_texts = X_test['text'].tolist()

    # Tokenize: This splits text, handles max length, adds special tokens
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=320)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=320)

    return train_encodings, test_encodings

class ComplaintsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item