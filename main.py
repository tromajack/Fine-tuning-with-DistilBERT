#main.py
from src.utils.logging import app_logger
import os
import requests
from datetime import datetime, timedelta
import pandas as pd
from src.utils.data_loader import dataloader
from src.utils.cleaning import clean_text
from src.utils.cleaning import label_encoding
from models.train import train_model
from models import train_test_splitter
from models.dataset import prepare_datasets, ComplaintsDataset
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

def main():
    try:
        if not os.path.exists(RAW_DATA_DIR / "consumer_complaints.csv"):
            save_path = RAW_DATA_DIR / f"consumer_complaints.csv"
            path = dataloader.download_complaints_csv(
            start_date="2021-01-01", 
            end_date="2025-01-01", 
            size=100000,
            save_path=save_path)
            app_logger.info(f"Data downloaded to {path}")
        else:
            path = RAW_DATA_DIR / f"consumer_complaints.csv"
        
        if not os.path.exists(PROCESSED_DATA_DIR / "processed_consumer_complaints.csv"):
            data = dataloader.load_complaints_csv(path)
            # Clean the 'narrative' column
            data['text'] = data['text'].apply(clean_text)
            app_logger.info("Text cleaning completed.")
            save_path = PROCESSED_DATA_DIR / f"processed_consumer_complaints.csv"
            data.to_csv(save_path, index=False)
            app_logger.info(f"Processed data saved to {save_path}")
        else:
            data = pd.read_csv(PROCESSED_DATA_DIR / "processed_consumer_complaints.csv",header=0)
            app_logger.info(f"Loaded processed data from {PROCESSED_DATA_DIR / 'processed_consumer_complaints.csv'}")
        print(f"Random cleaned text view: {data['text'].iloc[23:24].values}")
        data = label_encoding(data) #merges labels into common categories and encodes them
        app_logger.info(f"Label encoding completed. Random label view: {data['label'].iloc[23:24].values}")
        data.dropna(inplace=True)
        X_train, X_test, y_train, y_test = train_test_splitter.split_data(data)
        app_logger.info("Data split into train and test sets.")
        train_encodings, test_encodings = prepare_datasets(X_train, X_test, y_train, y_test)
        app_logger.info("Datasets prepared with tokenization.")
        train_dataset = ComplaintsDataset(train_encodings, y_train.tolist())
        test_dataset = ComplaintsDataset(test_encodings, y_test.tolist())
        app_logger.info("Dataset objects created.")
        num_labels = len(data['label'].unique())
        model = train_model(train_dataset, test_dataset, num_labels=num_labels, epochs=3, batch_size=16, learning_rate=5e-5)
        app_logger.info("Model training completed.")
    except Exception as e:
        app_logger.error(f"Failed to run: {e}")


if __name__ == "__main__":
    main()