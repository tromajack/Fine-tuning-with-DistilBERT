import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers import logging as hf_logging
import numpy as np
from src.config import MODEL_SAVE_PATH as model_save_path
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

hf_logging.set_verbosity_error()  # Suppress warnings from transformers library

def train_model(train_dataset, test_dataset, num_labels, epochs=3, batch_size=16, learning_rate=5e-5):
    """
    Fine-tune DistilBERT for multi-class text classification.
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. Load pre-trained model
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=num_labels
        )
        model.to(device)    
        
        # 2. Prepare Data Loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 3. Optimizer and Scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps), 
            num_training_steps=total_steps
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            model.train()
            train_losses = []

            for batch in tqdm(train_loader, desc="Training"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"], 
                    attention_mask=batch["attention_mask"], 
                    labels=batch["labels"]
                )
                loss = outputs.loss
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            avg_train_loss = np.mean(train_losses)
            print(f"  Avg train loss: {avg_train_loss:.4f}")
            # Validation
            model.eval()
            val_losses = []
            preds = []
            labels = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Validation"):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(
                        input_ids=batch["input_ids"], 
                        attention_mask=batch["attention_mask"], 
                        labels=batch["labels"]
                    )
                    loss = outputs.loss
                    logits = outputs.logits
                    val_losses.append(loss.item())
                    preds += torch.argmax(logits, dim=1).cpu().numpy().tolist()
                    labels += batch["labels"].cpu().numpy().tolist()
            avg_val_loss = np.mean(val_losses)
            val_acc = accuracy_score(labels, preds)
            val_f1 = f1_score(labels, preds, average='macro')
            print(f"  Validation loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f} | Macro F1: {val_f1:.4f}")

        # Save the trained model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

        print("\nClassification report on validation set:")
        print(classification_report(labels, preds))

        return model

    except Exception as e:
        print(f"Error during training: {e}")
        return None
    