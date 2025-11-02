import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
from torch.optim import Adam
import time

# Dataset Class
class MessageDataset(Dataset):
    def __init__(self, messages, labels, tokenizer, max_length=128):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        text = str(self.messages[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float32),
        }

# Load Data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, low_memory=False)
        if "message" not in df.columns or len(df.columns) < 5:
            raise ValueError("Invalid dataset structure. Ensure 'message' column exists and labels are present.")

        df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric, errors="coerce").fillna(0)
        labels = df.iloc[:, 4:].values.astype("float32")
        labels = np.clip(labels, 0, 1)  # Ensure binary values
        features = df["message"].values
        print(f"Loaded data from {file_path}. Unique label values: {np.unique(labels)}")
        return features, labels
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Training Function
def train_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (idx + 1) % 10 == 0:
            print(f"Batch {idx + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")
    return total_loss / len(data_loader)

# Validation Function
def validate_epoch(model, data_loader, loss_fn, device):
    model.eval()
    val_loss = 0
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            val_loss += loss_fn(outputs.logits, labels).item()

            preds.extend(torch.sigmoid(outputs.logits).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(data_loader)
    true_labels = np.array(true_labels)
    preds = (np.array(preds) > 0.5).astype(int)

    precision = precision_score(true_labels, preds, average="samples", zero_division=0)
    recall = recall_score(true_labels, preds, average="samples", zero_division=0)
    f1 = f1_score(true_labels, preds, average="samples", zero_division=0)

    print(f"Validation - Unique True Labels: {np.unique(true_labels)}")
    print(f"Validation - Unique Predictions: {np.unique(preds)}")

    return avg_val_loss, precision, recall, f1

# Main Training Loop
def main():
    try:
        train_features, train_labels = load_data(r"C:\Users\ebins\MiniPro\data\disaster_response_messages-main\disaster_response_training.csv")
        val_features, val_labels = load_data(r"C:\Users\ebins\MiniPro\data\disaster_response_messages-main\disaster_response_validation.csv")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=train_labels.shape[1]
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_dataset = MessageDataset(train_features, train_labels, tokenizer)
        val_dataset = MessageDataset(val_features, val_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        optimizer = Adam(model.parameters(), lr=2e-5)
        loss_fn = nn.BCEWithLogitsLoss()

        epochs = 5
        for epoch in range(epochs):
            print(f"\nStarting Epoch {epoch + 1}/{epochs}")
            start_time = time.time()

            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            print(f"Epoch {epoch + 1} completed. Training Loss: {train_loss:.4f}")

            val_loss, precision, recall, f1 = validate_epoch(model, val_loader, loss_fn, device)
            print(f"Validation - Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            print(f"Time taken for epoch: {time.time() - start_time:.2f} seconds")

        torch.save(model.state_dict(), "bert_model.pth")
        print("Model training completed and saved as 'bert_model.pth'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
