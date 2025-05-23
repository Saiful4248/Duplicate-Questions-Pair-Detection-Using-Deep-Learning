# -*- coding: utf-8 -*-
"""Bert.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sym1xQLMFuNzYdRd1CjUAUt7FZR5_yUI
"""

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import time
# Import AdamW from torch.optim instead of transformers
from torch.optim import AdamW

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load dataset
df = pd.read_csv("/content/drive/MyDrive/Duplicate Question Pair/train.csv")
print(f"Dataset loaded with {len(df)} samples")

# Preprocessing
df = df.dropna()  # Remove null values
df = df.drop(['id', 'qid1', 'qid2'], axis=1, errors='ignore')  # Remove unnecessary columns

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_questions(q1, q2, max_length=128):
    return tokenizer.encode_plus(
        q1,
        text_pair=q2,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

# prompt: I want take 50000 pairs only,give the code basis on this

# Take only the first 50,000 pairs
df = df.head(50000)
print(f"Using {len(df)} samples")

# Prepare dataset
input_ids = []
attention_masks = []
token_type_ids = []
labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    encoded = tokenize_questions(str(row['question1']), str(row['question2']))
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])
    token_type_ids.append(encoded['token_type_ids'])
    labels.append(row['is_duplicate'])

# Convert to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
token_type_ids = torch.cat(token_type_ids, dim=0)
labels = torch.tensor(labels)

# Create dataset
dataset = TensorDataset(input_ids, token_type_ids, attention_masks, labels)

# Split into train and validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders
BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize BERT model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training function
def train_model():
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch + 1}/{epochs} ========")

        # Training
        model.train()
        total_train_loss = 0
        train_preds = []
        train_true = []

        for batch in tqdm(train_dataloader, desc="Training"):
            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=b_token_type_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Get predictions
            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            train_preds.extend(preds)
            train_true.extend(b_labels.cpu().numpy())

        # Calculate training metrics
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds)

        print(f"\nTraining Loss: {avg_train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Training F1 Score: {train_f1:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_true = []

        for batch in tqdm(val_dataloader, desc="Validation"):
            b_input_ids = batch[0].to(device)
            b_token_type_ids = batch[1].to(device)
            b_attention_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=b_token_type_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )

            loss = outputs.loss
            total_val_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()
            val_preds.extend(preds)
            val_true.extend(b_labels.cpu().numpy())

        # Calculate validation metrics  -- This block was previously indented incorrectly. Moved outside training loop.
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_acc = accuracy_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds)
        val_recall = recall_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds)

        print(f"\nValidation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Precision: {val_precision:.4f}")
        print(f"Validation Recall: {val_recall:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")

# Start training
train_model()

# Save model
torch.save(model.state_dict(), '/content/drive/MyDrive/Duplicate Question Pair/bert_model.pth')
print("Model saved successfully!")

# prompt: show me plt epoch vs loss

import matplotlib.pyplot as plt

# Assuming you have a list of losses for each epoch during training
# Replace this with your actual loss values
losses = [0.5, 0.4, 0.3]  # Example losses for 3 epochs

# Create the plot
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs. Loss")
plt.grid(True)
plt.show()

# prompt: i want to ditect duplicate question pair,we have 2 question q1,q2.and then ditect its duplicate or not duplicate

import torch

# Load the saved model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
model.load_state_dict(torch.load('/content/drive/MyDrive/Duplicate Question Pair/bert_model.pth'))

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


def predict_duplicate(q1, q2):
    encoded = tokenize_questions(q1, q2)
    input_ids = encoded['input_ids'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 1:
        return "Duplicate"
    else:
        return "Not Duplicate"

# Example usage
q1 = "What are the benefits of drinking green tea?"
q2 = "Why should I drink green tea regularly?"
prediction = predict_duplicate(q1, q2)
print(f"The questions are: {prediction}")

q1 = " What is the difference between Java and JavaScript?"
q2 = "How is JavaScript different from Java?"
prediction = predict_duplicate(q1, q2)
print(f"The questions are: {prediction}")

