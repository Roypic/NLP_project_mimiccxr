
# ===============================
# 0. Imports
# ===============================
import json
import os

from matplotlib import  pyplot as plt
import numpy as np
import pandas as pd
from deep_utils import DirUtils, JsonUtils
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


CSV_PATH = "sampled_1000_data.csv"
TEXT_COL = "radgraph_json_info"  # column with full report text
LABEL_COL = "class_list"  # column with 75-length label vector

NUM_LABELS = 75  # 75 outcomes per report
NUM_CLASSES = 3  # {-1,0,1} → {0,1,2}
MAX_LEN = 512
BATCH_SIZE = 4
LR = 2e-5
EPOCHS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Load data & map labels

df = pd.read_csv(CSV_PATH)
# df = df[:500]


def parse_label_vector(s):
    """
    s: string like '[-1.0, 1.0, 0.0, ...]'.
    Returns a list[int] length 75 with values in {0,1,2},
    mapping -1→0, 0→1, 1→2.
    """
    arr = np.array(json.loads(s))  # e.g. shape (75,)
    if len(arr) != NUM_LABELS:
        raise ValueError(f"Expected {NUM_LABELS} labels, got {len(arr)}")
    mapped = arr.astype(int) + 1  # -1→0, 0→1, 1→2
    return mapped.tolist()


df["label_vector"] = df[LABEL_COL].apply(parse_label_vector)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42, shuffle=True)

print("Train size:", len(train_df), "Val size:", len(val_df), "Test size:", len(test_df))

# Tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Dataset

class RadiologyReportDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512):
        self.texts = df[TEXT_COL].tolist()
        self.labels = df["label_vector"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = torch.tensor(self.labels[idx], dtype=torch.long)  # (75,)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels
        }


train_dataset = RadiologyReportDataset(train_df, tokenizer, MAX_LEN)
val_dataset = RadiologyReportDataset(val_df, tokenizer, MAX_LEN)
test_dataset = RadiologyReportDataset(test_df, tokenizer, MAX_LEN)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Model: ClinicalBERT + 75×3 head

class MultiHeadClinicalBERT(nn.Module):
    def __init__(self, model_name, num_labels=75, num_classes=3, dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.num_classes = num_classes

        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # usually 768

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels * num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # CLS embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, H)
        cls_emb = self.dropout(cls_emb)

        logits = self.classifier(cls_emb)  # (B, 75*3)
        logits = logits.view(-1, self.num_labels, self.num_classes)  # (B, 75, 3)

        loss = None
        if labels is not None:
            # labels: (B, 75) with integers in {0,1,2}
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_classes),  # (B*75, 3)
                labels.view(-1)  # (B*75,)
            )
        return loss, logits


model = MultiHeadClinicalBERT(MODEL_NAME, NUM_LABELS, NUM_CLASSES)
model.to(DEVICE)

# Optimizer & scheduler

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# Training & validation

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # (B, 75)

        optimizer.zero_grad()
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)  # (B, 75)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return avg_loss, all_preds, all_labels

def multi_f1_score(y_true, y_pred):
    f1_scores_macro = []
    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true[:, i], y_pred[:, i], average='macro')
        f1_scores_macro.append(f1)

    mean_f1_macro = np.mean(f1_scores_macro)
    return mean_f1_macro
# ===============================
# 8. Run training
# ===============================
val_loss_values = []
train_loss_values = []
f1_score_values = []
best_f1_score = 0
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
    val_loss, val_preds, val_labels = eval_one_epoch(model, val_loader, DEVICE)
    val_loss_values.append(val_loss)
    train_loss_values.append(train_loss)
    print(val_labels.shape, val_preds.shape, np.unique(val_preds))
    val_f1_score = multi_f1_score(val_labels, val_preds)
    f1_score_values.append(val_f1_score)
    if val_f1_score > best_f1_score:
        torch.save(model.state_dict(), "clinicalbert_75x3_best.pt")
        best_f1_score = val_f1_score

    torch.save(model.state_dict(), "clinicalbert_75x3_last.pt")
    # print("Model saved to clinicalbert_75x3_last.pt")
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val   loss: {val_loss:.4f}")
    print(f"  Val   f1_score: {val_f1_score:.4f}")


plt2 = plt.twinx()

plt.plot(val_loss_values, color='red', label='Validation Loss')
plt.plot(train_loss_values, color='blue', label='Training Loss')
plt.legend(loc='upper left')
# Add labels and title for clarity
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt2.plot(f1_score_values, color='Purple', label='F1 score')
plt2.legend(loc='upper right')


plt2.set_ylabel("f1_score")
plt.title('Training vs Validation Loss')
plt.tight_layout()
plt.grid(True, alpha=0.3)

plt.savefig("loss_just_finetune.png")

test_loss, test_preds, test_labels = eval_one_epoch(model, test_loader, DEVICE)
JsonUtils.dump("test_preds.json", test_preds.tolist())
JsonUtils.dump("test_labels.json", test_labels.tolist())
test_f1_score = multi_f1_score(test_labels, test_preds)
DirUtils.write_txt("test_f1_score.txt", [f'{test_f1_score}'], mode="w")
# Save model

