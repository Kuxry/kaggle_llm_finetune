import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 设置路径和参数
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'
SUBMISSION_CSV = 'submission.csv'
BATCH_SIZE = 32
EPOCHS = 10
MODEL_NAME = 'microsoft/deberta-v3-base'
PATIENCE = 3  # 用于提前停止的容忍度

# 1. 定义数据集类
class ChatBotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. 加载分词器和模型
print("加载分词器和预训练模型...")
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
model = DebertaV2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# 使用 GPU（如果可用）
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# 3. 加载和预处理训练数据
print("加载和预处理训练数据...")
train_df = pd.read_csv(TRAIN_CSV)

def get_label(row):
    if row['winner_model_a'] == 1:
        return 'a'
    elif row['winner_model_b'] == 1:
        return 'b'
    else:
        return 'tie'

train_df['label'] = train_df.apply(get_label, axis=1)
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])

train_df['combined_input'] = (
    "Prompt: " + train_df['prompt'] +
    " Response A: " + train_df['response_a'] +
    " Response B: " + train_df['response_b']
)

train_df.dropna(subset=['combined_input', 'label_encoded'], inplace=True)

X = train_df['combined_input'].values
y = train_df['label_encoded'].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 创建数据集和 DataLoader
train_dataset = ChatBotDataset(X_train, y_train, tokenizer)
val_dataset = ChatBotDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. 设置优化器、损失函数和调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# 6. 训练和验证函数
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# 7. 训练模型并引入提前停止
best_accuracy = 0
best_loss = float('inf')
early_stopping_counter = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(X_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_loader,
        loss_fn,
        device,
        len(X_val)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    # 提前停止判断
    if val_loss < best_loss:
        best_loss = val_loss
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'best_model_state.bin')
        early_stopping_counter = 0  # 重置计数器
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= PATIENCE:
        print("验证损失没有提升，提前停止训练。")
        break

# 8. 加载最佳模型
print("加载最佳模型权重...")
model.load_state_dict(torch.load('best_model_state.bin'))

# 9. 预测并生成提交文件
test_df = pd.read_csv(TEST_CSV)
test_df['combined_input'] = (
    "Prompt: " + test_df['prompt'] +
    " Response A: " + test_df['response_a'] +
    " Response B: " + test_df['response_b']
)
test_df.dropna(subset=['combined_input'], inplace=True)
X_test = test_df['combined_input'].values

test_dataset = ChatBotDataset(X_test, [0] * len(X_test), tokenizer)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def get_predictions(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting", leave=True):  # 添加进度条
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions

pred_labels_encoded = get_predictions(model, test_loader, device)
pred_labels = label_encoder.inverse_transform(pred_labels_encoded)

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'winner_model_a': [1 if label == 'a' else 0 for label in pred_labels],
    'winner_model_b': [1 if label == 'b' else 0 for label in pred_labels],
    'winner_model_tie': [1 if label == 'tie' else 0 for label in pred_labels],
})
submission_df.to_csv(SUBMISSION_CSV, index=False)
print(f"提交文件已保存为 {SUBMISSION_CSV}")
