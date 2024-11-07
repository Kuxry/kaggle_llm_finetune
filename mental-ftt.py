import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import KBinsDiscretizer
from category_encoders import TargetEncoder
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
import rtdl

# 确保只使用 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 读取训练数据
train_data = pd.read_csv('mental/train.csv')
train_data.columns = train_data.columns.str.strip()

# 定义特征和目标
features = [
    'Gender', 'Age', 'City', 'Working Professional or Student', 'Profession',
    'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
    'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 'Degree',
    'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
    'Financial Stress', 'Family History of Mental Illness'
]
target = 'Depression'

# 填充训练数据中的缺失值
cat_features = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                'Sleep Duration', 'Dietary Habits', 'Degree',
                'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in cat_features:
    train_data[col].fillna('Unknown', inplace=True)
num_features = list(set(features) - set(cat_features))
for col in num_features:
    train_data[col].fillna(train_data[col].median(), inplace=True)

# 特征工程 - 创建交互特征
train_data['Age_WorkPressure'] = train_data['Age'] * train_data['Work Pressure']
train_data['Academic_WorkPressure'] = train_data['Academic Pressure'] * train_data['Work Pressure']
features += ['Age_WorkPressure', 'Academic_WorkPressure']

# 特征工程 - 目标编码
encoder = TargetEncoder(cols=['City', 'Profession'])
train_data[['City_encoded', 'Profession_encoded']] = encoder.fit_transform(train_data[['City', 'Profession']],
                                                                           train_data[target])
features += ['City_encoded', 'Profession_encoded']

# 特征工程 - 分箱处理
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
train_data['Age_binned'] = binner.fit_transform(train_data[['Age']])
features.append('Age_binned')

# 准备数据
X_num = train_data[num_features + ['Age_WorkPressure', 'Academic_WorkPressure', 'City_encoded', 'Profession_encoded',
                                   'Age_binned']].astype('float32')
X_cat = train_data[cat_features].apply(lambda x: x.astype('category').cat.codes).astype('int64')
X = pd.concat([X_num, X_cat], axis=1)
y = train_data[target].astype('float32')

# 获取类别特征的基数
cat_cardinalities = [X_cat[col].nunique() for col in cat_features]

# 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = rtdl.FTTransformer.make_default(
    n_num_features=len(X_num.columns),
    cat_cardinalities=cat_cardinalities,
    d_out=1
).to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = F.binary_cross_entropy_with_logits
scaler = GradScaler()  # 定义缩放器

# 将数据转换为 TensorDataset 并定义 DataLoader
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)
train_dataset = TensorDataset(X_tensor, y_tensor)
batch_size = 256 # 将批次大小设置为 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        with autocast():
            y_pred = model(X_batch[:, :len(X_num.columns)], X_batch[:, len(X_num.columns):].long())
            loss = loss_fn(y_pred.squeeze(), y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

# 交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train_data))
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = rtdl.FTTransformer.make_default(
        n_num_features=len(X_num.columns),
        cat_cardinalities=cat_cardinalities,
        d_out=1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with autocast():
                y_pred = model(X_batch[:, :len(X_num.columns)], X_batch[:, len(X_num.columns):].long())
                loss = loss_fn(y_pred.squeeze(), y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor[:, :len(X_num.columns)],
                          X_val_tensor[:, len(X_num.columns):].long()).squeeze().cpu().numpy()
        oof_predictions[val_idx] = (val_preds > 0.5).astype(int)

oof_accuracy = accuracy_score(train_data[target], oof_predictions)
print(f"Overall OOF ACCURACY: {oof_accuracy:.4f}")
