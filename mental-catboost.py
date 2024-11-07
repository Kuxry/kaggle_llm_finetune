import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from category_encoders import TargetEncoder

# 确保只使用 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 读取训练数据
train_data = pd.read_csv('mental/train.csv')  # 确保文件路径正确
train_data.columns = train_data.columns.str.strip()  # 去除列名的多余空格

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
train_data[['City_encoded', 'Profession_encoded']] = encoder.fit_transform(train_data[['City', 'Profession']], train_data[target])
features += ['City_encoded', 'Profession_encoded']

# 特征工程 - 分箱处理
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
train_data['Age_binned'] = binner.fit_transform(train_data[['Age']])
features.append('Age_binned')

# 定义模型参数
# 定义模型参数
catboost_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate':0.05717731252821768,
    'iterations': 1339,
    'depth': 4,
    'random_strength':0.7219379325298366,
    'l2_leaf_reg': 0.8834758662376494,
    'random_seed': 42,
    'verbose': False,
    'task_type': 'GPU',
    'bagging_temperature': 0.29587141149836876,
    'border_count': 214
}

# 计算 Overall Train ACCURACY
model = CatBoostClassifier(**catboost_params)
model.fit(train_data[features], train_data[target], cat_features=cat_features)
train_predictions = model.predict(train_data[features])
train_accuracy = accuracy_score(train_data[target], train_predictions)
print(f"Overall Train ACCURACY: {train_accuracy:.4f}")

# 计算 Overall OOF ACCURACY (交叉验证)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train_data))
for train_idx, val_idx in skf.split(train_data[features], train_data[target]):
    X_train, X_val = train_data.iloc[train_idx][features], train_data.iloc[val_idx][features]
    y_train, y_val = train_data.iloc[train_idx][target], train_data.iloc[val_idx][target]

    model = CatBoostClassifier(**catboost_params)
    model.fit(X_train, y_train, cat_features=cat_features)
    val_preds = model.predict(X_val)
    oof_predictions[val_idx] = val_preds

oof_accuracy = accuracy_score(train_data[target], oof_predictions)
print(f"Overall OOF ACCURACY: {oof_accuracy:.4f}")



# 加载测试数据集并进行相同的预处理
test_data = pd.read_csv('mental/test.csv')
test_data.columns = test_data.columns.str.strip()
for col in cat_features:
    test_data[col].fillna('Unknown', inplace=True)
for col in num_features:
    test_data[col].fillna(train_data[col].median(), inplace=True)

# 创建测试数据的新特征
test_data['Age_WorkPressure'] = test_data['Age'] * test_data['Work Pressure']
test_data['Academic_WorkPressure'] = test_data['Academic Pressure'] * test_data['Work Pressure']
test_data[['City_encoded', 'Profession_encoded']] = encoder.transform(test_data[['City', 'Profession']])
test_data['Age_binned'] = binner.transform(test_data[['Age']])

# 使用训练好的模型进行预测
test_predictions = model.predict(test_data[features])

# 创建结果 DataFrame，确保列名为 "id" 和 "Depression"
output = pd.DataFrame({
    'id': test_data['id'],
    'Depression': test_predictions
})

# 保存预测结果
output.to_csv('submission.csv', index=False)
print("预测结果已保存到 submission.csv")
