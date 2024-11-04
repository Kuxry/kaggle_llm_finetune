import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import optuna
import os

# 确保只使用 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 读取训练数据
train_data = pd.read_csv('mental/train.csv')  # 确保文件路径正确

# 去除列名的多余空格
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

# 特征工程示例：创建新的特征
train_data['Age_WorkPressure'] = train_data['Age'] * train_data['Work Pressure']

# 更新特征列表
features.append('Age_WorkPressure')

# 超参数优化
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_seed': 42,
        'task_type': 'GPU',
        'devices': '0',
        'logging_level': 'Silent',
    }
    model = CatBoostClassifier(**params)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    for train_idx, val_idx in skf.split(train_data[features], train_data[target]):
        X_train, X_val = train_data.iloc[train_idx][features], train_data.iloc[val_idx][features]
        y_train, y_val = train_data.iloc[train_idx][target], train_data.iloc[val_idx][target]
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
        preds = model.predict(X_val)
        accuracies.append(accuracy_score(y_val, preds))
    return sum(accuracies) / len(accuracies)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
best_params = study.best_params

# 使用最佳参数训练模型
best_params.update({
    'task_type': 'GPU',
    'devices': '0'
})
model = CatBoostClassifier(**best_params)
model.fit(train_data[features], train_data[target])

# 加载测试数据集
test_data = pd.read_csv('mental/test.csv')
test_data.columns = test_data.columns.str.strip()  # 去除列名的多余空格

# 填充测试数据中的缺失值
for col in cat_features:
    test_data[col].fillna('Unknown', inplace=True)
for col in num_features:
    test_data[col].fillna(train_data[col].median(), inplace=True)

# 创建新的特征
test_data['Age_WorkPressure'] = test_data['Age'] * test_data['Work Pressure']

# 预测
test_predictions = model.predict(test_data[features])

# 创建结果 DataFrame，确保列名为 "id" 和 "Depression"
output = pd.DataFrame({
    'id': test_data['id'],
    'Depression': test_predictions
})

# 保存预测结果
output.to_csv('submission.csv', index=False, sep='\t')
print("预测结果已保存到 submission.csv")
