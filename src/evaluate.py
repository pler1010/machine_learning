import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, jaccard_score

# 读取和准备数据
filepath=sys.argv[1]
df = pd.read_csv(f'{filepath}/selected_data.txt')
X = df.iloc[:, :-1].values
y = LabelEncoder().fit_transform(df['type'])

# 标准化
X_scaled = StandardScaler().fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 训练分类器
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv_acc = cross_val_score(clf, X_scaled, y, cv=5).mean()

print(f"测试准确率: {acc:.4f}")
print(f"交叉验证准确率: {cv_acc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

clf2 = LogisticRegression(max_iter=1000, random_state=43)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)
labels=set(y_test)
for item in labels:
    res=jaccard_score(
        y_pred==item,
        y_pred2==item,
        average='binary'
    )
    print(item,"Jaccard",res)