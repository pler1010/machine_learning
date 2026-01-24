import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv('clean_data.txt')
X = df.iloc[:, :-1].values
y = df['type'].values

# 标准化和PCA
X_scaled = StandardScaler().fit_transform(X)
X_pca = PCA(n_components=2).fit_transform(X_scaled)
# X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)

# 绘制结果
plt.figure(figsize=(10, 8))
for label in np.unique(y):
    plt.scatter(X_pca[y==label, 0], X_pca[y==label, 1], label=label, alpha=0.7, s=100)

plt.title('PCA 2D Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Type')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()