import sys
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和绘图样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用通用字体
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

filepath=sys.argv[1]
adata = sc.read_h5ad(f'{filepath}/SC-1_dense.h5ad')

n=adata.shape[0]
m=adata.shape[1]
X=adata.X
gene_names=adata.var_names
labels=np.array(adata.obs['subtype'].values)

print('原始数据')
print('细胞数:'+str(n))
print('基因数:'+str(m))
print('标签分布:')
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))
print('数据稀疏度:'+str(np.mean(X<=0)))

# 计算原始数据中每个细胞的总表达量
cell_total_expression = np.array([np.sum(np.maximum(X[i], 0)) for i in range(n)])

# 创建图形，设置子图布局
fig = plt.figure(figsize=(20, 10))

# 子图1: 原始数据的标签分布
ax1 = plt.subplot(2, 4, 1)
original_label_counts = {}
for label in labels:
    original_label_counts[label] = original_label_counts.get(label, 0) + 1

# 按数量排序
sorted_labels = sorted(original_label_counts.items(), key=lambda x: x[1], reverse=True)
labels_sorted = [item[0] for item in sorted_labels]
counts_sorted = [item[1] for item in sorted_labels]

bars = ax1.bar(range(len(labels_sorted)), counts_sorted, color='skyblue')
ax1.set_title('original labels', fontsize=14, fontweight='bold')
ax1.set_xlabel('subtype')
ax1.set_ylabel('counts')
ax1.set_xticks(range(len(labels_sorted)))
ax1.set_xticklabels(labels_sorted, rotation=45, ha='right')

# 在柱状图上添加数量标注
for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts_sorted),
             f'{count:,}', ha='center', va='bottom', fontsize=10)

# 子图2: 原始数据的细胞总表达量分布
ax2 = plt.subplot(2, 4, 2)
# 使用对数刻度处理表达量数据（因为表达量通常跨度很大）
log_expression = np.log10(cell_total_expression + 1)
hist_data = ax2.hist(log_expression, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
ax2.set_title('original expression quantity', fontsize=14, fontweight='bold')
ax2.set_xlabel('log10(quantity + 1)')
ax2.set_ylabel('counts')
ax2.axvline(x=np.mean(log_expression), color='red', linestyle='--', linewidth=2, 
            label=f'mean: {np.mean(log_expression):.2f}')
ax2.axvline(x=np.median(log_expression), color='blue', linestyle='--', linewidth=2,
            label=f'middle: {np.median(log_expression):.2f}')
ax2.legend(loc='upper right')

# 添加统计信息文本
stats_text = f'min: {np.min(cell_total_expression):.0f}\n'
stats_text += f'max: {np.max(cell_total_expression):.0f}\n'
stats_text += f'mean: {np.mean(cell_total_expression):.1f}\n'
stats_text += f'middle: {np.median(cell_total_expression):.1f}'
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
         verticalalignment='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 子图3: 细胞总表达量的箱线图（按标签分组）
ax3 = plt.subplot(2, 4, 3)
# 准备数据：按标签分组
expression_by_label = {}
for i, label in enumerate(labels):
    if label not in expression_by_label:
        expression_by_label[label] = []
    expression_by_label[label].append(cell_total_expression[i])

# 转换为适合箱线图的格式
boxplot_data = [expression_by_label[label] for label in labels_sorted]
box = ax3.boxplot(boxplot_data, patch_artist=True)
ax3.set_title('original expression quantity', fontsize=14, fontweight='bold')
ax3.set_xlabel('subtype')
ax3.set_ylabel('expression quantity')
ax3.set_xticklabels(labels_sorted, rotation=45, ha='right')
ax3.set_yscale('log')  # 使用对数刻度

# 设置箱线图颜色
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# 子图4: 表达量密度图（核密度估计）
ax4 = plt.subplot(2, 4, 4)
sns.kdeplot(log_expression, ax=ax4, color='purple', linewidth=2, fill=True, alpha=0.3)
ax4.set_title('original expression quantity', fontsize=14, fontweight='bold')
ax4.set_xlabel('log10(quantity + 1)')
ax4.set_ylabel('density')

# 应用筛选条件
n_list=[True for i in range(n)]
m_list=[True for i in range(m)]

for i in range(n):
    if labels[i] not in ['ER+','ER+HER2+','HER2+','TN']:
        n_list[i]=False
    if np.sum(np.maximum(X[i],0)) < 5:
        n_list[i]=False

for i in range(m):
    if np.sum(np.maximum(X[:,i],0)) < 5:
        m_list[i]=False

# 筛选数据
filtered_labels = labels[n_list]
filtered_X = X[n_list][:, m_list]
filtered_n = filtered_X.shape[0]
filtered_m = filtered_X.shape[1]

# 计算筛选后数据的细胞总表达量
filtered_cell_total_expression = np.array([np.sum(np.maximum(filtered_X[i], 0)) for i in range(filtered_n)])

print('\n筛选后数据')
print('细胞数:'+str(filtered_n))
print('基因数:'+str(filtered_m))
print('标签分布:')
filtered_unique, filtered_counts = np.unique(filtered_labels, return_counts=True)
print(dict(zip(filtered_unique, filtered_counts)))
print('数据稀疏度:'+str(np.mean(filtered_X<=0)))
print(f'细胞保留率: {filtered_n/n*100:.2f}%')
print(f'基因保留率: {filtered_m/m*100:.2f}%')

# 子图5: 筛选后数据的标签分布
ax5 = plt.subplot(2, 4, 5)
filtered_label_counts = {}
for label in filtered_labels:
    filtered_label_counts[label] = filtered_label_counts.get(label, 0) + 1

# 确保顺序与原始数据一致
filtered_labels_sorted = []
filtered_counts_sorted = []
for label in ['ER+','ER+HER2+','HER2+','TN']:
    if label in filtered_label_counts:
        filtered_labels_sorted.append(label)
        filtered_counts_sorted.append(filtered_label_counts[label])

bars_filtered = ax5.bar(range(len(filtered_labels_sorted)), filtered_counts_sorted, color='lightgreen')
ax5.set_title('processed labels', fontsize=14, fontweight='bold')
ax5.set_xlabel('subtype')
ax5.set_ylabel('counts')
ax5.set_xticks(range(len(filtered_labels_sorted)))
ax5.set_xticklabels(filtered_labels_sorted, rotation=45, ha='right')

# 在柱状图上添加数量标注
for i, (bar, count) in enumerate(zip(bars_filtered, filtered_counts_sorted)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(filtered_counts_sorted),
             f'{count:,}', ha='center', va='bottom', fontsize=10)

# 子图6: 筛选后数据的细胞总表达量分布
ax6 = plt.subplot(2, 4, 6)
# 使用对数刻度处理表达量数据
filtered_log_expression = np.log10(filtered_cell_total_expression + 1)
ax6.hist(filtered_log_expression, bins=50, color='orange', alpha=0.7, edgecolor='black')
ax6.set_title('processed expression quantity', fontsize=14, fontweight='bold')
ax6.set_xlabel('log10(quantity + 1)')
ax6.set_ylabel('counts')
ax6.axvline(x=np.mean(filtered_log_expression), color='red', linestyle='--', linewidth=2,
            label=f'mean: {np.mean(filtered_log_expression):.2f}')
ax6.axvline(x=np.median(filtered_log_expression), color='blue', linestyle='--', linewidth=2,
            label=f'middle: {np.median(filtered_log_expression):.2f}')
ax6.legend(loc='upper right')

# 添加统计信息文本
filtered_stats_text = f'min: {np.min(filtered_cell_total_expression):.0f}\n'
filtered_stats_text += f'max: {np.max(filtered_cell_total_expression):.0f}\n'
filtered_stats_text += f'mean: {np.mean(filtered_cell_total_expression):.1f}\n'
filtered_stats_text += f'middle: {np.median(filtered_cell_total_expression):.1f}'
ax6.text(0.05, 0.95, filtered_stats_text, transform=ax6.transAxes, 
         verticalalignment='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 子图7: 筛选后细胞总表达量的箱线图（按标签分组）
ax7 = plt.subplot(2, 4, 7)
# 准备数据：按标签分组
filtered_expression_by_label = {}
for i, label in enumerate(filtered_labels):
    if label not in filtered_expression_by_label:
        filtered_expression_by_label[label] = []
    filtered_expression_by_label[label].append(filtered_cell_total_expression[i])

# 转换为适合箱线图的格式
filtered_boxplot_data = [filtered_expression_by_label[label] for label in filtered_labels_sorted]
filtered_box = ax7.boxplot(filtered_boxplot_data, patch_artist=True)
ax7.set_title('processed expression quantity', fontsize=14, fontweight='bold')
ax7.set_xlabel('subtype')
ax7.set_ylabel('quantity')
ax7.set_xticklabels(filtered_labels_sorted, rotation=45, ha='right')
ax7.set_yscale('log')  # 使用对数刻度

# 设置箱线图颜色
for patch, color in zip(filtered_box['boxes'], colors):
    patch.set_facecolor(color)

# 子图8: 筛选前后表达量分布对比
ax8 = plt.subplot(2, 4, 8)
# 绘制密度曲线对比
sns.kdeplot(log_expression, ax=ax8, label='original', color='blue', linewidth=2)
sns.kdeplot(filtered_log_expression, ax=ax8, label='processed', color='red', linewidth=2)
ax8.set_title('processed expression quantity', fontsize=14, fontweight='bold')
ax8.set_xlabel('log10(quantity + 1)')
ax8.set_ylabel('density')
ax8.legend()

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig('data_filtering_analysis.png', dpi=300, bbox_inches='tight')
print('\n图形已保存为: data_filtering_analysis.png')

# 显示图形
plt.show()
print()

# temp_list = []
# for i in range(m):
#     if m_list[i]:
#         temp_list.append((i,np.mean(X[:,i]>0)))
# temp_list.sort(key=lambda x:x[1],reverse=True)
# for i in range(m):
#     m_list[i]=False
# for i in range(500,700):
#     m_list[temp_list[i][0]]=True

with open(f"{filepath}/clean_data.txt","w") as f:
    for i in range(m):
        if m_list[i]:
            f.write(gene_names[i]+",")
    f.write("type\n")

    for i in range(n):
        if not n_list[i]:
            continue
        for j in range(m):
            if m_list[j]:
                f.write(str(X[i][j])+",")
        f.write(str(labels[i])+'\n')