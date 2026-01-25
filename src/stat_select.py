import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums

filepath = sys.argv[1]
k = int(sys.argv[2])

# 读取数据
df = pd.read_csv(f'{filepath}/clean_data.txt')
features = [col for col in df.columns if col != df.columns[-1]]
label_col = df.columns[-1]

# 获取所有亚型
subtypes = df[label_col].unique()
selected_features = set()

# 对每个亚型进行一对多比较
for subtype in subtypes:
    mask = df[label_col] == subtype
    group1 = df.loc[mask, features]
    group2 = df.loc[~mask, features]
    
    # Wilcoxon秩和检验
    p_values = []
    log2_fc = []
    
    for gene in features:
        stat, p_val = ranksums(group1[gene], group2[gene])
        p_values.append(p_val)
        
        mean1 = np.mean(np.maximum(group1[gene],0))
        mean2 = np.mean(np.maximum(group2[gene],0))
        log2_fc.append(np.log2((mean1 + 1e-10) / (mean2 + 1e-10)))
    
    # 创建结果DataFrame并排序
    result_df = pd.DataFrame({
        'Gene': features,
        'log2FC': log2_fc,
        'p_value': p_values
    })
    
    # 按log2FC绝对值排序，取前k个
    result_df['abs_log2FC'] = np.abs(result_df['log2FC'])
    top_genes = result_df.sort_values('abs_log2FC', ascending=False).head(k)
    
    selected_features.update(top_genes['Gene'])
    
    # 绘制火山图
    plt.figure(figsize=(8, 6))
    
    # 显著点
    sig_mask = result_df['p_value'] < 0.9
    plt.scatter(result_df.loc[sig_mask, 'log2FC'], 
                -np.log10(result_df.loc[sig_mask, 'p_value'] + 1e-10),
                alpha=0.7, s=15, c='blue')
    
    # 非显著点
    ns_mask = ~sig_mask
    plt.scatter(result_df.loc[ns_mask, 'log2FC'], 
                -np.log10(result_df.loc[ns_mask, 'p_value'] + 1e-10),
                alpha=0.5, s=10, c='gray')
    
    plt.axhline(y=-np.log10(0.9), color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('log2 Fold Change')
    plt.ylabel('-log10(p-value)')
    plt.title(f'{subtype} vs Others')
    plt.tight_layout()
    
    plt.savefig(f'{filepath}/../fig/volcano_{subtype}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 保存选择的数据
print('selected_counts:',len(selected_features))
selected_df = df[list(selected_features) + [label_col]]
selected_df.to_csv(f'{filepath}/selected_data.txt', index=False)