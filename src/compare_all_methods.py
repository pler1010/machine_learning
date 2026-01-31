import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import ranksums
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def minmax_norm(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def evaluate_feature_set(X_train, y_train, X_val, y_val, feature_indices, name):
    if len(feature_indices) == 0:
        return 0.0
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train[:, feature_indices], y_train)
    acc = clf.score(X_val[:, feature_indices], y_val)
    print(f"  [{name}] Validation Accuracy: {acc:.4f}")
    return acc

def run_stat_select(X, y, top_n):
    f_scores, p_vals = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores)
    top_indices = np.argsort(f_scores)[::-1][:top_n]
    return top_indices

def run_ml_select(X, y, top_n):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return np.argsort(rf.feature_importances_)[::-1][:top_n]

def run_dl_select(X, y, top_n):
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
    mlp.fit(X, y)
    importances = np.sum(np.abs(mlp.coefs_[0]), axis=1)
    return np.argsort(importances)[::-1][:top_n]

def run_cluster_hybrid(X, y, top_n):
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores)
    norm_f = minmax_norm(f_scores)
    
    top_k_indices = np.argsort(f_scores)[::-1][:3000]
    
    pca = PCA(n_components=30, random_state=42)
    X_pca = pca.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=5).fit(X_pca)
    labels = kmeans.labels_
    
    cons_scores = np.zeros(X.shape[1])
    
    for idx in top_k_indices:
        col = X[:, idx]
        ss_within = 0.0
        for k in range(4):
            mask = labels == k
            if np.sum(mask) > 0:
                ss_within += np.sum((col[mask] - np.mean(col[mask]))**2)
        
        ss_total = np.sum((col - np.mean(col))**2)
        if ss_within > 0:
            cons_scores[idx] = (ss_total - ss_within) / ss_within
            
    norm_c = minmax_norm(cons_scores)
    
    alpha = 0.9
    final_scores = alpha * norm_f + (1 - alpha) * norm_c
    return np.argsort(final_scores)[::-1][:top_n]

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_all_methods.py <datapath> [top_n]")
        sys.exit(1)

    filepath = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
    
    clean_path = os.path.join(filepath, "clean_data.txt")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"找不到 {clean_path}")

    fig_dir = os.path.join(filepath, "..", "fig")
    ensure_dir(fig_dir)

    print(f"[Comparison] Loading data from {clean_path}...")
    df = pd.read_csv(clean_path)
    label_col = df.columns[-1]
    feature_names = np.array([c for c in df.columns if c != label_col])
    
    X_raw = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = LabelEncoder().fit_transform(df[label_col].values)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"[Comparison] Data Split: Train={X_train.shape}, Val={X_val.shape}")
    print(f"[Comparison] Evaluating methods on Independent Validation Set (LR Accuracy)...")
    
    results = {}
    times = {}
    
    print("\n1. Running Stat Select (ANOVA F-test)...")
    t0 = time.time()
    idx_stat = run_stat_select(X_train_scaled, y_train, top_n)
    times['Stat'] = time.time() - t0
    results['Stat (ANOVA)'] = evaluate_feature_set(X_train_scaled, y_train, X_val_scaled, y_val, idx_stat, "Stat")
    
    print("\n2. Running ML Select (RandomForest)...")
    t0 = time.time()
    idx_ml = run_ml_select(X_train_scaled, y_train, top_n)
    times['ML'] = time.time() - t0
    results['ML (RF)'] = evaluate_feature_set(X_train_scaled, y_train, X_val_scaled, y_val, idx_ml, "ML")
    
    print("\n3. Running DL Select (MLP Weights)...")
    t0 = time.time()
    idx_dl = run_dl_select(X_train_scaled, y_train, top_n)
    times['DL'] = time.time() - t0
    results['DL (MLP)'] = evaluate_feature_set(X_train_scaled, y_train, X_val_scaled, y_val, idx_dl, "DL")
    
    print("\n4. Running Cluster Hybrid (Ours)...")
    t0 = time.time()
    idx_ours = run_cluster_hybrid(X_train_scaled, y_train, top_n)
    times['Cluster'] = time.time() - t0
    results['Cluster (Hybrid)'] = evaluate_feature_set(X_train_scaled, y_train, X_val_scaled, y_val, idx_ours, "Cluster")
    
    print("\n5. Running Random Baseline...")
    np.random.seed(42)
    idx_rand = np.random.choice(X_train.shape[1], top_n, replace=False)
    results['Random'] = evaluate_feature_set(X_train_scaled, y_train, X_val_scaled, y_val, idx_rand, "Random")

    print("\n[Output] Generating comparison plots...")
    
    methods = list(results.keys())
    accs = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accs, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f'])
    plt.title("Method Comparison on Independent Validation Set", fontsize=14)
    plt.ylabel("Validation Accuracy (Logistic Regression)")
    plt.ylim(0, 1.05)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "final_method_comparison_acc.png"), dpi=300)
    plt.close()
    
    time_methods = list(times.keys())
    time_vals = list(times.values())
    
    plt.figure(figsize=(8, 5))
    plt.bar(time_methods, time_vals, color='purple', alpha=0.7)
    plt.title("Computation Time Comparison", fontsize=14)
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "final_method_comparison_time.png"), dpi=300)
    plt.close()
    
    genes_stat = feature_names[idx_stat]
    genes_ml = feature_names[idx_ml]
    genes_dl = feature_names[idx_dl]
    genes_cluster = feature_names[idx_ours]
    
    comparisons = [
        ("Stat", genes_stat),
        ("ML", genes_ml),
        ("DL", genes_dl)
    ]
    
    print("\n=== Overlap Analysis (vs Cluster Hybrid) ===")
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, genes_other) in enumerate(comparisons):
        intersection = set(genes_cluster) & set(genes_other)
        inter_size = len(intersection)
        
        print(f"Cluster vs {name}: Overlap = {inter_size}/{top_n}")
        
        plt.subplot(1, 3, i+1)
        bars = plt.bar(["Overlap", "Non-overlap"], 
                       [inter_size, top_n - inter_size], 
                       color=['#2ca02c', '#d62728'])
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{int(height)}', ha='center', va='bottom', fontsize=12)
            
        plt.title(f"vs {name}", fontsize=14)
        plt.ylabel("Gene Count" if i == 0 else "")
        plt.ylim(0, top_n * 1.15)
        
    plt.suptitle(f"Cluster Hybrid Feature Overlap Analysis (Top{top_n})", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "final_combined_overlap_bar.png"), dpi=300)
    plt.close()

    for name, genes_other in comparisons:
        try:
            from matplotlib_venn import venn2
            plt.figure(figsize=(5, 5))
            venn2([set(genes_cluster), set(genes_other)], set_labels=('Cluster', name))
            plt.title(f"Cluster vs {name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"final_venn_cluster_vs_{name.lower()}.png"), dpi=300)
            plt.close()
        except ImportError:
            pass

    print(f"[Output] Plots saved to {fig_dir}")
    print("\n=== Final Summary ===")
    for m, acc in results.items():
        print(f"{m:<20}: {acc:.4f}")

if __name__ == "__main__":
    main()
