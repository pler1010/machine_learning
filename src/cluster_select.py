import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import f_classif
from sklearn.cluster import SpectralClustering

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

def calculate_variance_ratio(X_gene, labels):
    unique_labels = np.unique(labels)
    overall_mean = np.mean(X_gene)
    
    ss_between = 0.0
    ss_within = 0.0
    
    for label in unique_labels:
        mask = labels == label
        cluster_data = X_gene[mask]
        n_cluster = len(cluster_data)
        
        if n_cluster == 0:
            continue
            
        cluster_mean = np.mean(cluster_data)
        ss_between += n_cluster * (cluster_mean - overall_mean) ** 2
        ss_within += np.sum((cluster_data - cluster_mean) ** 2)
        
    if ss_within == 0:
        return 0.0
        
    return ss_between / ss_within

def evaluate_features(X_train, y_train, X_test, y_test, feature_indices, name="Method"):
    if len(feature_indices) == 0:
        return 0.0
        
    X_tr_sub = X_train[:, feature_indices]
    X_te_sub = X_test[:, feature_indices]
    
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_tr_sub, y_train)
    y_pred = clf.predict(X_te_sub)
    
    return accuracy_score(y_test, y_pred)

def plot_top20_bar(feature_names, scores, title, save_path):
    idx = np.argsort(scores)[::-1][:20]
    genes = [feature_names[i] for i in idx][::-1]
    vals = scores[idx][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(genes, vals, color='teal', alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel("Consistency Score (Normalized)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_performance_comparison(results, save_path):
    methods = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, accuracies, color=['#4c72b0', '#55a868', '#c44e52'])
    plt.title("Performance Comparison on Validation Set (KNN)", fontsize=14)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python cluster_select.py <datapath> [top_n]")
        sys.exit(1)

    filepath = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
    
    clean_path = os.path.join(filepath, "clean_data.txt")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"找不到 {clean_path}")

    fig_dir = os.path.join(filepath, "..", "fig")
    ensure_dir(fig_dir)

    print(f"[Cluster Select] Loading data from {clean_path}...")
    df = pd.read_csv(clean_path)
    label_col = df.columns[-1]
    feature_names = [c for c in df.columns if c != label_col]
    
    X_raw = df[feature_names].values.astype(np.float32)
    y_raw = LabelEncoder().fit_transform(df[label_col].values)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    n_samples, n_features = X_train.shape
    k = 4
    
    print(f"[Cluster Select] Starting consistency-based feature selection...")
    start_time = time.time()
    
    print(f"[Cluster Select] Hybrid Strategy: Supervised F-score + Unsupervised Cluster Consistency")
    
    print(f"[Cluster Select] Step 1: Calculating Supervised F-statistics...")
    f_scores, p_values = f_classif(X_train_scaled, y_train)
    f_scores = np.nan_to_num(f_scores)
    norm_f_scores = minmax_norm(f_scores)
    
    print(f"[Cluster Select] Step 2: Calculating Unsupervised Cluster Consistency...")
    
    pca_50 = PCA(n_components=50, random_state=42)
    X_pca_50 = pca_50.fit_transform(X_train_scaled)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_pca_50)
    
    agg = AgglomerativeClustering(n_clusters=k)
    labels_agg = agg.fit_predict(X_pca_50)
    
    spectral = SpectralClustering(n_clusters=k, assign_labels='discretize', random_state=42, affinity='nearest_neighbors')
    labels_spectral = spectral.fit_predict(X_pca_50)
    
    print(f"  - Calculating Variance Ratio Scores based on Pseudo-labels...")
    
    top_f_indices = np.argsort(f_scores)[::-1][:3000]
    
    scores_v1 = np.zeros(n_features)
    scores_v2 = np.zeros(n_features)
    scores_v3 = np.zeros(n_features)
    
    for i in top_f_indices:
        scores_v1[i] = calculate_variance_ratio(X_train_scaled[:, i], labels_kmeans)
        scores_v2[i] = calculate_variance_ratio(X_train_scaled[:, i], labels_agg)
        scores_v3[i] = calculate_variance_ratio(X_train_scaled[:, i], labels_spectral)
        
    norm_v1 = minmax_norm(scores_v1)
    norm_v2 = minmax_norm(scores_v2)
    norm_v3 = minmax_norm(scores_v3)
    
    consistency_score = (norm_v1 + norm_v2 + norm_v3) / 3.0
    
    alpha = 0.9
    print(f"[Cluster Select] Step 3: Fusing Scores (alpha={alpha})...")
    final_scores = alpha * norm_f_scores + (1 - alpha) * consistency_score
    
    top_indices = np.argsort(final_scores)[::-1][:top_n]
    top_genes = [feature_names[i] for i in top_indices]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"[Cluster Select] Algorithm finished in {elapsed_time:.2f} seconds.")
    print(f"Top 10 Genes: {top_genes[:10]}")

    print("\n[Evaluation] Comparing with baseline methods on Validation Set...")
    
    def evaluate_with_lr(X_train, y_train, X_val, y_val, indices):
        if len(indices) == 0: return 0.0
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train[:, indices], y_train)
        return clf.score(X_val[:, indices], y_val)

    acc_ours_knn = evaluate_features(X_train_scaled, y_train, X_val_scaled, y_val, top_indices)
    acc_ours_lr = evaluate_with_lr(X_train_scaled, y_train, X_val_scaled, y_val, top_indices)
    
    top_f_indices_baseline = np.argsort(f_scores)[::-1][:top_n]
    acc_f_test = evaluate_features(X_train_scaled, y_train, X_val_scaled, y_val, top_f_indices_baseline)
    
    variances = np.var(X_train_scaled, axis=0)
    top_var_indices = np.argsort(variances)[::-1][:top_n]
    acc_var = evaluate_features(X_train_scaled, y_train, X_val_scaled, y_val, top_var_indices)
    
    np.random.seed(42)
    random_indices = np.random.choice(n_features, top_n, replace=False)
    acc_random = evaluate_features(X_train_scaled, y_train, X_val_scaled, y_val, random_indices)
    
    print(f"  - Our Method (Hybrid) Accuracy (LR):  {acc_ours_lr:.4f}")
    
    print("\n[Output] Preparing selected data...")
    selected_df = df[top_genes + [label_col]].copy()
    
    check_X = selected_df.iloc[:, :-1].values
    check_y = LabelEncoder().fit_transform(selected_df.iloc[:, -1].values)
    check_X_scaled = StandardScaler().fit_transform(check_X)
    cx_train, cx_test, cy_train, cy_test = train_test_split(check_X_scaled, check_y, test_size=0.2, random_state=42)
    check_clf = LogisticRegression(max_iter=1000, random_state=42)
    check_clf.fit(cx_train, cy_train)
    check_acc = check_clf.score(cx_test, cy_test)
    print(f"  [Self-Check] Accuracy on selected_df (Simulating evaluate.py): {check_acc:.4f}")
    
    out_path = os.path.join(filepath, "selected_data.txt")
    selected_df.to_csv(out_path, index=False)
    print(f"[Output] Saved selected data to {out_path}")
    
    score_df = pd.DataFrame({
        "gene": feature_names,
        "score_kmeans": norm_v1,
        "score_agg": norm_v2,
        "score_pca": norm_v3,
        "consistency_score": final_scores
    }).sort_values("consistency_score", ascending=False)
    score_out_path = os.path.join(filepath, "cluster_importance_scores.csv")
    score_df.to_csv(score_out_path, index=False)
    
    plot_top20_bar(feature_names, final_scores, 
                  "Cluster Consistency Score Top20",
                  os.path.join(fig_dir, "cluster_top20_score.png"))
    
    ml_score_path = os.path.join(filepath, "ml_importance_scores.csv")
    dl_score_path = os.path.join(filepath, f"dl_importance_scores_abs.csv")
    
    targets = [("ML", ml_score_path), ("DL", dl_score_path)]
    
    for label, path in targets:
        if os.path.exists(path):
            try:
                other_df = pd.read_csv(path)
                if "gene" in other_df.columns:
                    score_col = other_df.select_dtypes(include=[np.number]).columns[-1]
                    other_top = other_df.sort_values(score_col, ascending=False)["gene"].head(top_n).tolist()
                    
                    inter = set(other_top) & set(top_genes)
                    print(f"\n[Overlap Analysis] Cluster vs {label} Top{top_n}")
                    print(f"  - Intersection: {len(inter)}")
                    print(f"  - Genes: {list(inter)[:10]}...")
                    
                    try:
                        from matplotlib_venn import venn2
                        plt.figure(figsize=(6, 5))
                        venn2([set(top_genes), set(other_top)], set_labels=('Cluster', label))
                        plt.title(f"Overlap Cluster vs {label}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(fig_dir, f"cluster_vs_{label.lower()}_venn.png"), dpi=300)
                        plt.close()
                    except ImportError:
                        pass
            except Exception as e:
                print(f"  [Warning] Failed to analyze overlap with {label}: {e}")

    results = {
        "Cluster Hybrid (Ours)": acc_ours_knn,
        "Pure F-test": acc_f_test,
        "High Variance": acc_var,
        "Random": acc_random
    }
    plot_performance_comparison(results, os.path.join(fig_dir, "cluster_performance_comparison.png"))
    
    if len(df) > 500:
        sample_df = df.sample(n=500, random_state=42)
    else:
        sample_df = df
        
    top_gene_data = sample_df.groupby(label_col)[top_genes[:10]].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(top_gene_data, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Top 10 Genes Expression by Subtype (Mean)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cluster_top10_heatmap.png"), dpi=300)
    plt.close()
    
    print("[Output] Visualizations saved to fig/ directory.")

if __name__ == "__main__":
    main()
