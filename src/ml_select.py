import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def minmax_norm(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """归一化到 [0,1]，避免不同模型重要性尺度不一致"""
    a_min = np.min(arr)
    a_max = np.max(arr)
    if a_max - a_min < eps:
        return np.zeros_like(arr)
    return (arr - a_min) / (a_max - a_min + eps)


def get_xgb_importance(booster, n_features: int, importance_type: str) -> np.ndarray:
    """
    booster.get_score() 返回 dict: {'f0': value, 'f1': value, ...}
    importance_type 支持: gain / cover / weight
    """
    score_dict = booster.get_score(importance_type=importance_type)
    imp = np.zeros(n_features, dtype=np.float64)
    for k, v in score_dict.items():
        if k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            if 0 <= idx < n_features:
                imp[idx] = v
    return imp


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_top20_bar(feature_names, scores, title, save_path):
    """Top20 条形图"""
    idx = np.argsort(scores)[::-1][:20]
    genes = [feature_names[i] for i in idx][::-1]  # 反过来便于横向bar从小到大
    vals = scores[idx][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(genes, vals)
    plt.title(title)
    plt.xlabel("Importance (normalized)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rf_vs_xgb_scatter(feature_names, rf_score, xgb_score, highlight_genes, save_path):
    """RF vs XGB 散点图"""
    plt.figure(figsize=(7, 6))
    plt.scatter(rf_score, xgb_score, s=12, alpha=0.5)
    plt.xlabel("RF Gini Importance (norm)")
    plt.ylabel("XGB Importance Mean (norm)")
    plt.title("RF vs XGB Feature Importance Scatter")

    # 标注重要基因（一般标 Final Top10）
    gene_to_idx = {g: i for i, g in enumerate(feature_names)}
    for g in highlight_genes:
        if g in gene_to_idx:
            i = gene_to_idx[g]
            plt.scatter([rf_score[i]], [xgb_score[i]], s=50)
            plt.text(rf_score[i], xgb_score[i], g, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()




def plot_score_distribution(rf_score, xgb_score, final_score, save_path,
                            bins=80, zoom_max=0.05, eps=1e-12):
    """
    重要性分布直方图
    """
    rf_score = np.asarray(rf_score)
    xgb_score = np.asarray(xgb_score)
    final_score = np.asarray(final_score)

    # 统一 bins 的边界，便于对齐对比
    all_scores = np.concatenate([rf_score, xgb_score, final_score])
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    # 一些统计信息
    def nonzero_cnt(x): return int(np.sum(x > eps))
    def q99(x): return float(np.quantile(x, 0.99))

    rf_nz, xgb_nz, fin_nz = nonzero_cnt(rf_score), nonzero_cnt(xgb_score), nonzero_cnt(final_score)
    rf_q99, xgb_q99, fin_q99 = q99(rf_score), q99(xgb_score), q99(final_score)

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.25)

    # 主图：log-y 直方图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(rf_score, bins=bin_edges, alpha=0.45, label=f"RF(Gini)  nz={rf_nz}")
    ax1.hist(xgb_score, bins=bin_edges, alpha=0.45, label=f"XGB(mean) nz={xgb_nz}")
    ax1.hist(final_score, bins=bin_edges, alpha=0.45, label=f"Final     nz={fin_nz}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Importance Score (normalized)")
    ax1.set_ylabel("Count (log)")
    ax1.set_title("Importance Score Distributions (log-y + zoom)")

    # 标出 Top1% 分位阈值
    ax1.axvline(rf_q99, linestyle="--", linewidth=1)
    ax1.axvline(xgb_q99, linestyle="--", linewidth=1)
    ax1.axvline(fin_q99, linestyle="--", linewidth=1)
    ax1.text(0.98, 0.95,
             f"99th pct: RF={rf_q99:.3f}, XGB={xgb_q99:.3f}, Final={fin_q99:.3f}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9)

    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.2)

    # 放大图：只看 0~zoom_max
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.hist(rf_score, bins=np.linspace(0, zoom_max, 60), alpha=0.45, label="RF(Gini)")
    ax2.hist(xgb_score, bins=np.linspace(0, zoom_max, 60), alpha=0.45, label="XGB(mean)")
    ax2.hist(final_score, bins=np.linspace(0, zoom_max, 60), alpha=0.45, label="Final")
    ax2.set_xlim(0, zoom_max)
    ax2.set_xlabel(f"Zoomed in: [0, {zoom_max}]")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



def jaccard_set(a, b):
    a, b = set(a), set(b)
    if len(a | b) == 0:
        return 0.0
    return len(a & b) / len(a | b)


def plot_alpha_jaccard_heatmap(feature_names, rf_score, xgb_score, top_n, alpha_list, save_path):
    """不同 alpha 下 TopN 特征集合的 Jaccard 相似度热图"""
    top_sets = []
    for alpha in alpha_list:
        final = alpha * rf_score + (1 - alpha) * xgb_score
        idx = np.argsort(final)[::-1][:top_n]
        genes = [feature_names[i] for i in idx]
        top_sets.append(genes)

    m = len(alpha_list)
    jac = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            jac[i, j] = jaccard_set(top_sets[i], top_sets[j])

    plt.figure(figsize=(7, 6))
    sns.heatmap(jac, annot=True, fmt=".2f",
                xticklabels=[str(a) for a in alpha_list],
                yticklabels=[str(a) for a in alpha_list],
                cmap="YlGnBu")
    plt.title(f"Top{top_n} Gene Set Stability (Jaccard)")
    plt.xlabel("alpha")
    plt.ylabel("alpha")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top10_heatmap_by_subtype(df, label_col, top10_genes, save_path):
    """Top10基因在各 subtype 的均值热图"""
    heatmap_df = df.groupby(label_col)[top10_genes].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="coolwarm", center=heatmap_df.values.mean())
    plt.title("Top10 Genes Mean Expression by Subtype")
    plt.xlabel("Genes")
    plt.ylabel("Subtype")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_venn_optional(rf_top, xgb_top, final_top, save_path):
    """画3集合Venn图（需要 matplotlib-venn）"""
    try:
        from matplotlib_venn import venn3
    except ImportError:
        print("[ML Select] matplotlib-venn 未安装，跳过 Venn 图绘制。")
        return

    plt.figure(figsize=(7, 6))
    venn3([set(rf_top), set(xgb_top), set(final_top)],
          set_labels=("RF Top100", "XGB Top100", "Final Top100"))
    plt.title("Top100 Gene Overlap (RF / XGB / Final)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """
    用法：
      python driver.py --mode ml_select -d 100 -a 0.5

    sys.argv:
      [1] filepath(data目录)
      [2] top_n
      [3] alpha
    """
    filepath = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
    alpha = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.5

    clean_path = os.path.join(filepath, "clean_data.txt")
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"找不到 {clean_path}，请先运行 pretreat 生成 clean_data.txt")

    # fig目录
    fig_dir = os.path.join(filepath, "..", "fig")
    ensure_dir(fig_dir)

    df = pd.read_csv(clean_path)
    label_col = df.columns[-1]
    features = [c for c in df.columns if c != label_col]

    X = df[features].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df[label_col].values)

    n_samples, n_features = X.shape
    print(f"[ML Select] Samples={n_samples}, Features={n_features}, Classes={len(np.unique(y))}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # RandomForest (Gini)
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rf.fit(X_train, y_train)
    rf_imp = rf.feature_importances_.astype(np.float64)
    rf_imp_norm = minmax_norm(rf_imp)
    print("[ML Select] RF importance done.")

    # XGBoost (gain / cover / weight)
    if XGBClassifier is None:
        raise ImportError("未安装 xgboost，请执行 pip install xgboost")

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    booster = xgb.get_booster()
    gain_imp = get_xgb_importance(booster, n_features, "gain")
    cover_imp = get_xgb_importance(booster, n_features, "cover")
    weight_imp = get_xgb_importance(booster, n_features, "weight")

    gain_norm = minmax_norm(gain_imp)
    cover_norm = minmax_norm(cover_imp)
    weight_norm = minmax_norm(weight_imp)

    xgb_mean_norm = (gain_norm + cover_norm + weight_norm) / 3.0
    xgb_mean_norm = minmax_norm(xgb_mean_norm)
    print("[ML Select] XGB importance (gain/cover/weight) done.")

    # 集成融合
    final_score = alpha * rf_imp_norm + (1 - alpha) * xgb_mean_norm
    top_idx = np.argsort(final_score)[::-1][:top_n]
    top_genes = [features[i] for i in top_idx]

    print(f"[ML Select] alpha={alpha}, Top{top_n} genes selected.")
    print("Top10 genes:", top_genes[:10])

    # 输出 selected_data.txt
    selected_df = df[top_genes + [label_col]]
    out_path = os.path.join(filepath, "selected_data.txt")
    selected_df.to_csv(out_path, index=False)
    print(f"[ML Select] Saved: {out_path}")

    # 输出重要性打分表
    imp_table = pd.DataFrame({
        "gene": features,
        "rf_gini": rf_imp_norm,
        "xgb_gain": gain_norm,
        "xgb_cover": cover_norm,
        "xgb_weight": weight_norm,
        "xgb_mean": xgb_mean_norm,
        "final_score": final_score
    }).sort_values("final_score", ascending=False)

    imp_csv_path = os.path.join(filepath, "ml_importance_scores.csv")
    imp_table.to_csv(imp_csv_path, index=False)
    print(f"[ML Select] Saved: {imp_csv_path}")


    # 可视化输出
    # Top20 bar（3张）
    plot_top20_bar(features, rf_imp_norm,
                   "RF(Gini) Feature Importance Top20",
                   os.path.join(fig_dir, "ml_rf_top20.png"))

    plot_top20_bar(features, xgb_mean_norm,
                   "XGB(mean gain/cover/weight) Feature Importance Top20",
                   os.path.join(fig_dir, "ml_xgb_top20.png"))

    plot_top20_bar(features, final_score,
                   f"Final( alpha={alpha} ) Feature Importance Top20",
                   os.path.join(fig_dir, "ml_final_top20.png"))

    # RF vs XGB scatter（标注Final Top10）
    plot_rf_vs_xgb_scatter(features, rf_imp_norm, xgb_mean_norm,
                           highlight_genes=top_genes[:10],
                           save_path=os.path.join(fig_dir, "ml_rf_vs_xgb_scatter.png"))

    # 分数分布图
    plot_score_distribution(rf_imp_norm, xgb_mean_norm, final_score,
                            os.path.join(fig_dir, "ml_importance_distribution.png"))

    # alpha稳定性热图（Top100 Jaccard）
    alpha_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    plot_alpha_jaccard_heatmap(features, rf_imp_norm, xgb_mean_norm,
                               top_n=top_n, alpha_list=alpha_list,
                               save_path=os.path.join(fig_dir, "ml_alpha_jaccard_heatmap.png"))

    # Top10 均值热图
    plot_top10_heatmap_by_subtype(df, label_col, top_genes[:10],
                                  os.path.join(fig_dir, "ml_top10_heatmap.png"))

    # Venn图（需要 matplotlib-venn）
    rf_top = [features[i] for i in np.argsort(rf_imp_norm)[::-1][:100]]
    xgb_top = [features[i] for i in np.argsort(xgb_mean_norm)[::-1][:100]]
    final_top = [features[i] for i in np.argsort(final_score)[::-1][:100]]

    plot_venn_optional(rf_top, xgb_top, final_top,
                       os.path.join(fig_dir, "ml_top100_venn.png"))

    print("[ML Select] Task3 plots saved in fig/ directory.")


if __name__ == "__main__":
    main()
