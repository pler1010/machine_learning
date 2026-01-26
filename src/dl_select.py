import sys
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader


# =========================
#  PyTorch
# =========================
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def minmax_norm(x, eps=1e-12):
    x = np.asarray(x)
    mn, mx = x.min(), x.max()
    if mx - mn < eps:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn + eps)


def jaccard_set(a, b):
    a, b = set(a), set(b)
    if len(a | b) == 0:
        return 0.0
    return len(a & b) / len(a | b)


class GeneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """
    多层感知机：Linear + BN + ReLU + Dropout
    """
    def __init__(self, input_dim, num_classes=4, hidden_dims=(512, 256), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))   # ✅ BatchNorm
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout)) # ✅ Dropout
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(model, train_loader, val_loader, device,
              epochs=30, lr=1e-3, weight_decay=1e-4, patience=6):
    """
    训练 MLP，多分类交叉熵 + early stopping
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    bad_count = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += xb.size(0)
        val_acc = val_correct / val_total

        print(f"[DL Train] Epoch {ep:02d}/{epochs} | train_loss={train_loss:.4f} "
              f"| train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        # early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_count = 0
        else:
            bad_count += 1
            if bad_count >= patience:
                print(f"[DL Train] Early stop triggered. Best val_acc={best_val_acc:.4f}")
                break

    # 恢复最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc


def compute_gradient_importance(model, data_loader, device, mode="abs"):
    """
    梯度重要性：
    对每个 batch 输入 x requires_grad=True
    取真实类别 logit 的和，反传得到 d(logit_y)/dx
    再累加 abs(grad) 或 grad^2
    """
    model.eval()
    all_importance = None
    total_count = 0

    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        xb.requires_grad_(True)

        logits = model(xb)  # [B, C]
        # 取真实类别的 logit: logits[range(B), y]
        chosen = logits.gather(1, yb.view(-1, 1)).sum()

        model.zero_grad()
        if xb.grad is not None:
            xb.grad.zero_()
        chosen.backward()

        grad = xb.grad.detach()  # [B, D]

        if mode == "abs":
            imp = grad.abs()
        elif mode == "square":
            imp = grad.pow(2)
        else:
            raise ValueError("mode must be 'abs' or 'square'")

        imp_sum = imp.sum(dim=0).cpu().numpy()  # [D]

        if all_importance is None:
            all_importance = imp_sum
        else:
            all_importance += imp_sum

        total_count += xb.size(0)

    # 平均到每个样本
    all_importance = all_importance / max(total_count, 1)
    return all_importance


def plot_top20_bar(feature_names, scores, title, save_path):
    idx = np.argsort(scores)[::-1][:20]
    genes = [feature_names[i] for i in idx][::-1]
    vals = scores[idx][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(genes, vals)
    plt.title(title)
    plt.xlabel("Importance (normalized)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distribution(scores, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=60, alpha=0.8)
    plt.title(title)
    plt.xlabel("Importance (normalized)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_overlap_bar(intersection_size, top_n, save_path):
    plt.figure(figsize=(6, 4))
    plt.bar(["Overlap", "Non-overlap"], [intersection_size, top_n - intersection_size])
    plt.title(f"Overlap Count (Top{top_n})")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_venn_optional(set_a, set_b, save_path, label_a="ML Top100", label_b="DL Top100"):
    try:
        from matplotlib_venn import venn2
    except ImportError:
        print("[DL Select] matplotlib-venn 未安装，跳过 Venn 图绘制。")
        return
    plt.figure(figsize=(6, 5))
    venn2([set(set_a), set(set_b)], set_labels=(label_a, label_b))
    plt.title("Feature Set Overlap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """
    用法（driver 会传参）：
      python driver.py --mode dl_select -d 100 -g abs -e 30

    sys.argv:
      [1] filepath(data目录)
      [2] top_n
      [3] grad_mode: abs / square
      [4] epochs
    """
    if torch is None:
        raise ImportError("未安装 PyTorch，请执行：pip install torch")

    filepath = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
    grad_mode = sys.argv[3] if len(sys.argv) >= 4 else "abs"
    epochs = int(sys.argv[4]) if len(sys.argv) >= 5 else 30

    clean_path = os.path.join(filepath, "clean_data.txt")
    if not os.path.exists(clean_path):
        raise FileNotFoundError("找不到 clean_data.txt，请先运行 pretreat")

    # fig目录：data/../fig
    fig_dir = os.path.join(filepath, "..", "fig")
    ensure_dir(fig_dir)

    # 读取数据
    df = pd.read_csv(clean_path)
    label_col = df.columns[-1]
    feature_names = [c for c in df.columns if c != label_col]

    X_raw = df[feature_names].values.astype(np.float32)
    y_raw = df[label_col].values
    y = LabelEncoder().fit_transform(y_raw)

    # 标准化（MLP更稳）
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print(f"[DL Select] Samples={n_samples}, Features={n_features}, Classes={n_classes}")

    # 训练/验证/测试划分
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # DataLoader
    train_loader = DataLoader(GeneDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(GeneDataset(X_val, y_val), batch_size=256, shuffle=False)
    test_loader = DataLoader(GeneDataset(X_test, y_test), batch_size=256, shuffle=False)

    # 用全体数据计算梯度重要性更合理（训练完后）
    full_loader = DataLoader(GeneDataset(X, y), batch_size=128, shuffle=False)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DL Select] Device:", device)

    # 构建 MLP（✅ Dropout + BatchNorm）
    model = MLPClassifier(
        input_dim=n_features,
        num_classes=n_classes,
        hidden_dims=(512, 256),
        dropout=0.3
    ).to(device)

    # 训练
    best_val_acc = train_mlp(
        model, train_loader, val_loader,
        device=device,
        epochs=epochs,
        lr=1e-3,
        weight_decay=1e-4,
        patience=6
    )
    print(f"[DL Select] Best val_acc={best_val_acc:.4f}")

    # 测试集简单评估
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            test_correct += (pred == yb).sum().item()
            test_total += xb.size(0)
    test_acc = test_correct / max(test_total, 1)
    print(f"[DL Select] Test accuracy (MLP): {test_acc:.4f}")

    # ==============================
    # ✅ 梯度重要性计算（任务4关键）
    # ==============================
    grad_importance = compute_gradient_importance(
        model, full_loader, device=device, mode=grad_mode
    )
    grad_importance_norm = minmax_norm(grad_importance)

    # TopN 选择
    top_idx = np.argsort(grad_importance_norm)[::-1][:top_n]
    top_genes = [feature_names[i] for i in top_idx]

    print(f"[DL Select] grad_mode={grad_mode}, Top{top_n} genes selected.")
    print("Top10 genes:", top_genes[:10])

    # 输出 selected_data.txt（供 evaluate.py 使用）
    selected_df = df[top_genes + [label_col]]
    out_path = os.path.join(filepath, "selected_data.txt")
    selected_df.to_csv(out_path, index=False)
    print(f"[DL Select] Saved: {out_path}")

    # 输出打分表
    score_df = pd.DataFrame({
        "gene": feature_names,
        f"grad_{grad_mode}": grad_importance_norm
    }).sort_values(f"grad_{grad_mode}", ascending=False)

    score_path = os.path.join(filepath, f"dl_importance_scores_{grad_mode}.csv")
    score_df.to_csv(score_path, index=False)
    print(f"[DL Select] Saved: {score_path}")

    # ==============================
    # ✅ 与任务3(ML)结果重叠率对比
    # ==============================
    ml_score_path = os.path.join(filepath, "ml_importance_scores.csv")
    if os.path.exists(ml_score_path):
        ml_df = pd.read_csv(ml_score_path)
        ml_top = ml_df.sort_values("final_score", ascending=False)["gene"].head(top_n).tolist()

        inter = set(ml_top) & set(top_genes)
        jac = jaccard_set(ml_top, top_genes)

        print("\n[DL vs ML Overlap]")
        print(f"ML Top{top_n} ∩ DL Top{top_n} = {len(inter)}")
        print(f"Jaccard = {jac:.4f}")
        print(f"Overlap ratio = {len(inter)/top_n:.4f}")

        # 保存 overlap 结果表（方便写报告）
        overlap_path = os.path.join(filepath, f"dl_vs_ml_overlap_top{top_n}.txt")
        with open(overlap_path, "w", encoding="utf-8") as f:
            f.write(f"ML Top{top_n} ∩ DL Top{top_n} = {len(inter)}\n")
            f.write(f"Jaccard = {jac:.4f}\n")
            f.write(f"Overlap ratio = {len(inter)/top_n:.4f}\n")
            f.write("Intersection genes:\n")
            for g in sorted(list(inter)):
                f.write(g + "\n")
        print(f"[DL Select] Saved overlap report: {overlap_path}")

        # 可视化：重叠条形图 & Venn
        plot_overlap_bar(len(inter), top_n, os.path.join(fig_dir, "dl_vs_ml_overlap_bar.png"))
        plot_venn_optional(ml_top, top_genes, os.path.join(fig_dir, "dl_vs_ml_venn.png"))
    else:
        print("[DL Select] 未找到 ml_importance_scores.csv（请先运行任务3：ml_select）")

    # ==============================
    # ✅ 可视化输出（任务4建议）
    # ==============================
    plot_top20_bar(
        feature_names,
        grad_importance_norm,
        f"DL Gradient Importance Top20 ({grad_mode})",
        os.path.join(fig_dir, f"dl_grad_top20_{grad_mode}.png")
    )
    plot_distribution(
        grad_importance_norm,
        f"DL Gradient Importance Distribution ({grad_mode})",
        os.path.join(fig_dir, f"dl_grad_distribution_{grad_mode}.png")
    )

    print("[DL Select] Task4 plots saved in fig/ directory.")


if __name__ == "__main__":
    main()
