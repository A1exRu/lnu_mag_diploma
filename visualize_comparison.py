import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize():
    csv_path = "comparison_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run compare_methods.py first.")
        return

    print(f"📊 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    max_err = 2.0
    df_plot = df.copy()

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    error_cols = [c for c in df.columns if c.startswith("err_")]
    melted_df = df_plot.melt(value_vars=error_cols, var_name="Method", value_name="AbsRel")
    melted_df["Method"] = melted_df["Method"].str.replace("err_", "").str.upper()

    sns.boxplot(data=melted_df, x="Method", y="AbsRel", palette="Set2")
    plt.ylim(0, max_err)
    plt.title("Розподіл відносної похибки (AbsRel) за методами")
    plt.ylabel("AbsRel (менше - краще)")
    plt.savefig("error_distribution.png")
    print("✅ Saved error_distribution.png")

    plt.figure(figsize=(12, 7))
    methods = ["stereo", "mono_mde", "pinhole"]
    colors = ["blue", "green", "red"]

    for method, color in zip(methods, colors):
        err_col = f"err_{method}"
        if err_col in df.columns:
            valid_df = df_plot[df_plot[err_col].notna() & (df_plot[err_col] < max_err)]
            sns.regplot(data=valid_df, x="z_gt", y=err_col, scatter_kws={'alpha': 0.2},
                        line_kws={'label': method.upper()}, color=color, lowess=True)

    plt.xlim(0, 100)
    plt.ylim(0, 1.0)
    plt.title("Залежність похибки від реальної відстані (Lowess smoothing)")
    plt.xlabel("Ground Truth Distance (m)")
    plt.ylabel("AbsRel Error")
    plt.legend()
    plt.savefig("error_vs_distance.png")
    print("✅ Saved error_vs_distance.png")

    plt.figure(figsize=(10, 6))
    for method, color in zip(methods, colors):
        err_col = f"err_{method}"
        if err_col in df.columns:
            sns.kdeplot(df_plot[err_col], label=method.upper(), fill=True, color=color, clip=(0, max_err))

    plt.title("Щільність розподілу похибок")
    plt.xlabel("AbsRel")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("error_density.png")
    print("✅ Saved error_density.png")

    print("\n🚀 Всі графіки успішно згенеровані!")


if __name__ == "__main__":
    visualize()
