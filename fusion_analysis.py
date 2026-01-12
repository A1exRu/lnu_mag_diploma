import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CSV_INPUT = "comparison_results.csv"
OUTPUT_PLOT = "fusion_comparison.png"


def load_data(path):
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return None
    df = pd.read_csv(path)

    return df


def fusion_strategies(df):
    """
    Додає різні стратегії ф'южену в dataframe.
    """

    has_stereo = df['z_stereo'].notna()
    has_pinhole = df['z_pinhole'].notna()
    has_mde = df['z_mono_mde'].notna()

    df['z_fusion_mean'] = df[['z_stereo', 'z_mono_mde', 'z_pinhole']].mean(axis=1)

    def weighted_fusion_row(row):
        z_s, z_m, z_p = row['z_stereo'], row['z_mono_mde'], row['z_pinhole']
        vals = []
        weights = []

        if np.isfinite(z_s):
            w_s = 1.0 if z_s < 40 else 0.5
            vals.append(z_s)
            weights.append(w_s)

        if np.isfinite(z_m):
            w_m = 0.2
            vals.append(z_m)
            weights.append(w_m)

        if np.isfinite(z_p):
            w_p = 0.2
            vals.append(z_p)
            weights.append(w_p)

        if not vals:
            return np.nan
        return np.average(vals, weights=weights)

    df['z_fusion_weighted'] = df.apply(weighted_fusion_row, axis=1)

    df['z_fusion_stereo_pref'] = df['z_stereo'].fillna((df['z_mono_mde'] + df['z_pinhole']) / 2)

    for col in ['z_fusion_mean', 'z_fusion_weighted', 'z_fusion_stereo_pref']:
        err_name = col.replace('z_fusion_', 'err_fusion_')
        df[err_name] = np.abs(df[col] - df['z_gt']) / (df['z_gt'] + 1e-6)

    return df


def visualize_fusion(df):
    plt.figure(figsize=(15, 10))

    error_cols = [
        ('Stereo', 'err_stereo'),
        ('Mono MDE', 'err_mono_mde'),
        ('Mono Pinhole', 'err_pinhole'),
        ('Fusion (Mean)', 'err_fusion_mean'),
        ('Fusion (Weighted)', 'err_fusion_weighted'),
        ('Fusion (Stereo-Pref)', 'err_fusion_stereo_pref')
    ]

    plot_data = []
    for label, col in error_cols:
        valid_errs = df[col].dropna()
        temp_df = pd.DataFrame({'Method': label, 'AbsRel': valid_errs})
        plot_data.append(temp_df)

    combined_plot_df = pd.concat(plot_data)

    plt.subplot(2, 1, 1)
    sns.boxplot(x='Method', y='AbsRel', data=combined_plot_df)
    plt.ylim(0, 1.0)
    plt.title('Distribution of AbsRel Error (Comparison with Fusion Strategies)')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    sns.regplot(x='z_gt', y='err_fusion_weighted', data=df, label='Fusion (Weighted)', scatter_kws={'alpha': 0.5},
                line_kws={'color': 'red'})
    sns.regplot(x='z_gt', y='err_stereo', data=df, label='Stereo', scatter_kws={'alpha': 0.3},
                line_kws={'color': 'blue', 'linestyle': '--'})

    plt.ylim(0, 1.0)
    plt.xlim(0, 80)
    plt.xlabel('Ground Truth Distance (m)')
    plt.ylabel('AbsRel Error')
    plt.title('Error vs Distance: Stereo vs Weighted Fusion')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Visualization saved to {OUTPUT_PLOT}")


def main():
    print("🚀 Starting Fusion Analysis...")
    df = load_data(CSV_INPUT)
    if df is None: return

    df = fusion_strategies(df)

    print("\n=== METRICS COMPARISON (Median AbsRel) ===")
    methods = {
        "Stereo": "err_stereo",
        "Mono MDE": "err_mono_mde",
        "Mono Pinhole": "err_pinhole",
        "Fusion (Mean)": "err_fusion_mean",
        "Fusion (Weighted)": "err_fusion_weighted",
        "Fusion (Stereo-Pref)": "err_fusion_stereo_pref"
    }

    for name, col in methods.items():
        median_err = df[col].median()
        mean_err = df[col].mean()
        print(f"{name:25s}: Median={median_err:.4f}, Mean={mean_err:.4f}")

    visualize_fusion(df)

    df.to_csv("fusion_results.csv", index=False)
    print("\nDetailed results saved to fusion_results.csv")


if __name__ == "__main__":
    main()
