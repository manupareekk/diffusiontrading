"""
Generate B&W figures for comprehensive research paper V2.
Matches the metrics and data from the updated academic text.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# Publication settings
rc('font', family='serif', size=9)
rc('text', usetex=False)
rc('axes', edgecolor='#000000', linewidth=0.8)

COLORS_BW = {
    'diffusion': '#000000',
    'baseline': '#4D4D4D',
    'empirical': '#666666',
    'benchmark': '#999999',
    'gan': '#666666',
}

import os
os.makedirs('figures2', exist_ok=True)

# ==================================================================
# Figure 1: F1 Score Ablation (Table 2 visualization)
# ==================================================================

fig, ax = plt.subplots(figsize=(3.5, 2.5), facecolor='white')

configurations = ['Raw Data', 'MSE Only', 'MSE+Fourier+TV\n(Proposed)']
f1_scores = [0.452, 0.558, 0.806]

hatches = ['xxx', '///', '']
bars = ax.bar(configurations, f1_scores, color=['white', 'white', 'black'],
              edgecolor='black', linewidth=1.5, hatch=hatches, width=0.6, alpha=0.8)

for bar, score in zip(bars, f1_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

ax.axhline(y=0.5, color='#666666', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(0.15, 0.515, 'Random', fontsize=7, style='italic', color='#333333')

ax.set_ylabel('F1 Score (1-Hour Horizon)', fontsize=9, fontweight='bold')
ax.set_title('Denoising Efficacy: Fourier Loss Ablation', fontsize=10, fontweight='bold', pad=8)
ax.set_ylim(0, 0.88)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=7)
plt.xticks(rotation=0, ha='center')

plt.tight_layout()
plt.savefig('figures2/fig1_f1_ablation.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created fig1_f1_ablation.png")
plt.close()

# ==================================================================
# Figure 2: Predictive Score Comparison (Diffusion vs GAN)
# ==================================================================

fig, ax = plt.subplots(figsize=(3.5, 2.5), facecolor='white')

methods = ['GAN\n(WGAN)', 'Diffusion\n(Ours)']
scores = [3.453, 1.213]

hatches = ['///', '']
bars = ax.bar(methods, scores, color=['white', 'black'],
              edgecolor='black', linewidth=1.5, hatch=hatches, width=0.5, alpha=0.8)

for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Add improvement annotation
improvement = ((scores[0] - scores[1]) / scores[0]) * 100
ax.text(0.5, 2.5, f'3× Better\n({improvement:.0f}% ↓)', fontsize=9, ha='center',
        fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
        edgecolor='black', linewidth=1.5))

ax.set_ylabel('Predictive Score (MAE)\nLower = Better', fontsize=9, fontweight='bold')
ax.set_title('Generative Fidelity (TRADES Metric)', fontsize=10, fontweight='bold', pad=8)
ax.set_ylim(0, 3.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('figures2/fig2_predictive_score.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created fig2_predictive_score.png")
plt.close()

# ==================================================================
# Figure 3: Portfolio Performance Comparison (From Table 1)
# ==================================================================

fig, ax = plt.subplots(figsize=(3.5, 2.8), facecolor='white')

strategies = ['Diffusion\n(Factor-MVO)', 'Baseline\n(Ranking)', 'Empirical\nMVO', 'S&P 500\n(Buy&Hold)']
sharpe_ratios = [1.68, 1.42, 0.96, 0.72]

hatches = ['', '///', '\\\\\\', 'xxx']
colors_list = ['black', 'white', 'white', 'white']
bars = ax.bar(strategies, sharpe_ratios, color=colors_list, edgecolor='black',
              linewidth=1.5, hatch=hatches, alpha=0.85)

for bar, ratio in zip(bars, sharpe_ratios):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.04,
            f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

ax.set_ylabel('Sharpe Ratio (Daily)', fontsize=9, fontweight='bold')
ax.set_title('Portfolio Performance (Net of Costs)', fontsize=10, fontweight='bold', pad=8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.6)
ax.set_ylim(0, 1.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('figures2/fig3_sharpe_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created fig3_sharpe_comparison.png")
plt.close()

print("\n✅ All figures for Paper V2 created!")
print("\nFigures created:")
print("  1. fig1_f1_ablation.png - Fourier loss ablation study")
print("  2. fig2_predictive_score.png - GAN vs Diffusion comparison")
print("  3. fig3_sharpe_comparison.png - Portfolio performance")
