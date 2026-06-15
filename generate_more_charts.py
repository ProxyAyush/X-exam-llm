import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('paper/figures', exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# --- 1. Polar/Radar Chart: Domain Vulnerability ---
categories = ['Math (GSM8K)', 'Legal (CaseHOLD)', 'Medical (MedQA)', 'Logical (TruthfulQA)', 'Hallucination (HaluEval)', 'Extreme (HLE)']
# Vulnerability mapped as the Accuracy Drop (absolute percentage points) or Rejection Rate. 
# Let's use Rejection Rate as the "Scrutiny Volume" axis.
rejection_rates = [27.14, 37.40, 41.40, 39.90, 58.17, 37.72]

# Number of variables
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
rejection_rates += rejection_rates[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, weight='bold')

ax.plot(angles, rejection_rates, linewidth=2, linestyle='solid', color='#e74c3c')
ax.fill(angles, rejection_rates, '#e74c3c', alpha=0.25)

ax.set_title("Adversarial Scrutiny Distribution by Domain", size=15, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('paper/figures/domain_vulnerability_radar.png', dpi=300)
plt.savefig('paper/figures/domain_vulnerability_radar.svg', format='svg')
plt.close()


# --- 2. Stacked Bar Chart: Verdict Composition ---
datasets = ['GSM8K', 'TruthfulQA', 'LawSE', 'CaseHOLD', 'HLE', 'MedQA', 'MedMCQA', 'HaluEval']
accept_rates = [71.49, 59.36, 62.38, 62.20, 62.28, 57.58, 49.98, 41.06]
reject_rates = [27.14, 39.90, 36.99, 37.40, 37.72, 41.40, 48.03, 58.17]

fig, ax = plt.subplots(figsize=(12, 7))

# Sort by reject rates
sorted_indices = np.argsort(reject_rates)
datasets = [datasets[i] for i in sorted_indices]
accept_rates = [accept_rates[i] for i in sorted_indices]
reject_rates = [reject_rates[i] for i in sorted_indices]

x = np.arange(len(datasets))
width = 0.65

ax.bar(x, accept_rates, width, label='Judge Verdict: ACCEPT', color='#2ecc71', edgecolor='black')
ax.bar(x, reject_rates, width, bottom=accept_rates, label='Judge Verdict: REJECT (Cross-Examiner Success)', color='#e74c3c', edgecolor='black')

ax.set_ylabel('Percentage (%)', fontsize=12, weight='bold')
ax.set_title('Verdict Composition: The Frequency of Adversarial Interventions', fontsize=16, weight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, rotation=30, ha='right')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.1), fontsize=11)

# Add horizontal line at 50%
ax.axhline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

for i in range(len(datasets)):
    ax.text(i, accept_rates[i]/2, f'{accept_rates[i]:.1f}%', ha='center', va='center', color='black', weight='bold', fontsize=10)
    ax.text(i, accept_rates[i] + reject_rates[i]/2, f'{reject_rates[i]:.1f}%', ha='center', va='center', color='white', weight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('paper/figures/verdict_composition.png', dpi=300)
plt.savefig('paper/figures/verdict_composition.svg', format='svg')
plt.close()

print("Generated Radar Chart and Stacked Bar Chart.")
