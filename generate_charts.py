import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('paper/figures', exist_ok=True)

# 1. Bar Chart: Accuracy Shift (The Capitulation Effect)
datasets = ['TruthfulQA', 'GSM8K', 'MedQA', 'HaluEval']
baseline = [5.50, 84.02, 100.00, 94.55]
xexam = [4.59, 78.69, 60.00, 35.85]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline, width, label='Zero-Shot Baseline', color='#3498db')
rects2 = ax.bar(x + width/2, xexam, width, label='X-Exam (Adversarial)', color='#e74c3c')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('The Adversarial Capitulation Effect: Accuracy Degradation under Cross-Examination', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(fontsize=11)

# Add text labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('paper/figures/accuracy_shift.svg', format='svg')
plt.savefig('paper/figures/accuracy_shift.png', dpi=300)
plt.close()

# 2. Scatter / Bubble: Rejection Rate vs Accuracy Drop
rejection_rates = [39.9, 27.1, 41.4, 58.2] # TQA, GSM8K, MedQA, HaluEval
acc_drop = [-0.91, -5.33, -40.00, -58.70] # Negative means drop

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(rejection_rates, acc_drop, s=[200, 200, 200, 200], c='#e74c3c', alpha=0.7, edgecolors='k')

for i, txt in enumerate(datasets):
    ax.annotate(txt, (rejection_rates[i]+1, acc_drop[i]-1), fontsize=11, fontweight='bold')

# Trend line
z = np.polyfit(rejection_rates, acc_drop, 1)
p = np.poly1d(z)
plt.plot(rejection_rates, p(rejection_rates), "r--", alpha=0.8)

ax.set_xlabel('Adversarial Rejection Rate (%)', fontsize=12)
ax.set_ylabel('Accuracy Delta (Percentage Points)', fontsize=12)
ax.set_title('Correlation: Higher Scrutiny Leads to Deeper Capitulation', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('paper/figures/scrutiny_vs_capitulation.svg', format='svg')
plt.savefig('paper/figures/scrutiny_vs_capitulation.png', dpi=300)
plt.close()

print("High-quality SVGs and PNGs generated in paper/figures/")
