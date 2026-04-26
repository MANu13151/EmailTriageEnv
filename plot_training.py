import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('static/training_logs.json', 'r') as f:
    data = json.load(f)

categories = ['Overall', 'Easy', 'Medium', 'Hard']
baseline_scores = [
    data['baseline']['avg_reward'],
    data['baseline']['per_difficulty']['easy']['avg_reward'],
    data['baseline']['per_difficulty']['medium']['avg_reward'],
    data['baseline']['per_difficulty']['hard']['avg_reward']
]

trained_scores = [
    data['trained']['avg_reward'],
    data['trained']['per_difficulty']['easy']['avg_reward'],
    data['trained']['per_difficulty']['medium']['avg_reward'],
    data['trained']['per_difficulty']['hard']['avg_reward']
]

x = np.arange(len(categories))
width = 0.35

# Modern styling
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Custom colors matching the dashboard (red/green)
color_baseline = '#ef4444'
color_trained = '#22c55e'

rects1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline', color=color_baseline, alpha=0.85)
rects2 = ax.bar(x + width/2, trained_scores, width, label='Trained (GRPO)', color=color_trained, alpha=0.85)

ax.set_ylabel('Average Reward', fontsize=13, fontweight='bold', labelpad=10)
ax.set_title('Agent Performance Improvement via GRPO Training', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')

# Add a horizontal line at 0 for reference
ax.axhline(0, color='black', linewidth=1.2, linestyle='--')

# Function to autolabel bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # Format specifically for negative vs positive
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5 if height > 0 else -18),  # offset points
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig('static/training_improvement.svg', format='svg', bbox_inches='tight')
print("Plot saved to static/training_improvement.svg")
