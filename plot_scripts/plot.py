import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. HARDCODED DATA
# ==========================================

# Model Names
models = [
    "SAGE + Qwen 3B",
    "GAT + Qwen 3B",
    "GATv2 + Qwen 3B",
    "GAT + Mistral 7B "
]

# Test Micro F1 Scores (Accuracy/Precision focused)
micro_scores = [
    0.3238,  # SAGE + Qwen
    0.4104,  # GAT + Qwen
    0.4515,  # GATv2 + Qwen
    0.3571   # Mistral 
]

# Test Macro F1 Scores (Diversity/Tail focused)
macro_scores = [
    0.1159,  # SAGE + Qwen
    0.1964,  # GAT + Qwen
    0.1369,  # GATv2 + Qwen
    0.1942   # Mistral 
]

# ==========================================
# 2. PLOTTING SETUP
# ==========================================

# Set up positions for bars
x = np.arange(len(models))
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 7))

# Create Bars
# Micro F1 (Greenish)
rects1 = ax.bar(x - width/2, micro_scores, width, label='Test Micro F1', color='#a8d583', edgecolor='grey')
# Macro F1 (Orangish)
rects2 = ax.bar(x + width/2, macro_scores, width, label='Test Macro F1', color='#f3b562', edgecolor='grey')

# ==========================================
# 3. STYLING
# ==========================================

# Labels, Title, and Custom x-axis tick labels
ax.set_ylabel('F1 Score')
ax.set_title('Generative Graph Models: Micro vs. Macro F1 Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend()

# Add a grid for easier reading
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
ax.set_ylim(0, 0.55) # Adjust Y-limit to fit data nicely

# Function to attach a label above each bar
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# Save and Show
save_path = "comparison_mistral_vs_qwen.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
plt.show()