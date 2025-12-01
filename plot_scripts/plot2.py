import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. HARDCODED DATA
# ==========================================

# Model Names
# We group them: Discriminative (Non-Generative) vs Generative (LLM-based)
models = [
    # --- Non-Generative (Discriminative) ---
    "GAT (BCE)",          # The baseline
    "GCN (Focal)",        # Better loss
    "SAGE (Focal)",       # Stronger architecture
    "GATv2 (Focal)",      # SOTA Discriminative
    
    # --- Generative (LLM-based) ---
    "GAT + Qwen 3B",      # Friend's best model
    "GAT + Mistral 7B " # Your model
]

# Test Micro F1 Scores (Accuracy/Precision focused)
# Data extracted from your uploaded images
micro_scores = [
    0.2175,  # GAT (BCE)
    0.4704,  # GCN (Focal)
    0.5928,  # SAGE (Focal)
    0.7085,  # GATv2 (Focal) - The highest Micro
    0.4104,  # GAT + Qwen
    0.3571   # GAT + Mistral 
]

# Test Macro F1 Scores (Diversity/Tail focused)
# Data extracted from your uploaded images
macro_scores = [
    0.0493,  # GAT (BCE)
    0.1829,  # GCN (Focal)
    0.1710,  # SAGE (Focal)
    0.2140,  # GATv2 (Focal)
    0.1964,  # GAT + Qwen
    0.1942   # GAT + Mistral 
]

# ==========================================
# 2. PLOTTING SETUP
# ==========================================

# Set up positions for bars
x = np.arange(len(models))
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(14, 8))

# Create Bars
# Micro F1 (Greenish)
rects1 = ax.bar(x - width/2, micro_scores, width, label='Test Micro F1', color='#a8d583', edgecolor='grey')
# Macro F1 (Orangish)
rects2 = ax.bar(x + width/2, macro_scores, width, label='Test Macro F1', color='#f3b562', edgecolor='grey')

# ==========================================
# 3. STYLING
# ==========================================

# Labels, Title, and Custom x-axis tick labels
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Discriminative vs. Generative Graph Models: Test Set Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha='right', fontsize=10)
ax.legend(fontsize=11)

# Add a grid for easier reading
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
ax.set_ylim(0, 0.8) # Adjust Y-limit to fit GATv2's high score

# Separator line between Discriminative and Generative
ax.axvline(x=3.5, color='grey', linestyle=':', linewidth=2)
ax.text(1.5, 0.75, 'Discriminative (GNN Only)', ha='center', fontsize=12, fontweight='bold', color='#555')
ax.text(4.5, 0.75, 'Generative (Retrieval+LLM)', ha='center', fontsize=12, fontweight='bold', color='#555')

# Function to attach a label above each bar
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

# Save and Show
save_path = "full_model_comparison.png"
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
plt.show()