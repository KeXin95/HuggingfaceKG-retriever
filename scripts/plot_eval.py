import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

RESULTS_FILE = "evaluation_results.json"
PLOT_FILE_SCATTER = "f1_vs_frequency_final.png"
PLOT_FILE_CONFUSION = "confusion_matrix_top10.png"

TASK_LIST = [
    'text-generation', 'question-answering', 'text-to-video', 'image-to-video', 
    'image-to-3d', 'robotics', 'translation', 'feature-extraction', 'text-to-3d', 
    'text-to-speech', 'automatic-speech-recognition', 'image-classification', 
    'table-question-answering', 'fill-mask', 'multiple-choice', 
    'visual-question-answering', 'summarization', 'image-to-text', 
    'image-feature-extraction', 'text-to-image', 'text-to-audio', 
    'reinforcement-learning', 'image-text-to-text', 'text-classification', 
    'sentence-similarity', 'zero-shot-classification', 'text-retrieval', 
    'token-classification', 'object-detection', 'audio-classification', 
    'image-segmentation', 'time-series-forecasting', 'video-classification', 
    'zero-shot-image-classification', 'any-to-any', 'image-to-image', 
    'depth-estimation', 'tabular-classification', 'tabular-regression', 
    'table-to-text', 'video-text-to-text', 'audio-to-audio', 
    'voice-activity-detection', 'audio-text-to-text', 
    'document-question-answering', 'visual-document-retrieval', 'text-ranking', 
    'graph-ml', 'tabular-to-text', 'unconditional-image-generation', 
    'mask-generation', 'keypoint-detection', 'zero-shot-object-detection', 
    'video-to-video'
]

def plot_f1_vs_frequency(class_f1s, class_counts, class_names):
    plt.figure(figsize=(12, 8))
    
    plt.scatter(class_counts, class_f1s, alpha=0.7, s=100, c='royalblue', edgecolors='k')
    
    plt.xscale('log')
    
    plt.xlabel('Number of Training Samples (Log Scale)', fontsize=14)
    plt.ylabel('Test F1 Score', fontsize=14)
    plt.title("The 'Smoking Gun': Test F1 Score vs. Training Class Frequency", fontsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.ylim(-0.05, 1.05)
    
    avg_f1 = np.mean(class_f1s)
    plt.axhline(y=avg_f1, color='r', linestyle='--', linewidth=2, label=f'Avg Macro F1: {avg_f1:.2f}')
    
    sorted_indices = np.argsort(class_f1s)
    for i in list(sorted_indices[-3:]) + list(sorted_indices[:3]):
        if class_counts[i] > 50:
            plt.annotate(class_names[i], (class_counts[i], class_f1s[i]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOT_FILE_SCATTER, dpi=300)
    print(f"Scatter plot saved to {PLOT_FILE_SCATTER}")

def plot_confusion_matrix_top_classes(y_true_bin, y_pred_bin, class_counts):
    top_indices = np.argsort(class_counts)[-10:][::-1]
    top_names = [TASK_LIST[i] for i in top_indices]
    
    y_true_top = y_true_bin[:, top_indices]
    y_pred_top = y_pred_bin[:, top_indices]
    
    plt.figure(figsize=(10, 6))
    
    metrics = []
    for i in range(len(top_names)):
        report = classification_report(y_true_top[:, i], y_pred_top[:, i], output_dict=True, zero_division=0)
        metrics.append([report['1']['precision'], report['1']['recall'], report['1']['f1-score']])
        
    metrics = np.array(metrics)
    
    sns.heatmap(metrics, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=["Precision", "Recall", "F1-Score"], 
                yticklabels=top_names)
    
    plt.title("Performance Metrics for Top 10 Frequent Classes", fontsize=16)
    plt.tight_layout()
    plt.savefig(PLOT_FILE_CONFUSION, dpi=300)
    print(f"Confusion metrics saved to {PLOT_FILE_CONFUSION}")

def main():
    print(f"Loading results from {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        
    train_counts = data["train_counts"]
    test_data = data["test"]
    y_true = test_data["ground_truth"]
    y_pred = test_data["predictions"]
    
    mlb = MultiLabelBinarizer(classes=range(len(TASK_LIST)))
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    report = classification_report(y_true_bin, y_pred_bin, target_names=TASK_LIST, output_dict=True, zero_division=0)
    
    class_f1s = []
    valid_counts = []
    valid_names = []
    
    print("\n--- Per-Class Performance ---")
    for i, task in enumerate(TASK_LIST):
        if task in report:
            f1 = report[task]['f1-score']
            count = train_counts[i]
            
            if count > 0:
                class_f1s.append(f1)
                valid_counts.append(count)
                valid_names.append(task)
                if count < 100 and f1 > 0.5:
                    print(f"Star Performer (Low Data): {task} (N={count}, F1={f1:.2f})")

    plot_f1_vs_frequency(class_f1s, valid_counts, valid_names)
    
    plot_confusion_matrix_top_classes(y_true_bin, y_pred_bin, train_counts)

if __name__ == "__main__":
    main()