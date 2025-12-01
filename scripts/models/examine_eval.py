import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

RESULTS_FILE = "evaluation_results.json"

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

def main():
    print(f"Loading {RESULTS_FILE}...")
    try:
        with open(RESULTS_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: File not found. Make sure you are in the directory containing the json file.")
        return

    print("\n" + "="*60)
    print("DATA STRUCTURE")
    print("="*60)
    print(f"Keys found: {list(data.keys())}")
    
    if "test" in data:
        test_data = data["test"]
        preds = test_data["predictions"]
        truth = test_data["ground_truth"]
        print(f"Test Set Size: {len(preds)} samples")
    else:
        print("Error: 'test' key not found in JSON.")
        return

    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS (First 10)")
    print("="*60)
    for i in range(min(10, len(preds))):
        p = preds[i]
        t = truth[i]
        match = "✅" if set(p) == set(t) else "❌"
        
        p_names = [TASK_LIST[x] for x in p if x < len(TASK_LIST)]
        t_names = [TASK_LIST[x] for x in t if x < len(TASK_LIST)]
        
        print(f"Sample {i+1}: {match}")
        print(f"  Pred:  {p} -> {p_names}")
        print(f"  Truth: {t} -> {t_names}")
        print("-" * 30)

    print("\n" + "="*60)
    print("METRICS CALCULATION")
    print("="*60)
    
    mlb = MultiLabelBinarizer(classes=range(len(TASK_LIST)))
    y_true_bin = mlb.fit_transform(truth)
    y_pred_bin = mlb.transform(preds)

    report = classification_report(
        y_true_bin, 
        y_pred_bin, 
        target_names=TASK_LIST, 
        zero_division=0
    )
    
    print(report)
    
    report_dict = classification_report(
        y_true_bin, 
        y_pred_bin, 
        output_dict=True, 
        zero_division=0
    )
    
    print(f"\nFINAL SUMMARY:")
    print(f"  Micro F1: {report_dict['micro avg']['f1-score']:.4f} (Best metric for overall performance)")
    print(f"  Macro F1: {report_dict['macro avg']['f1-score']:.4f} (Average per class)")
    print(f"  Exact Accuracy: {accuracy_score(y_true_bin, y_pred_bin):.4f} (All labels must match perfectly)")

if __name__ == "__main__":
    main()