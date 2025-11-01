import torch
import re
from collections import defaultdict
import warnings

TRIPLES_FILE = 'graph_data/triples_cleaned.pt'
OUTPUT_FILE = 'graph_data/task_classification_eval_set.pt'

TASK_TO_IDX = {
    'text-generation': 0, 'question-answering': 1, 'text-to-video': 2, 
    'image-to-video': 3, 'image-to-3d': 4, 'robotics': 5, 'translation': 6, 
    'feature-extraction': 7, 'text-to-3d': 8, 'text-to-speech': 9, 
    'automatic-speech-recognition': 10, 'image-classification': 11, 
    'table-question-answering': 12, 'fill-mask': 13, 'multiple-choice': 14, 
    'visual-question-answering': 15, 'summarization': 16, 'image-to-text': 17, 
    'image-feature-extraction': 18, 'text-to-image': 19, 'text-to-audio': 20, 
    'reinforcement-learning': 21, 'image-text-to-text': 22, 
    'text-classification': 23, 'sentence-similarity': 24, 
    'zero-shot-classification': 25, 'text-retrieval': 26, 
    'token-classification': 27, 'object-detection': 28, 
    'audio-classification': 29, 'image-segmentation': 30, 
    'time-series-forecasting': 31, 'video-classification': 32, 
    'zero-shot-image-classification': 33, 'any-to-any': 34, 
    'image-to-image': 35, 'depth-estimation': 36, 
    'tabular-classification': 37, 'tabular-regression': 38, 
    'table-to-text': 39, 'video-text-to-text': 40, 'audio-to-audio': 41, 
    'voice-activity-detection': 42, 'audio-text-to-text': 43, 
    'document-question-answering': 44, 'visual-document-retrieval': 45, 
    'text-ranking': 46, 'graph-ml': 47, 'tabular-to-text': 48, 
    'unconditional-image-generation': 49, 'mask-generation': 50, 
    'keypoint-detection': 51, 'zero-shot-object-detection': 52, 
    'video-to-video': 53
}

pipeline_regex = re.compile(r"Pipeline: ([a-zA-Z0-9-]+)")

node_name_regex = re.compile(r"Node: (.*?)\. Type: model")

def extract_info(node_text: str):
    """
    Parses the node text.
    Returns (node_name, task_id) if it's a model with a task.
    Returns (None, -1) otherwise.
    """
    name_match = node_name_regex.search(node_text)
    if not name_match:
        return None, -1 
        
    node_name = name_match.group(1).strip()
    
    task_match = pipeline_regex.search(node_text)
    if task_match:
        task_name = task_match.group(1).lower()
        if task_name in TASK_TO_IDX:
            return node_name, TASK_TO_IDX[task_name]

    return None, -1

def main():
    print(f"Loading {TRIPLES_FILE}...")
    triples = torch.load(TRIPLES_FILE, weights_only=False)
    
    unique_models = {} 
    
    print("Parsing triples to find unique models and their tasks...")
    for head, _, tail in triples:
        head_name, head_task = extract_info(head)
        if head_name and head_name not in unique_models:
            unique_models[head_name] = head_task
        
        tail_name, tail_task = extract_info(tail)
        if tail_name and tail_name not in unique_models:
            unique_models[tail_name] = tail_task
            
    eval_dataset = []
    for node_name, task_id in unique_models.items():
        if task_id != -1: 
            eval_dataset.append((node_name, task_id))
            
    print(f"\nFound {len(unique_models)} unique model nodes.")
    print(f"Created evaluation set with {len(eval_dataset)} models with a valid task.")

    print(f"Saving evaluation set to {OUTPUT_FILE}...")
    torch.save(eval_dataset, OUTPUT_FILE)
    
    print("\nDone. Sample from eval set:")
    sample_name, sample_id = eval_dataset[0]
    print(f"Node Name: {sample_name}")
    print(f"Task ID: {sample_id} ({list(TASK_TO_IDX.keys())[sample_id]})")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*weights_only.*")
    main()
