import torch
import re
import warnings
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader

# --- 1. Configuration ---
BATCH_SIZE = 16

EVAL_FILE = 'graph_data/task_classification_eval_set.pt'
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
FINETUNED_MODEL_PATH = "./llm_task_classification/llm_task_classification-final"

TASK_TO_IDX = {
    'text-generation': 0, 'question-answering': 1, 'text-to-video': 2, 
    'image-to-video': 3, 'image-to-3d': 4, 'robotics': 5, 'translation': 6, 
    'feature-extraction': 7, 'text-to-3d': 8, 'text-to-speech': 9, 
    'automatic-speech-recognition': 10, 'image-classification': 11, 
    'table-question-answering': 12, 'fill-mask': 13, 'multiple-choice': 14, 
    'visual-question-answering': 15, 'summarization': 16, 'image-to-text': 17, 
    'image-feature-extraction': 18, 'text-to-image': 19, 'text-to-audio': 20, 
    'reinforcement-learning': 21, 'image-text-to-text': 22, 'text-classification': 23, 
    'sentence-similarity': 24, 'zero-shot-classification': 25, 'text-retrieval': 26, 
    'token-classification': 27, 'object-detection': 28, 'audio-classification': 29, 
    'image-segmentation': 30, 'time-series-forecasting': 31, 'video-classification': 32, 
    'zero-shot-image-classification': 33, 'any-to-any': 34, 'image-to-image': 35, 
    'depth-estimation': 36, 'tabular-classification': 37, 'tabular-regression': 38, 
    'table-to-text': 39, 'video-text-to-text': 40, 'audio-to-audio': 41, 
    'voice-activity-detection': 42, 'audio-text-to-text': 43, 
    'document-question-answering': 44, 'visual-document-retrieval': 45, 
    'text-ranking': 46, 'graph-ml': 47, 'tabular-to-text': 48, 
    'unconditional-image-generation': 49, 'mask-generation': 50, 
    'keypoint-detection': 51, 'zero-shot-object-detection': 52, 'video-to-video': 53
}

TASK_LIST_PROMPT = f"""Here's a list of possible tasks. Please predict the right task for this model and output only the index.
task_to_idx: {TASK_TO_IDX}
"""

IDX_TO_TASK = {v: k for k, v in TASK_TO_IDX.items()}

def format_classification_prompt(node_name: str) -> str:
    prompt = f"{TASK_LIST_PROMPT}\n\nFor this node, here's the info:\n{node_name}"
    return f"<s>[INST] {prompt} [/INST]"

def extract_prediction(text: str) -> int:
    """Extract the first number from the generated text"""
    match = re.search(r'\d+', text)
    if match:
        pred_id = int(match.group(0))
        if pred_id in IDX_TO_TASK:
            return pred_id
    return -1

def main():
    # --- 2. Load Evaluation Data ---
    print(f"Loading evaluation set from {EVAL_FILE}...")
    eval_dataset_full = torch.load(EVAL_FILE, weights_only=False)
    print(f"Loaded {len(eval_dataset_full)} total samples.")

    # Re-create the same train/test split as training (seed=42)
    print("Re-creating train/test split (seed=42) to isolate test set...")
    dataset = Dataset.from_dict({
        'name': [name for name, _ in eval_dataset_full],
        'label': [label for _, label in eval_dataset_full]
    })
    dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)
    
    eval_dataset = dataset_splits['test']
    print(f"Using {len(eval_dataset)} samples for evaluation (10% test split).")

    # --- 3. Configure 4-bit Quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # --- 4. Load Base Model and Tokenizer ---
    print(f"Loading base model: {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- 5. Load Fine-Tuned Adapters ---
    print(f"\nLoading fine-tuned LoRA adapters from {FINETUNED_MODEL_PATH}...")
    model = PeftModel.from_pretrained(model, FINETUNED_MODEL_PATH)
    model = model.eval()
    print("Model loaded successfully.")

    # --- 6. Run Evaluation (Batched) ---
    all_ground_truth_ids = []
    all_predicted_ids = []
    
    print("\nStarting batched evaluation...")
    
    data_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=lambda batch: (
            [item['name'] for item in batch], 
            [item['label'] for item in batch]
        )
    )

    for batch in tqdm(data_loader, desc="Evaluating"):
        node_names, ground_truth_ids = batch
        
        prompt_texts = [format_classification_prompt(name) for name in node_names]
        
        inputs = tokenizer(
            prompt_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,  # Greedy decoding for consistency
            )
        
        for i in range(len(outputs)):
            input_length = inputs.input_ids[i].shape[0]
            generated_tokens = outputs[i][input_length:]
            answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            predicted_id = extract_prediction(answer)
            
            all_predicted_ids.append(predicted_id)
            all_ground_truth_ids.append(ground_truth_ids[i])

    # --- 7. Report Results ---
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    # Filter out invalid predictions (-1) for cleaner report
    valid_indices = [i for i, pred in enumerate(all_predicted_ids) if pred != -1]
    valid_ground_truth = [all_ground_truth_ids[i] for i in valid_indices]
    valid_predictions = [all_predicted_ids[i] for i in valid_indices]
    
    num_invalid = len(all_predicted_ids) - len(valid_predictions)
    if num_invalid > 0:
        print(f"Warning: {num_invalid} invalid predictions (no number found in output)")
    
    report = classification_report(
        valid_ground_truth, 
        valid_predictions, 
        labels=list(IDX_TO_TASK.keys()), 
        target_names=list(IDX_TO_TASK.values()),
        zero_division=0
    )
    
    accuracy = accuracy_score(valid_ground_truth, valid_predictions)
    
    report_file_name = "evaluation_report_task_classification.txt"
    print(f"\nSaving report to {report_file_name}...")
    
    with open(report_file_name, 'w') as f:
        f.write("FINE-TUNED TASK CLASSIFICATION EVALUATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Samples: {len(all_predicted_ids)}\n")
        f.write(f"Valid Predictions: {len(valid_predictions)}\n")
        f.write(f"Invalid Predictions: {num_invalid}\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    print(f"\nTotal Samples: {len(all_predicted_ids)}")
    print(f"Valid Predictions: {len(valid_predictions)}")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    print(report)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Using_kex_impl=False*")
    main()