import torch
import os
import warnings
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# --- 1. Configuration ---
DATA_FILE = 'graph_data/task_classification_eval_set.pt'
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
NEW_MODEL_NAME = "llm_task_classification"
MAX_SEQ_LENGTH = 2048

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

def format_training_prompt(example: dict) -> str:
    """
    Turns a (node_name, task_id) pair into a Mistral-style prompt.
    'example' is a dict from the Hugging Face Dataset.
    """
    node_name = example['name']
    task_id = example['label']
    
    prompt = f"{TASK_LIST_PROMPT}\n\nFor this node, here's the info:\n{node_name}"
    
    return f"<s>[INST] {prompt} [/INST] {task_id} </s>"

def main():
    print(f"Loading data from {DATA_FILE}...")
    data_tuples = torch.load(DATA_FILE, weights_only=False)
    
    data_dict = {
        'name': [name for name, _ in data_tuples],
        'label': [label for _, label in data_tuples]
    }
    dataset = Dataset.from_dict(data_dict)
    
    dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_splits['train']
    eval_dataset = dataset_splits['test']
    
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"\nSample training example:")
    print(format_training_prompt(train_dataset[0]))
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nLoading base model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=NEW_MODEL_NAME,
        num_train_epochs=1,
        per_device_train_batch_size=16,      
        gradient_accumulation_steps=1,  
        learning_rate=2e-4,
        logging_steps=25,
        optim="paged_adamw_8bit",
        save_strategy="epoch",          
        eval_strategy="epoch",   
        fp16=True,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        load_best_model_at_end=True,    
    )

    # Format the dataset text field
    def formatting_func(example):
        return format_training_prompt(example)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,        
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,  # Use processing_class for TRL 0.24.0
    )

    print("\nStarting fine-tuning for TASK CLASSIFICATION...")
    trainer.train()
    print("Training finished.")

    os.makedirs(f"./{NEW_MODEL_NAME}", exist_ok=True)
    final_model_path = f"./{NEW_MODEL_NAME}/{NEW_MODEL_NAME}-final"
    trainer.save_model(final_model_path)
    print(f"Fine-tuned task classification model saved to {final_model_path}")
    
    print("\n" + "="*80)
    print("FINAL VALIDATION METRICS (from training)")
    print("="*80)
    # Print last few log entries
    for entry in trainer.state.log_history[-5:]:
        print(entry)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Using_kex_impl=False*")
    main()