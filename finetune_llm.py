import torch
import os
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import warnings

# --- 1. Configuration ---
TRIPLES_FILE = 'graph_data/triples_cleaned.pt'
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
NEW_MODEL_NAME = "llm_finetune_ablation"
MAX_SEQ_LENGTH = 4096

def format_triplet_as_instruction(triplet: tuple) -> str:
    """
    Converts a (head, relation, tail) triplet into a 
    Mistral-style instruction-following string.
    """
    head, relation, tail = triplet
    
    prompt = (
        "You will be given a subject and a relation. "
        f"Your task is to provide the corresponding object.\n\n"
        f"Subject: {head}\n"
        f"Relation: {relation}"
    )
    
    return f"<s>[INST] {prompt} [/INST] {tail} </s>"

def load_data_as_hf_dataset(filepath: str) -> Dataset:
    """
    Loads the list of triples and converts it into a 
    Hugging Face Dataset object.
    """
    print(f"Loading triples from {filepath}...")
    triples = torch.load(filepath, weights_only=False)
    
    formatted_texts = [
        {"text": format_triplet_as_instruction(t)} for t in triples
    ]
    
    dataset = Dataset.from_list(formatted_texts)
    
    print(f"Loaded and formatted {len(dataset)} examples.")
    print("\n--- Example Formatted Text ---")
    print(dataset[0]['text'][:1000] + "...")
    print("-------------------------------\n")
    
    return dataset

def main():
    # --- 2. Load and Format Dataset ---
    dataset = load_data_as_hf_dataset(TRIPLES_FILE)
    train_dataset = dataset 

    # --- 3. Configure 4-bit Quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- 4. Load Model ---
    print(f"Loading base model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # --- 5. Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 6. Configure LoRA ---
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

    # --- 7. Configure Training Arguments ---
    training_args = TrainingArguments(
        output_dir=NEW_MODEL_NAME,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=25,
        optim="paged_adamw_8bit",
        save_steps=500,
        fp16=True,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
    )

    # --- 8. Formatting function ---
    def formatting_func(example):
        """Extract the 'text' field from each example"""
        return example["text"]

    # --- 9. Initialize SFTTrainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer, 
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    
    import glob
    checkpoints = sorted(glob.glob("./llm_finetune_ablation/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))
    
    print(f"Starting Finetuning Process...")
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f'Found Checkpoint: {latest_checkpoint}')
        try:
            print(f"\n\nResuming from checkpoint: {latest_checkpoint}\n\n")
            trainer.train(resume_from_checkpoint=latest_checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print(f"\n\nStarting from scratch instead\n\n")
            trainer.train()
    else:
        print(f"\n\nNo checkpoint found, starting from scratch\n\n")
        trainer.train()
    print("Training finished.")

    # --- 11. Save the Final Model ---
    os.makedirs("./llm_finetune_ablation/FINAL_CHECKPOINT", exist_ok=True)
    final_model_path = f"./llm_finetune_ablation/FINAL_CHECKPOINT/{NEW_MODEL_NAME}"
    trainer.save_model(final_model_path)
    print(f"Fine-tuned model saved to {final_model_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Using_kex_impl=False*")
    main()