# ===================================================================
# QWEN2-VL-2B-INSTRUCT FINE-TUNING - HYBRID OPTIMIZED
# Hardware: 20GB VRAM + 64GB RAM
# ===================================================================

import torch
import os
import gc
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict, Image as HFImage
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# === 1. CONFIGURATION (The Safe Settings) ===
@dataclass
class TrainingConfig:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    output_dir: str = "./qwen2vl-wave-ui-finetuned"
    
    # Dataset
    train_data_path: str = r"C:\AI_Training\wave_ui_web_train.parquet" # Update this!
    
    # 20GB VRAM OPTIMIZED SETTINGS
    # Batch 4 is too risky. We use Batch 2 + Accumulation 8 = Effective 16
    batch_size: int = 2          
    gradient_accumulation_steps: int = 8
    
    learning_rate: float = 2e-4
    num_epochs: int = 1         # Start with 1. 55k images is a lot.
    
    # LoRA (Middle Ground)
    lora_r: int = 32            # 64 is heavy, 16 is light. 32 is the sweet spot.
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    
    # Resolution (Cap at 1024 to stay safe, 1280 is risky on Batch 2)
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1024 * 28 * 28 
    
    # Precision
    use_flash_attention: bool = False # Set True ONLY if you successfully installed flash-attn

config = TrainingConfig()

# === 2. MEMORY OPTIMIZATION ===
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# === 3. LOAD & FIX DATASET ===
print("📂 Loading Dataset...")
dataset = load_dataset("parquet", data_files=config.train_data_path, split="train")

# --- CRITICAL FIX: Convert Bytes to Images ---
print("🔄 Casting Image Column...")
dataset = dataset.cast_column("image", HFImage()) 
# Now dataset['image'] returns PIL objects, not bytes!

# Shuffle
dataset = dataset.shuffle(seed=42)
print(f"✅ Training samples: {len(dataset):,}")

# === 4. LOAD MODEL ===
print("🔄 Loading Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    config.model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    # Auto-fallback: Use Flash Attn if requested, else default to sdpa
    attn_implementation="flash_attention_2" if config.use_flash_attention else "sdpa"
)

processor = AutoProcessor.from_pretrained(
    config.model_id, 
    min_pixels=config.min_pixels, 
    max_pixels=config.max_pixels
)

# === 5. APPLY LORA ===
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# === 6. THE COLLATOR (Your structure was good!) ===
class Qwen2VLCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts, images = [], []
        for item in batch:
            # Construct Qwen Conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item['image']}, # This is now a PIL Image!
                        {"type": "text", "text": item['instruction']}
                    ]
                },
                {"role": "assistant", "content": [{"type": "text", "text": item['output']}]}
            ]
            
            # Formatting
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages)
            
            texts.append(text)
            images.append(image_inputs)

        # Tokenize
        batch_out = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        # Labels logic
        labels = batch_out["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch_out["labels"] = labels
        
        return batch_out

# === 7. TRAINER ===
args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_epochs,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="steps",
    save_steps=250, # Save more frequently
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False, # Required for custom collator
    report_to="none",
    dataloader_num_workers=4
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=Qwen2VLCollator(processor)
)

print("🚀 STARTING TRAINING...")
trainer.train()

trainer.save_model(config.output_dir + "/final_model")
processor.save_pretrained(config.output_dir + "/final_model")
print("✅ DONE!")