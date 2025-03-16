import os
import torch
import bitsandbytes as bnb
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
from typing import List, Dict, Any

# 🚀 Step 1: Load or Cache Dataset
dataset_path = "/home/guangyu/MD/dataset/videos/hf_video_dataset_processed"

if os.path.exists(dataset_path):
    print("✅ Loading cached dataset...")
    dataset = load_from_disk(dataset_path)
else:
    print("⚠️ Processed dataset not found. Loading raw dataset...")
    dataset = load_from_disk("/home/guangyu/MD/dataset/videos/hf_video_dataset")

# 🚀 Step 2: Load Model and Processor
model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# 🚀 Step 3: Add LoRA Adaptation
lora_config = LoraConfig(
    r=2,  
    lora_alpha=16,  
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.05,  
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 🚀 Step 4: Custom Data Collator for Dynamic Padding
import torch

from typing import List, Dict, Any

import torch
from typing import List, Dict, Any


import torch
from typing import List, Dict, Any


class LlavaNextVideoDataCollatorWithPadding:
    """Custom Data Collator for Video + Text Fine-Tuning with Manual Padding"""
    
    def __init__(self, processor, model_max_length: int = 1600):
        self.processor = processor
        self.model_max_length = model_max_length  # Model's max sequence length
        self.PAD_TOKEN_ID = self.processor.tokenizer.pad_token_id  # Get padding token ID
        self.image_token = 151647
        print('self.image_token', self.image_token)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Manually pads inputs, attention masks, labels, and video frames"""

        # Extract fields from features (flattening (1, seq_length) → (seq_length))
        input_ids = [torch.tensor(feat["input_ids"]).squeeze(0) for feat in features]
        attention_masks = [torch.tensor(feat["attention_mask"]).squeeze(0) for feat in features]
        pixel_values_videos = [feat["pixel_values_videos"][0] for feat in features]
        pixel_values_videos = torch.tensor(pixel_values_videos)

 
        #print(pixel_values_videos.shape)

        # ✅ Dynamically determine max_length in batch (shape[1])
        batch_max_length = min(self.model_max_length, max(seq.shape[0] for seq in input_ids))
        #print('batch_max_length:', batch_max_length)

        # ✅ Manual Padding for `input_ids`
        padded_input_ids = []
        for cur_input_ids in input_ids:
            # ✅ Truncate if longer than batch_max_length
            cur_input_ids = cur_input_ids[:batch_max_length]  

            pad_length = max(0, batch_max_length - cur_input_ids.shape[0])  # Ensure pad_length is non-negative
            padded_seq = torch.cat([
                cur_input_ids,  # Original sequence
                torch.full((pad_length,), self.PAD_TOKEN_ID, dtype=cur_input_ids.dtype)  # Padding
            ], dim=0)
            #print('padded_seq:', padded_seq.shape)
            padded_input_ids.append(padded_seq)

        padded_input_ids = torch.stack(padded_input_ids)

        # ✅ Manual Padding for `attention_mask`
        padded_attention_masks = []
        for mask in attention_masks:
            mask = mask[:batch_max_length]  # ✅ Truncate if too long
            pad_length = max(0, batch_max_length - mask.shape[0])  # Ensure pad_length is non-negative
            padded_mask = torch.cat([
                mask,  # Original attention mask
                torch.zeros((pad_length,), dtype=mask.dtype)  # Padding with 0s
            ], dim=0)
            padded_attention_masks.append(padded_mask)

        padded_attention_masks = torch.stack(padded_attention_masks)

        # ✅ Manual Padding for `labels` (Ignore padding in loss)
        padded_labels = padded_input_ids.clone()
        padded_labels[padded_labels == self.PAD_TOKEN_ID] = -100  # Set padding tokens to -100 for loss masking
        padded_labels[padded_labels == self.image_token] = -100


        # ✅ Stack `pixel_values_videos` into a single tensor
        #print('pixel_values_videos', pixel_values_videos)
       # padded_pixel_values_videos = torch.cat(pixel_values_videos, dim=0)

        return {
            "input_ids": padded_input_ids,  # (batch_size, batch_max_length)
            "attention_mask": padded_attention_masks,  # (batch_size, batch_max_length)
            "labels": padded_labels,  # (batch_size, batch_max_length)
            "pixel_values_videos": pixel_values_videos  # (batch_size, num_frames, H, W, C)
        }



# 🚀 Step 5: Data Preprocessing Function
def preprocess_data(batch):
    """Convert dataset samples into model-compatible format."""
    video_path = os.path.join("/home/guangyu/MD/dataset", batch["video_path"])
    question = batch["question"]
    options = batch["options"]
    correct_answer = batch["correct_answer"]
    
    # 📝 Generate Prompt
    prompt_text = (
        "Instruction: You are given a video along with a question and multiple-choice options. "
        "Analyze the video carefully and select the most accurate answer from the given choices.\n\n"
    )
    prompt_text += f"Question: {question}\nOptions:\n"
    for key, option in options.items():
        prompt_text += f"{key}: {option}\n"
    
    prompt_text += "\n Assistant: The correct option is " + correct_answer

    # 📝 Prepare Model Input
    conversation = [{"role": "user", "content": [
        {"type": "video", "path": video_path},
        {"type": "text", "text": prompt_text}
    ]}]

    inputs = processor.apply_chat_template(
        conversation,
        num_frames=2,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    return inputs

# 🚀 Step 6: Process and Cache Dataset
if not os.path.exists(dataset_path):  
    print("⚠️ Processing and caching dataset...")
    dataset = dataset.map(preprocess_data, batched=False)
    dataset.save_to_disk(dataset_path)  
    print("✅ Dataset cached successfully!")
else:
    print("✅ Using cached dataset.")

# 🚀 Step 7: Train-Test Split (80% Train, 20% Eval)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
print('dataset', dataset)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# 🚀 Step 8: Initialize Custom Data Collator
data_collator = LlavaNextVideoDataCollatorWithPadding(
    processor=processor,
    model_max_length=20000  # Set max_length based on model's max limit
)
print('dataset', dataset)
# 🚀 Step 9: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./llava-lora-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  
    save_total_limit=2,
    save_steps=5000,
    num_train_epochs=5,
    fp16=True,  
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",  
    eval_steps=10,
    save_strategy="steps",
    optim="adamw_bnb_8bit",
    learning_rate=2e-4,
    warmup_ratio=0.03,
    weight_decay=0.01,
    report_to="none"
)

# 🚀 Step 10: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator  # ✅ Custom collator dynamically pads batch sequences
)

# 🚀 Step 11: Start Training
trainer.train()

from tqdm.auto import tqdm

print("🎉 LoRA Fine-Tuning Completed Successfully!")

model.eval()

def eval_preprocess_data(video_path, question, options):
    """Convert dataset samples into model-compatible format."""
    video_path = os.path.join("/home/guangyu/MD/dataset", video_path)
    
    # 📝 Generate Prompt
    prompt_text = (
        "Instruction: You are given a video along with a question and multiple-choice options. "
        "Analyze the video carefully and select the most accurate answer from the given choices.\n\n"
    )
    prompt_text += f"Question: {question}\nOptions:\n"
    for key, option in options.items():
        prompt_text += f"{key}: {option}\n"
    
    prompt_text += "\n Assistant: The correct option is " 

    # 📝 Prepare Model Input
    conversation = [{"role": "user", "content": [
        {"type": "video", "path": video_path},
        {"type": "text", "text": prompt_text}
    ]}]

    inputs = processor.apply_chat_template(
        conversation,
        num_frames=2,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    return inputs

eval_dataset = eval_dataset.remove_columns(["pixel_values_videos", "input_ids", "attention_mask"])  # Add more if needed
print('Eval datast', eval_dataset)


correct_count = 0
total_count = len(eval_dataset)

for data in tqdm(eval_dataset, desc="Evaluating Zero-Shot Performance"):
    video_path = '/home/guangyu/MD/dataset/'+data["video_path"]
    question = data["question"]
    options = data["options"]
    correct_answer = data["correct_answer"]
    
    # Generate prompt
    inputs = eval_preprocess_data(video_path, question, options)
    

    out = model.generate(**inputs, max_new_tokens=100)
    ans = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].split('Assistant: The correct option is')[-1]
            
    # Check if model's response contains the correct answer
    #print('\n\n\n\n\n\ncheck answer')
    #print('out', correct_answer, type(ans), len(ans))
    #print('ans:', ans)
    
    if correct_answer in ans:       
        correct_count += 1
        #print('correct count *****', correct_count)
    #print('\n\n\n\n\n')

# Calculate accuracy
accuracy = (correct_count / total_count) * 100
print(f"Zero-shot accuracy: {accuracy:.2f}%")
