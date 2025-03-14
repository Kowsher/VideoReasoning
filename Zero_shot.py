import os
import torch
from datasets import load_from_disk
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from tqdm.auto import tqdm

def generate_prompt(video_path, question, options):
    """Generate a structured prompt for the model."""
    prompt_text = (
        "Instruction: You are given a video along with a question and multiple-choice options. "
        "Analyze the video carefully and select the most accurate answer from the given choices.\n\n"
    )
    prompt_text += f"Question: {question}\nOptions:\n"
    for key, option in options.items():
        prompt_text += f"{key}: {option}\n"
    
    prompt_text += "\n The correct option is: "

    #print('prompt_text: ', prompt_text)
    
    
    return [
        {"role": "user", "content": [
            {"type": "video", "path": video_path},
            {"type": "text", "text": prompt_text}
        ]}
    ]


def evaluate_zero_shot_performance(dataset_path: str, model_name: str):
    """Evaluate zero-shot performance on the given dataset."""
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    # Load model and processor
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)
    
    correct_count = 0
    total_count = len(dataset)
    
    for data in tqdm(dataset, desc="Evaluating Zero-Shot Performance"):
        video_path = '/home/guangyu/MD/dataset/'+data["video_path"]
        question = data["question"]
        options = data["options"]
        correct_answer = data["correct_answer"]
        
        # Generate prompt
        conversation = generate_prompt(video_path, question, options)
        
        
        # Process input
        inputs = processor.apply_chat_template(
            conversation,
            num_frames=8,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, torch.float16)
      
        

        out = model.generate(**inputs, max_new_tokens=100)
        ans = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].split('The correct option is:')[-1]
                
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
    
    return accuracy

# Example usage
evaluate_zero_shot_performance("/home/guangyu/MD/dataset/videos/hf_video_dataset", "llava-hf/llava-onevision-qwen2-7b-ov-hf")

# frames = 8, Zero-shot accuracy: 79.06%
# fame = 16, Zero-shot accuracy: 79.25%
# fame = 32, Zero-shot accuracy: 74.54%
# frame = 64, Zero-shot accuracy: 70.81%

