from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json 


def answer_questions(math_questions, batch_size, num_questions, model, tokenizer):
    responses = []
    for i in range(0, num_questions, batch_size):
        batch = math_questions[i : i + batch_size]
        batch_inputs = [f"[INST] {system_prompt}\n{item['problem']} [/INST]" for item in batch]
        
        inputs = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200)

        # Decode and store responses
        decoded_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        for item, response in zip(batch, decoded_responses):
            responses.append({
                "unique_id": item["unique_id"],
                "problem": item["problem"],
                "response": response
            })
        
    return responses


if __name__ == '__main__':

    #Trying to keep it model agnostic 
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    base_model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir="/n/holylabs/LABS/dam_lab/Users/hdiaz/measuring_learning_custom", trust_remote_code=True )

    #can I leave the field model in parameter does it have to say model string name
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="/n/holylabs/LABS/dam_lab/Users/hdiaz/measuring_learning_custom", trust_remote_code=True)
    # cache_dir is so that it doesn't save in cluster personal directory and hit 100G

    #curious to change system_prompt
    system_prompt= "Provide insightful explanations to grade-school students for these math questions"

    #Load in questions 
    with open("MATH_train.json", "r") as f:
        data = json.load(f)

    #pulling from json file:
    math_questions = [{"unique_id": item["unique_id"], "problem": item["problem"]} for item in data]

    # Batch processing setup
    batch_size = 8  # Adjust based on VRAM
    num_questions = len(math_questions)

    responses = answer_questions(
        math_questions = math_questions,
        batch_size = batch_size,
        num_questions = num_questions,
        model = base_model,
        tokenizer = tokenizer,
    )

    # Save responses to JSON file
    with open("phase_one_responses.json", "w") as f:
        json.dump(responses, f, indent=4)

