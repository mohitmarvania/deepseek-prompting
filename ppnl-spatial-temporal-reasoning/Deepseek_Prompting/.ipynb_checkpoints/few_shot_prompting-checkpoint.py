"""
Implementing 2-shot prompting along with <think>...</think> tag to check the response of the model.
"""
import json
import time
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def create_prompt(world, nl_description):
    # Parsing the start and goal coordinates from nl_description
    # Format: "Go from (x1,y1) to (x2,y2)"
    coords_part = nl_description.split("Go from ")[1]
    start_coords, goal_coords = coords_part.split(" to ")
    
    # Extract start coordinates
    start_x, start_y = map(int, start_coords.strip("()").split(","))
    start = (start_x, start_y)
    
    # Extract goal coordinates
    goal_x, goal_y = map(int, goal_coords.strip("()").split(","))
    goal = (goal_x, goal_y)
    
    # Find obstacles
    obstacles = []
    for i in range(len(world)):
        for j in range(len(world[0])):
            if world[i][j] == 1:
                obstacles.append(f"({i},{j})")
    
    prompt = f"""#Path Planning Task
    
    ## Instruction:
    You are a path planning expert. Find the optimal path in a grid environment, avoiding obstacles while tracking position at each step. (0,0) is the top-left cell of the grid; (N-1,N-1) means N-1th row and N-1th column. Take your time to carefully analyze the grid configuration before planning the path.

    Here are few examples:
    ###
    Task: You are in a 6 by 6 world. There are obstacles that you have to avoid at: (2,1). Go from (0,1) to (3,4)
    Actions: right right right down down down
    ###
    Task: You are in a 6 by 6 world. There are obstacles that you have to avoid at: (0,4). Go from (5,4) to (2,4)
    Actions: up up up
    ###
    
    Now solve this:
    Task: You are in a 6 by 6 world. There are obstacles that you have to avoid at: {', '.join(obstacles)}. Go from ({start[0]},{start[1]}) to ({goal[0]},{goal[1]})
    
    **Constraints**
    - Allowed Movements: [Up/Down/Right/Left]
    - Position tracking required
    
    **Output Format**
    <Think>
    [Your detailed reasoning about obstacle avoidance and path selection]
    </Think>
    |summary|
    optimal Path: [Movements sequence]
    
    <Think>"""
    
    return prompt


def main():
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Load model and tokenizer
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # load test data
    with open('../single_goal/6x6worlds/test_unseen.json', 'r') as f:
        test_data = json.load(f)
        
    results = []
    
    for example in tqdm(test_data):
        try:
            prompt = create_prompt(example['world'], example['nl_description'])
            
            # Generate Response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.2,
                max_new_tokens=4000,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated path from response
            if not os.path.exists('outputs'):
                os.makedirs('outputs')
                
            generated_path = response.split("|summary|")[-1].strip()
            if "Optimal path:" in generated_path:
                generated_path = generated_path.split("Optimal path:")[-1].strip()
            
            result = {
                "english": example['nl_description'],
                "ground_truth": example['agent_as_a_point'],
                "generated": response,
                "optimal_path": generated_path
            }
            
            results.append(result)
            
            # Save the results after each example
            with open('outputs/deepseek_few_shot_full_results.json', 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
        
        # Add small delay between requests
        time.sleep(2)


if __name__ == "__main__":
    main()
