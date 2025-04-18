"""
Implementing only zero-shot prompting on deepseek R1 distill Qwen 7B model for Path Planning task.
As per the deepseek research paper "Deepseek R1 degrades its performance if few-shot prompting is applied. Thus it is 
recommended to apply zero-shot prompting".
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
    You are a path planning expert. Find the optimal path in a grid environment, avoiding obstacles while tracking position at each step. Take your time to carefully analyze the grid configuration before planning the path.
    
    **Grid Configuration**
    Size: 6x6
    Start: ({start[0]},{start[1]})
    Goal: ({goal[0]}. {goal[1]})
    Obstacles: [{', '.join(obstacles)}]
    (0,0) is located in the upper-left corner and (5,5) is located at bottom-right corner at 5th row and 5th column.
    
    **Constraints**
    - Allowed Movements: [Up/Down/Right/Left]
    - left goes from (x-1,y) to (x,y)
    - right goes from (x,y) to (x+1,y)
    - up goes from (x,y) to (x,y-1)
    - down goes from (x,y) to (x,y+1)
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
    with open('../../single_goal/6x6worlds/test_unseen.json', 'r') as f:
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
            with open('outputs/deepseek_zero_shot_expl_results.json', 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
        
        # Add small delay between requests
        time.sleep(2)


if __name__ == "__main__":
    main()
