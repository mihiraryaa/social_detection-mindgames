from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import re
from agent import Agent
from prompts import *

# Load base model ONCE
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

# Load first adapter as default
model = PeftModel.from_pretrained(base_model, "mihirArya/qwafia_10")

# Load second adapter (give it a name)
model.load_adapter("mihirArya/qwafiaB_10", adapter_name="final_action_agent")

# Now you have two adapters loaded:
# - "default" (agent1)
# - "final_action_agent" 

# Switch between them
def review_agent(observation_state):
    model.set_adapter("default")  # Switch to agent1
    #add="When you are giving the mafia scenarios, rank it be possibility"
    #add="When you are giving the mafia scenarios, rank them by possibility"
    messages = [
    {"role": "system", "content": reviewer_prompt},
    {"role": "user", "content": f"""## Here's the current observation state: \n\n {observation_state}
    ## Your task is: 
            <task>
        Analyze the observation state using the structure above:
        
        1. Extract FACTS (only [GAME] confirmed events), including your player id
        2. Profile each player (behavior, claims, likelihood)
        3. Identify hallucinations/false claims
        4. Provide mafia scenarios (if in team village)
        5. Specify next action recommendation to maximize the win probability for your player id (Don't include your [player id ] while giving the response)
        
        Be thorough and precise. Distinguish facts from claims.
        </task>
    """},
    ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    response_ids = model.generate(
    **inputs,
    max_new_tokens=2560,        
    temperature=0.3,          
    top_p=0.85,                
    repetition_penalty=1.1,  
    do_sample=True,            
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)[0][len(inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(response,'\n\n\n')
    return response 

def final_action_agent(observation_state, review):
    model.set_adapter("final_action_agent")  # Switch to final_action_agent
    messages = [
    {"role": "system", "content": final_action_prompt},
    {"role": "user", "content": f"""# Here is the observation state: \n\n {observation_state}\n\n # Following is the strategic review: \n\n{review}
    # Your Task
    <task>
Based on the strategic review and current observation:
1. Identify current phase from [GAME] announcements
2. Determine your role and valid actions
3. Execute the recommended action from the review
4. Use correct output format
5. Don't use your [player id] while giving your response
6. Return your final response in <response> </response> tags

Provide your action now:
</task>  
    
    """},
    ]
    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    response_ids = model.generate(**inputs, max_new_tokens=1000, temperature=0.6)[0][len(inputs.input_ids[0]):].tolist()
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    try:
        pattern = r"<response>(.*?)</response>"
        all_chunks = re.findall(pattern, response, re.DOTALL)
        return all_chunks[0].strip()
    except:
        response=re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        return response.strip()



def agent(observation_state):
    review = review_agent(observation_state)
    final_decision = final_action_agent(observation_state, review)
    return final_decision
        



class revac(Agent):
    """ Revac  """
    def __init__(self):
        super().__init__()

    def __call__(self, observation: str) -> str:
        res=agent(observation)
        
        return res 