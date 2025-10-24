# MindGames (Social Deduction Track)
### Efficient Division- Team Phoebus

## Setup Instructions

### 1️ Clone the repository
```bash
git clone https://github.com/mihiraryaa/social_detection-mindgames.git
cd social_detection-mindgames
```

### 2 Create and activate a virtual environment (recommended)
```bash
python3 -m venv venv

source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows
```

### 3️ Install dependencies
```bash
pip install -r requirements.txt
```

## How to run

### Option 1: Run via CLI 
```bash
python main.py --observation "Welcome to Secret Mafia!\n You are Player 2\n Your role: Villager ......."
```
This will print the agent’s response directly to stdout.


### Option 2: Import and call programmatically
```python
from agent import agent

observation = "The player accuses you of being the mafia. What do you say?"
response = agent(observation)
print(response)
```

## Function Specification
```agent(observation: str) -> str``` <br><br>
**Input:**
observation (string ): describes the current environment state.<br>
**Output:** 
Returns a natural language string representing the agent’s next action or message.