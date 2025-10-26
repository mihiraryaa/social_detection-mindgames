"""
Track 1: Social Detection Track
This script connects your agent to the online competition for the social detection track.
Environment: SecretMafia-v0
"""

import textarena as ta
from qwafia import revac

MODEL_NAME = "Revac"
MODEL_DESCRIPTION="Review and act"
team_hash="MG25-0F4427A48C"



multiagent=revac()

env = ta.make_mgc_online(
    track="Social Detection", 
    model_name=MODEL_NAME,
    model_description=MODEL_DESCRIPTION,
    team_hash=team_hash,
    agent=multiagent,
    small_category=True 
)
env.reset(num_players=1)

done = False
while not done:
    player_id, observation = env.get_observation()
    action = multiagent(observation)
    done, step_info = env.step(action=action)

rewards, game_info = env.close() 