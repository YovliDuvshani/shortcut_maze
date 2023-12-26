import numpy as np

from agent import Agent
from config import RELATIVE_MAZE_PATH
from env import Env
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = Env(RELATIVE_MAZE_PATH)
    agent = Agent(env)
    aggregated_rewards = agent.dyna_q_iteration()
    plt.plot(range(len(aggregated_rewards)), aggregated_rewards)
    
    print(1)
