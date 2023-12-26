from agent import Agent
from config import RELATIVE_MAZE_PATH
from env import Env
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = Env(RELATIVE_MAZE_PATH)
    exploring_agent = Agent(env, use_exploration_bonus=True)
    basic_agent = Agent(env, use_exploration_bonus=False)
    aggregated_rewards_exploring_agent = exploring_agent.dyna_q_iteration()
    aggregated_rewards_basic_agent = basic_agent.dyna_q_iteration()

    plt.plot(
        range(len(aggregated_rewards_exploring_agent)),
        aggregated_rewards_exploring_agent,
        label="exploring",
    )
    plt.plot(
        range(len(aggregated_rewards_basic_agent)),
        aggregated_rewards_basic_agent,
        label="base",
    )

    plt.legend()
    plt.show()
    print(1)

    # exploring_agent.q[StateEncoder(np.array([6, 9])).encode(np.array([0, 7])), ActionEncoder(env.possible_actions()).encode([0, 1])]
