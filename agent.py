import random
from typing import Tuple, Dict

import numpy as np

from config import NUMBER_OF_STEPS, BREAKPOINT_SHORTCUT, ALPHA, GAMMA, EPSILON, N_PLANING
from utils import State, EncodedState, EncodedAction, ActionEncoder, StateEncoder, Action


class Agent:
    def __init__(self, env):
        self.env = env
        self.model: Dict[Tuple[EncodedState, EncodedAction], Tuple[State, int]] = {}
        self.s_e = StateEncoder(np.array(env.load_map(False).shape))
        self.a_e = ActionEncoder(env.possible_actions())

        self.q = np.zeros((self.s_e.size, self.a_e.size))

    def dyna_q_iteration(self):
        number_episodes = 0
        aggregated_reward = 0
        state = self.env.initial_position()
        aggregated_rewards = []
        for step in range(NUMBER_OF_STEPS):
            action = self._select_eps_greedy_action(state)
            next_state, reward, terminal = self.env.transition(
                state, action, step > BREAKPOINT_SHORTCUT
            )

            next_action = self._select_greedy_action(state)
            encoded_state, encoded_action = self.s_e.encode(state), self.a_e.encode(action)
            self.model[(encoded_state, encoded_action)] = next_state, reward

            self.q[(encoded_state, encoded_action)] += ALPHA * (
                GAMMA * self.q[(self.s_e.encode(next_state), self.a_e.encode(next_action))]
                + reward
                - self.q[(encoded_state, encoded_action)]
            )

            if terminal:
                print(f"Episode {number_episodes} ended at step {step}")
                number_episodes += 1
                aggregated_reward += 1
                state = self.env.initial_position()
            else:
                state = next_state

            for _ in range(N_PLANING):
                self._learn_model()

            aggregated_rewards += [aggregated_reward]
        return aggregated_rewards

    def _select_eps_greedy_action(self, state: State):
        if random.random() > EPSILON:
            return self._select_greedy_action(state)
        return Action(random.choice(self.env.possible_actions()))

    def _select_greedy_action(self, state: State):
        encoded_state = self.s_e.encode(state)
        return Action(self.a_e.decode(np.argmax(self.q[encoded_state, :])))

    def _learn_model(self):
        encoded_state, encoded_action = random.choice(list(self.model.keys()))
        next_state, reward = self.model[(encoded_state, encoded_action)]
        next_action = self._select_greedy_action(next_state)
        self.q[encoded_state, encoded_action] += ALPHA * (
                GAMMA * self.q[(self.s_e.encode(next_state), self.a_e.encode(next_action))]
                + reward
                - self.q[(encoded_state, encoded_action)]
        )

