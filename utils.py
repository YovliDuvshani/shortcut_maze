from typing import List

import numpy as np

State = np.array
Action = np.array

EncodedState = int
EncodedAction = int


class StateEncoder:
    def __init__(self, maze_shape: np.array):
        self.maze_shape = maze_shape
        self.size = np.prod(maze_shape)

    def encode(self, state: State):
        return np.ravel_multi_index(state, self.maze_shape)

    def decode(self, encoded_state: EncodedState):
        return np.unravel_index(encoded_state, self.maze_shape)


class ActionEncoder:
    def __init__(self, possible_actions: List[Action]):
        self.possible_actions = possible_actions
        self.size = len(possible_actions)

    def encode(self, action):
        for i, possible_action in enumerate(self.possible_actions):
            if np.array_equal(action, possible_action):
                return i
        raise Exception

    def decode(self, encoded_action: EncodedAction):
        return self.possible_actions[encoded_action]


class MapFalselySpecified(Exception):
    pass
