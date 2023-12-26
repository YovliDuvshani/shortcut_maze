from functools import lru_cache
from typing import Tuple, List

import numpy as np

from utils import State, Action, MapFalselySpecified


class Env:
    def __init__(self, map_path: str):
        self.map_path = map_path

    def initial_position(self) -> State:
        return State(np.argwhere(self.load_map(False) == 0))[0]

    def transition(
        self, state: State, action: Action, shortcut_available: bool
    ) -> Tuple[State, int, bool]:
        maze_map = self.load_map(shortcut_available)
        new_state = State(state + action)
        if (
            (new_state >= maze_map.shape).any()
            or (new_state < np.zeros(2)).any()
            or maze_map[new_state[0], new_state[1]] == -1
        ):
            new_state = state

        terminal = maze_map[new_state[0], new_state[1]] == 2

        return new_state, int(terminal), terminal

    @lru_cache
    def load_map(self, shortcut_available: bool) -> np.array:
        with open(self.map_path) as f:
            map_txt = [row[:-1] for row in f.readlines()]

            map_array = np.zeros((len(map_txt), len(map_txt[0])))
            for i, row in enumerate(map_txt):
                for j, area in enumerate(row):
                    if area == "N" or (shortcut_available and area == "H"):
                        map_array[i, j] = 1
                    elif area == "S":
                        map_array[i, j] = 0
                    elif area == "B" or (not shortcut_available and area == "H"):
                        map_array[i, j] = -1
                    elif area == "G":
                        map_array[i, j] = 2
                    else:
                        raise MapFalselySpecified
        return map_array

    @staticmethod
    def possible_actions() -> List[Action]:
        return [
            np.array([i, j])
            for i in range(-1, 2)
            for j in range(-1, 2)
            if abs(i) + abs(j) <= 1
        ]
