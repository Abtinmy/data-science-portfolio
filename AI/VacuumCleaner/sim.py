from copy import deepcopy
import json
from typing import List


class Simulator:
    def __init__(self, map: List, agent_loc: List) -> None:
        self.map = map
        self.agent_loc = agent_loc

    def take_action(self, action: str) -> None:
        """Takes a valid action and make the changes if needed to the global map."""

        x, y = self.agent_loc
        if action == 'C':
            self.map[x][y] = 0
            return

        delta_x, delta_y = {
            "U": (-1, 0),
            "D": (1, 0),
            "R": (0, 1),
            "L": (0, -1)
        }[action]
        new_x = max(0, min(len(self.map) - 1, x + delta_x))
        new_y = max(0, min(len(self.map[0]) - 1, y + delta_y))

        if self.map[new_x][new_y] == -1:
            return

        self.agent_loc = [new_x, new_y]


class Interface:
    def __init__(self):
        pass

    def evolve(self, state: Simulator, action: str) -> None:
        """
        Takes an action with a state and checks that the given action
        is valid in that state and has the correct format.
        """

        if type(action) is not str:
            raise "action is not a string"
        action = action.upper()
        if action not in self.valid_actions(state):
            raise "action is not valid"
        state.take_action(action)

    @staticmethod
    def copy_state(state: Simulator) -> Simulator:
        """Returns a deep copy of given state."""

        _copy = Simulator(None, None)
        _copy.map = deepcopy(state.map)
        _copy.agent_loc = deepcopy(state.agent_loc)
        return _copy

    @staticmethod
    def perceive(state: Simulator):
        """Returns what agent will see in a given state as a json."""
        return json.dumps({
            "map": state.map,
            "location": state.agent_loc
        })

    @staticmethod
    def goal_test(state: Simulator) -> bool:
        """Checks that goal has been reached."""

        for row in state.map:
            if row.count(1) != 0:
                return False
        return True

    @staticmethod
    def valid_actions(state: Simulator):
        """Returns a list of legal actions in the given state."""
        return ["U", "L", "R", "D", "C"]
