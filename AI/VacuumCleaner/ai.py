import random
from time import time
from typing import List
from sim import Simulator, Interface
import json


class Agent:
    def __init__(self):
        self.predicted_actions = []

    def act(self, percept) -> str:
        """
        Takes a perception from the environment in the form of
        json string and returns an action as a string.
        """

        sensor_data = json.loads(percept)
        alg = self.bfs

        if not self.predicted_actions:
            t0 = time()
            initial_state = Simulator(sensor_data['map'], sensor_data['location'])
            self.predicted_actions = alg(initial_state)
            print("run time:", time() - t0)

        return self.predicted_actions.pop()

    @staticmethod
    def bfs(root_game: Simulator) -> List[str]:
        interface = Interface()
        q = [[root_game, []]]

        while q:
            node = q.pop(0)

            actions_list = interface.valid_actions(node)
            random.shuffle(actions_list)

            for action in actions_list:
                child_state = interface.copy_state(node[0])
                interface.evolve(child_state, action)
                q.append([child_state, [action] + node[1]])

                if interface.goal_test(child_state):
                    return [action] + node[1]
