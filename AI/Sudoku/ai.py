# https://link.springer.com/article/10.1007/s10732-007-9012-8

import json
import math
import random
from time import time
from typing import List, Dict, Tuple
import numpy as np
from gui import Graphics


class AI:
    def __init__(self):
        self.gui = Graphics()

    # the solve function takes a json string as input
    # and outputs the solved version as json
    def solve(self, problem):
        problem_data = json.loads(problem)
        table = np.array(problem_data["sudoku"])

        t0 = time()
        res = self.simulated_annealing(table)
        print("run time:", time() - t0)

        res_json = json.dumps({"sudoku": res.tolist()})
        return res_json

    def random_fill(self, table: np.array) -> np.array:
        empty_dict = {}
        fixed_points = np.zeros_like(table)
        for i in range(9):
            for j in range(9):
                if table[i, j] == 0:
                    unique_values_row = [x for x in np.unique(table[i, :]) if x != 0]
                    unique_values_col = [x for x in np.unique(table[:, j]) if x != 0]
                    unique_values_block = [x for x in np.unique(table[3 * (i // 3): 3 * (i // 3 + 1),
                                                                3 * (j // 3): 3 * (j // 3 + 1)]) if x != 0]
                    possible_values = [x for x in range(1, 10) if x not in np.unique(np.hstack((unique_values_block,
                                                                                                unique_values_col,
                                                                                                unique_values_row)))]
                    empty_dict[(i, j)] = possible_values
                else:
                    fixed_points[i, j] = 1

        change_fixed_points = True
        while empty_dict:
            empty_dict = {k: v for k, v in sorted(empty_dict.items(), key=lambda item: len(item[1]))}
            k = next(iter(empty_dict))
            v = empty_dict[k]
            empty_dict.pop(k)

            if len(v) == 1 and change_fixed_points:
                fixed_points[k[0], k[1]] = 1

            if len(v) > 1:
                change_fixed_points = False

            if len(v) == 0:
                change_fixed_points = False
                i, j = k[0], k[1]
                unique_values_block = [x for x in np.unique(table[3 * (i // 3): 3 * (i // 3 + 1),
                                                            3 * (j // 3): 3 * (j // 3 + 1)]) if x != 0]
                possible_values = [x for x in range(1, 10) if x not in unique_values_block]
                self.put_value(table, possible_values, empty_dict, k)
                continue

            self.put_value(table, v, empty_dict, k)

        return fixed_points

    def calculate_initial_temperature(self, table: np.array, fixed_points: np.array) -> float:
        differences = []
        for i in range(200):
            differences.append(self.cost_function(self.generate_state(table, fixed_points)))
        return np.std(differences)

    def simulated_annealing(self,
                            table: np.array,
                            cooling_factor: float = 0.99,
                            heating_factor: float = 2,
                            max_stuck_count: int = 100) -> np.array:
        fixed_points = self.random_fill(table)
        current_cost = self.cost_function(table)
        if current_cost == 0:
            return table

        temp = self.calculate_initial_temperature(table, fixed_points)
        num_iter = (81 - np.sum(fixed_points)) ** 2
        stuck_counter = 0
        while True:
            prev_cost = current_cost
            for i in range(num_iter):
                new_table = self.generate_state(table, fixed_points)
                new_cost = self.cost_function(new_table)
                # print(new_cost)

                if new_cost == 0:
                    return new_table

                delta = new_cost - current_cost
                if np.random.rand() < math.exp(-delta / temp):
                    table = new_table
                    current_cost = new_cost

            temp *= cooling_factor
            if new_cost >= prev_cost:
                stuck_counter += 1
            else:
                stuck_counter = 0

            if stuck_counter > max_stuck_count:
                temp += heating_factor

    @staticmethod
    def put_value(table: np.array, possible_values: List[int], points: Dict[Tuple, List], k: Tuple) -> None:
        value = random.choice(possible_values)
        table[k[0], k[1]] = value
        for k2, v2 in points.items():
            if (
                    k[0] == k2[0] or
                    k[1] == k2[1] or
                    (
                            3 * (k[0] // 3) <= k2[0] < 3 * (k[0] // 3 + 1) and
                            3 * (k[1] // 3) <= k2[1] < 3 * (k[1] // 3 + 1)
                    )
            ):
                if value in v2:
                    v2.remove(value)

    @staticmethod
    def cost_function(table: np.array) -> int:
        cost = 0
        for i in range(9):
            cost += (9 - len(np.unique(table[i, :]))) + (9 - len(np.unique(table[:, i])))
        return cost

    @staticmethod
    def generate_state(table: np.array, fixed_points: np.array) -> np.array:
        new_table = table.copy()
        sum_var_points = 9
        block = None
        i, j = 0, 0
        while sum_var_points > 7:
            block_number = random.randint(0, 8)
            i, j = 3 * (block_number // 3) + 1, 3 * (block_number % 3) + 1
            block = fixed_points[3 * (i // 3): 3 * (i // 3 + 1), 3 * (j // 3): 3 * (j // 3 + 1)]
            sum_var_points = np.sum(block)

        list_var_points = []
        for x in range(3):
            for y in range(3):
                if block[x, y] == 0:
                    list_var_points.append([x, y])

        point1, point2 = random.sample(list_var_points, k=2)
        new_table[point1[0] + 3 * (i // 3), point1[1] + 3 * (j // 3)] = table[
            point2[0] + 3 * (i // 3), point2[1] + 3 * (j // 3)]
        new_table[point2[0] + 3 * (i // 3), point2[1] + 3 * (j // 3)] = table[
            point1[0] + 3 * (i // 3), point1[1] + 3 * (j // 3)]

        return new_table

