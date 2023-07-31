import numpy as np


class Graphics:
    def __init__(self): pass

    @staticmethod
    def display(table: np.array):
        for i in range(9):
            row = ""
            if i % 3 == 0 and i > 0:
                print("------+-------+------")
            for j in range(9):
                if j % 3 == 0 and j > 0:
                    row += "| "
                row += "_" if table[i, j] == 0 else str(table[i, j])
                row += " "
            print(row)
