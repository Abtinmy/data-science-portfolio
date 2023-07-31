from ai import Agent
from gui import Graphics
from sim import Simulator, Interface


sample_input_json = {
    "location": [1, 1],
    "map": [
        [-1, 1],
        [-1, 0],
        [0, 1]
    ]
}


if __name__ == "__main__":
    game = Simulator(sample_input_json['map'], sample_input_json['location'])
    interface = Interface()
    agent = Agent()
    gui = Graphics()

    action_count = 0
    print("initial map")
    gui.display(game)
    while not (interface.goal_test(game)):
        action = agent.act(interface.perceive(game))
        print("attempting", action)

        interface.evolve(game, action)
        gui.display(game)
        print("\n")

        action_count += 1

    print(
        "\n\nпобеда!!!",
        "\nyour cost (number of actions):", action_count,
        '\n\ncurrent map state:'
    )
    gui.display(game)
