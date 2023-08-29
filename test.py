import os
import torch
import random
import pandas as pd

from agent.ppo import *
from environment.data import *
from environment.env import *


if __name__ == "__main__":
    algorithm = "RL" #["RL", "SD", "LD", "Random"]
    bay = 1
    num_crane = 2
    w_limit = 15.0

    file_path = "./input/sample.csv"
    if algorithm == "RL":
        model_path = './output/train/model/episode20000.pt'

    simulation_dir = './output/test/simulation/'
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    data = read_data(file_path, bay=bay, num_crane=num_crane)

    for i in range(1, num_crane + 1):
        if len(data["Crane-%d" % i]["sorting_plan"]) == 0:
            continue

        sorting_pile = data["Crane-%d" % i]["sorting_plan"]["pileno"].value_counts()
        relese_pile = data["Crane-%d" % i]["release_plan"]["pileno"].value_counts()
        initial_config = pd.concat([sorting_pile, relese_pile])
        initial_config.to_excel(simulation_dir + "initial_config_crane-%d.xlsx" % i)

        env = SteelStockYard(look_ahead=2, bay=bay, w_limit=w_limit,
                             num_from_pile=data["Crane-%d" % i]["num_from_pile"],
                             num_to_pile=data["Crane-%d" % i]["num_to_pile"],
                             num_release_pile=data["Crane-%d" % i]["num_release_pile"],
                             sorting_plan=data["Crane-%d" % i]["sorting_plan"],
                             release_plan=data["Crane-%d" % i]["release_plan"])

        if algorithm == "RL":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            agent = Network(env.state_size, env.action_size).to(device)
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            agent.load_state_dict(checkpoint['model_state_dict'])
        elif algorithm == "SD":
            agent = shortest_distance
        elif algorithm == "LD":
            agent = longest_distance
        else:
            agent = random_selection

        s = env.reset()
        done = False

        while not done:
            possible_actions = env.get_possible_actions()

            if algorithm == "RL":
                a = agent.get_action(s, possible_actions)
            else:
                a = agent(s, possible_actions)
            s_prime, r, done = env.step(a)
            s = s_prime

            if done:
                print(env.model.crane_dis_cum)
                break

        env.model.mointor.save(simulation_dir + 'event_log_{0}_crane-{1}.csv'.format(algorithm, i))