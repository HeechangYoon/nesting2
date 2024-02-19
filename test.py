
import torch
import random
import pandas as pd

from agent.ppo import *
from environment.data import *
from environment.env import *
from environment.simulation import *

from PIL import Image

def create_gif(image_data_list, file_path, duration=200):
    frames = []
    for image_data in image_data_list:
        image_data = (2 - np.array(image_data)) * 255 / 2  # Convert 0s and 1s to 0s and 255s
        image_data = image_data.astype(np.uint8)  # Convert to unsigned 8-bit integers
        image = Image.fromarray(image_data, 'L')  # Create image from numpy array
        frames.append(image)

    frames[0].save(file_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)

if __name__ == "__main__":
    algorithm = "RL"

    file_path = "./input/sample.csv"
    if algorithm == "RL":
        model_path = './output/train/model/model.pt'

    image_dir = './output/test/image/'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    result_dir = './output/test/result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # data = generate_data(read_data())

    env = HiNEST(rec=True)

    if algorithm == "RL":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        agent = Network(env.state_size, env.x_action_size, env.a_action_size).to(device)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        agent.load_state_dict(checkpoint['model_state_dict'])
    # elif algorithm == "SD":
    #     agent = shortest_distance
    # elif algorithm == "LD":
    #     agent = longest_distance
    # else:
    #     agent = random_selection

    reward_list = list()
    efficiency_list = list()
    batch_rate_list = list()
    for i in range(1000):
        s = env.reset()
        zero = np.zeros((210, 270))
        zero[:, 45:] += s[:, 45:, 0]
        image_list = list()
        image_list.append(zero)
        done = False

        while not done:
            possible_actions = env.get_possible_actions()

            if algorithm == "RL":
                a = agent.get_action(s, possible_actions)
                possible_x = env.get_possible_positions(a)
                a_x = agent.get_position(s, possible_x)
            else:
                a, a_x = agent(s, possible_actions)
            next_state, reward, efficiency, batch_rate, done, overlap, temp = env.step((a_x, a))
            s = next_state

            zero = np.zeros((210, 270))
            zero[:, 45:] += s[:, 45:, 0]
            if overlap:
                zero[:, :45] += temp

                image_list.append(zero)
            else:
                zero[:, :45] += env.model.plate.PixelPlate
                image_list.append(zero)

            if done:
                reward_list.append(reward)
                efficiency_list.append(efficiency)
                batch_rate_list.append(batch_rate)

                df = pd.DataFrame({'reward': reward_list, 'efficiency': efficiency_list, 'batch_rate': batch_rate_list})

                df.to_excel(result_dir + 'result.xlsx', index=True)

                file_path = image_dir + 'result{0}.gif'.format(i+1)
                create_gif(image_list, file_path)

                break