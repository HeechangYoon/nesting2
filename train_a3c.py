import os
import vessl

# from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Manager
from cfg_a3c import get_cfg
from agent.a3c import *
from environment.env import *

from PIL import Image


def save_image(image_data, file_path):
    image_data = np.array(image_data) * 255  # Convert 0s and 1s to 0s and 255s
    image_data = image_data.astype(np.uint8)  # Convert to unsigned 8-bit integers
    image = Image.fromarray(image_data, 'L')  # Create image from numpy array
    image.save(file_path)  # Save image to file

def create_gif(image_data_list, file_path, duration=200):
    frames = []
    for image_data in image_data_list:
        image_data = (2 - np.array(image_data)) * 255 / 2  # Convert 0s and 1s to 0s and 255s
        image_data = image_data.astype(np.uint8)  # Convert to unsigned 8-bit integers
        image = Image.fromarray(image_data, 'L')  # Create image from numpy array
        frames.append(image)

    frames[0].save(file_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)


if __name__ == "__main__":
    cfg = get_cfg()
    vessl.init(organization="snu-eng-dgx", project="nesting", hp=cfg)
    torch.multiprocessing.set_start_method('spawn')

    n_episode = cfg.n_episode
    n_agents = cfg.n_agents
    lr = cfg.lr
    gamma = cfg.gamma
    lmbda = cfg.lmbda
    t_max = cfg.t_max

    model_dir = '/output/train/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = '/output/train/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    image_dir = '/output/train/image/'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, efficiency, batch_rate, loss\n')

    # writer = SummaryWriter(log_dir)

    env = HiNEST(look_ahead=5)
    global_network = Network(env.state_size, env.x_action_size, env.a_action_size).to(device)
    global_network.share_memory()  # share the global parameters in multiprocessing
    shared_optimizer = SharedAdam(global_network.parameters(), lr=lr)  # global optimizer
    global_episode = mp.Value('i', 0)
    global_episode_reward = mp.Value('d', 0.0)
    global_efficiency = mp.Value('d', 0.0)
    global_batch_rate = mp.Value('d', 0.0)
    global_loss = mp.Value('d', 0.0)
    res_queue = mp.Queue()

    with Manager() as manager:
        lock = manager.Lock()
        workers = [Worker(i + 1, n_episode, t_max, gamma, lmbda, global_network, shared_optimizer,lock,
                          global_episode, global_episode_reward, global_efficiency, global_batch_rate, global_loss, res_queue)
                   for i in range(n_agents)]
        [w.start() for w in workers]
        res = []  # record episode reward to plot
        while True:
            res = res_queue.get()
            if res is not None:
                if res[0] % cfg.save_every == 0:
                    torch.save({"episode": res[0],
                                "model_state_dict": global_network.state_dict(),
                                "optimizer_state_dict": shared_optimizer.state_dict()},
                               model_dir + "model.pt")

                with open(log_dir + "train_log.csv", 'a') as f:
                    f.write('%d,%1.2f,%1.2f,%1.2f,1.2%f\n' % (res[0], res[1], res[2], res[3], res[4]))

                vessl.log(step=res[0], payload={'Episode_reward': res[1]})
                vessl.log(step=res[0], payload={'Efficiency': res[2]})
                vessl.log(step=res[0], payload={'Batch_rate': res[3]})
                vessl.log(step=res[0], payload={'Loss': res[4]})

                # writer.add_scalar("Training/Episode_reward", res[1], res[0])
                # writer.add_scalar("Training/Efficiency", res[2], res[0])
                # writer.add_scalar("Training/Batch_rate", res[3], res[0])
                # writer.add_scalar("Training/Loss", res[4], res[0])

            else:
                break
        [w.join() for w in workers]

    # writer.close()