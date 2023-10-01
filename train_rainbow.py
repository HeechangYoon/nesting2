import os
import vessl

# from torch.utils.tensorboard import SummaryWriter
from cfg_rainbow import get_cfg
from agent.rainbow import *
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

    n_step = cfg.n_step
    capacity = cfg.capacity
    alpha = cfg.alpha
    beta_start = cfg.beta_start
    beta_steps = cfg.beta_steps
    batch_size = cfg.batch_size
    N = cfg.N
    lr = cfg.lr
    lr_step = cfg.lr_step
    lr_decay = cfg.lr_decay
    gamma = cfg.gamma
    tau = cfg.tau

    model_dir = '/output/train/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = '/output/train/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    simulation_dir = '/output/train/simulation/'
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    image_dir = '/output/train/image/'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    env = HiNEST(look_ahead=5)
    agent = Agent(env.state_size, env.x_action_size, env.a_action_size, capacity, alpha, beta_start, beta_steps,
                  n_step, batch_size, lr, lr_step, lr_decay, tau, gamma, N)
    # writer = SummaryWriter(log_dir)

    if cfg.load_model:
        checkpoint = torch.load(cfg.model_path)
        start_episode = checkpoint['episode'] + 1
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        agent.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_episode = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, efficiency, batch_rate, loss\n')

    for e in range(start_episode, cfg.n_episode + 1):
        state = env.reset()
        update_step = 0
        r_epi = 0.0
        efficiency = 0.0
        batch_rate = 0.0
        loss_list = []
        done = False

        if cfg.get_gif:
            image_list = list()
            zero = np.zeros((210, 270))
            zero += state[:, :, 0]
            image_list.append(zero)

        while not done:
            possible_a = env.get_possible_actions()
            angle = agent.get_angle(state, possible_a)
            possible_x = env.get_possible_positions(a)
            position = agent.get_position(state, possible_x)

            next_state, reward, efficiency, batch_rate, done, overlap, temp = env.step((angle, position))
            loss = agent.step(state, angle, position, reward, next_state, done)
            if loss is not None:
                loss_list.append(loss)

            state = next_state
            r_epi += reward

            if cfg.get_gif:
                zero = np.zeros((210, 270))
                zero[:, 45:] += state[:, 45:, 0]
                if overlap:
                    zero[:, :45] += temp
                    image_list.append(zero)
                else:
                    zero[:, :45] += env.model.plate.PixelPlate
                    image_list.append(zero)

            if done:
                loss_avg = sum(loss_list) / len(loss_list)

                if cfg.get_gif:
                    if e % 500 == 0:
                        agent.save(e, model_dir)
                        create_gif(image_list, image_dir + str(e) + '.gif')

                with open(log_dir + "train_log.csv", 'a') as f:
                    f.write('%d,%1.2f,%1.2f,%1.2f,1.2%f\n' % (e, r_epi, efficiency, batch_rate, loss_avg))

                vessl.log(step=e, payload={"learnig_rate": agent.scheduler.get_last_lr()[0]})
                vessl.log(step=e, payload={'loss': loss_avg})
                vessl.log(step=e, payload={'reward': r_epi})
                vessl.log(step=e, payload={'efficiency': efficiency})
                vessl.log(step=e, payload={'batch_rate': batch_rate})

                break

        agent.scheduler.step()