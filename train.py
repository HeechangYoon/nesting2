import os
import vessl

from cfg import get_cfg
from torch.utils.tensorboard import SummaryWriter
from agent.ppo import *
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

    lr = cfg.lr
    gamma = cfg.gamma
    lmbda = cfg.lmbda
    eps_clip = cfg.eps_clip
    K_epoch = cfg.K_epoch
    T_horizon = cfg.T_horizon

    model_dir = '/output/train/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = '/output/train/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    simulation_dir = '/output/train/simulation/'
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    image_dir = './output/train/image/'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    env = HiNEST(look_ahead=5)
    agent = Agent(env.state_size, env.x_action_size, env.a_action_size, lr, gamma, lmbda, eps_clip, K_epoch)
    writer = SummaryWriter(log_dir)

    if cfg.load_model:
        checkpoint = torch.load(cfg.model_path)
        start_episode = checkpoint['episode'] + 1
        agent.network.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_episode = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, loss\n')

    for e in range(start_episode, cfg.n_episode + 1):
        # vessl.log(payload={"Train/learnig_rate": agent.scheduler.get_last_lr()[0]}, step=e)
        # writer.add_scalar("Training/Learning Rate", agent.scheduler.get_last_lr()[0], e)

        s = env.reset()
        update_step = 0
        r_epi = 0.0
        efficiency = 0.0
        batch_rate = 0.0
        avg_loss = 0.0
        done = False

        if cfg.get_gif:
            image_list = list()
            temp = np.zeros((210, 45))
            temp += env.model.plate.PixelPlate
            image_list.append(temp)

        while not done:
            possible_actions = env.get_possible_actions()
            a, prob, mask = agent.get_action(s, possible_actions)
            possible_x = env.get_possible_positions(a)
            a_x, prob_x, mask_x = agent.get_position(s, possible_x)

            s_prime, r, efficiency, batch_rate, done, overlap, temp = env.step((a_x, a))

            agent.put_data((s, a_x, a, r, s_prime, prob_x, prob, mask_x, mask, done))
            s = s_prime

            r_epi = r

            update_step += 1

            if cfg.get_gif:
                if overlap:
                    image_list.append(temp)
                else:
                    temp = np.zeros((210,45))
                    temp += env.model.plate.PixelPlate
                    image_list.append(temp)

            if done:
                # image = env.model.plate.PixelPlate
                # save_image(image, image_dir + str(e) + '.png')
                if cfg.get_gif:
                    create_gif(image_list, image_dir + str(e) + '.gif')
                vessl.log(step=e, payload={'reward': r_epi})
                vessl.log(step=e, payload={'efficiency': efficiency})
                vessl.log(step=e, payload={'batch_rate': batch_rate})
                break
            avg_loss += agent.train()

        # agent.scheduler.step()

        # print("episode: %d | reward: %.2f | distance : %d | loss : %.2f" % (e, r_epi, env.model.crane_dis_cum["Crane1"], avg_loss))
        with open(log_dir + "train_log.csv", 'a') as f:
            f.write('%d,%1.2f,1.2%f\n' % (e, r_epi, avg_loss))

        writer.add_scalar("Training/Reward", r_epi, e)
        writer.add_scalar("Training/Loss", avg_loss / update_step, e)
        # for i in env.model.crane_dis_cum.keys():
        #     writer.add_scalar("Performance/Distance", env.model.crane_dis_cum[i], e)

        # if e % 20 == 0:
        #     with torch.no_grad():
        #         delay, move, priority_ratio = [], [], []
        #         for path in validation_path:
        #             df_ship_test = pd.read_excel(path, sheet_name="ship", engine='openpyxl')
        #             df_initial_test = pd.read_excel(path, sheet_name="initial", engine='openpyxl')
        #             test_env = QuayScheduling(df_quay, df_ship_test, df_initial_test, w_delay, w_move, w_priority, rand=False)
        #
        #             s = test_env.reset()
        #             done = False
        #
        #             while not done:
        #                 possible_actions = test_env.get_possible_actions()
        #                 a, prob, mask = agent.get_action(s, possible_actions)
        #                 s_prime, r, done = test_env.step(a)
        #                 s = s_prime
        #
        #                 if done:
        #                     log = test_env.get_logs()
        #                     break
        #
        #             average_delay = calculate_average_delay(log)
        #             move_ratio = calculate_move_ratio(log)
        #             priority_ratio = calculate_priority_ratio(log)
        #
        #             name = path.split("/")[-1][:-5] + "_log.csv"
        #             with open(log_dir + name, 'a') as f:
        #                 f.write('%d,%1.4f, %1.4f, %1.4f\n' % (e, average_delay, move_ratio, priority_ratio))
        #
        #             writer.add_scalars("Validation/Average Delay", {name.split("_")[1]: average_delay}, e)
        #             writer.add_scalars("Validation/Move Ratio", {name.split("_")[1]: move_ratio}, e)
        #             writer.add_scalars("Validation/Priority Ratio", {name.split("_")[1]: priority_ratio}, e)


        if e % 1000 == 0:
            agent.save_network(e, model_dir)
            # env.model.mointor.save(simulation_dir + "episode%d.csv" % e)

    writer.close()