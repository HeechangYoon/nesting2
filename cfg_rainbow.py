import argparse

def get_cfg():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--n_episode", type=int, default=10000, help="number of episodes")
    parser.add_argument("--load_model", type=bool, default=False, help="load the trained model")
    parser.add_argument("--get_gif", type=bool, default=True, help="get gif file")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--n_step", type=int, default=3, help="Multistep IQN")
    parser.add_argument("--capacity", type=int, default=1000, help="Replay memory size")
    parser.add_argument("--alpha", type=float, default=0.6, help="Control paramter for priorizted sampling")
    parser.add_argument("--beta_start", type=float, default=0.4, help="Correction parameter for importance sampling")
    parser.add_argument("--beta_steps", type=int, default=1500000, help="Total number of steps for annealing")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updating the DQN")
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument("--lr_step", type=int, default=2500, help="Step size to reduce learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="Learning rate decay ratio")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.001, help="Soft update parameter tau")

    return parser.parse_args()