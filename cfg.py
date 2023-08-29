import argparse

def get_cfg():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--n_episode", type=int, default=10000, help="number of episodes")
    parser.add_argument("--load_model", type=bool, default=False, help="load the trained model")
    parser.add_argument("--get_gif", type=bool, default=True, help="get gif file")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--lr", type=float, default=0.1e-3 , help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="clipping paramter")
    parser.add_argument("--K_epoch", type=int, default=1, help="optimization epoch")
    parser.add_argument("--T_horizon", type=int, default=100, help="running horizon")

    return parser.parse_args()