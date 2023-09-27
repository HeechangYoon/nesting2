import argparse

def get_cfg():

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--n_episode", type=int, default=10000, help="number of episodes")
    parser.add_argument("--load_model", type=bool, default=False, help="load the trained model")
    parser.add_argument("--get_gif", type=bool, default=True, help="get gif file")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--n_agents", type=int, default=1, help="the number of the independent actors")
    parser.add_argument("--lr", type=float, default=0.1e-3, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--t_max", type=int, default=100, help="running horizon")

    parser.add_argument("--eval_every", type=int, default=200, help="Evaluate every x episodes")
    parser.add_argument("--save_every", type=int, default=1000, help="Save a model every x episodes")

    return parser.parse_args()