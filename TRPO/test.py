import numpy as np
import argparse
import pickle
import gym
import quanser_robots
from quanser_robots import GentlyTerminating

from trpo import line_search
from policy import NN
from sampling import sample_sp
from value_estimation import compute_discounted_rewards
from plotting import LearningCurvePlotter


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, default=800,
                    help="number of TRPO iterations, that shall be performed")
parser.add_argument("-s", "--save", type=str, default=None,
                    help="the filename to save a figure of the learning curve and the learned policy. \
                         If no name is given, no data will be stored.")
parser.add_argument("-l", "--load", type=str, default=None,
                    help="the filename to load a learned policy. \
                         If no name is given, a new policy will be created.")
parser.add_argument("--env", type=str, default="Qube-v0",
                    help="The name of the environment, that shall be used \
                         (Qube_v0, Pendulum-v0, CartpoleStabShort-v0, ...)")
parser.add_argument("-g", "--gamma", type=float, default=0.99,
                    help="discount factor")
parser.add_argument("--delta", type=float, default=0.1,
                    help="KL threshold in linesearch")
parser.add_argument("-e", "--episodes", type=int, default=60,
                    help="number of episodes, that shall be performed per TRPO step")
args = parser.parse_args()
if args.save is not None:
    with open("settings/%s.pkl" %args.save, "wb") as output:
        pickle.dump(args.__dict__, output, pickle.HIGHEST_PROTOCOL)

plotter = LearningCurvePlotter(args.iterations, args.save)
env = GentlyTerminating(gym.make(args.env))

# Load policy
if args.load is not None:
    input = open("policies/%s.pkl" %args.load, "rb")
    data = pickle.load(input)
    policy = data.get("policy")
else:
    policy = NN(env.observation_space.shape[0], env.action_space.shape[0])


for i in range(args.iterations):
    print("Iteration ", i, ":")

    states, actions, rewards = sample_sp(env, policy, args.episodes)
    Q = np.concatenate([compute_discounted_rewards(r, args.gamma) for r in rewards])

    policy = line_search(args.delta, np.concatenate(states), np.concatenate(actions), Q, policy)
    print("STD: ", policy.model.log_std.exp())



    # Save in file
    policy.save_model(args.save)
    # Plotting
    plotter.update(np.concatenate(rewards).mean())
