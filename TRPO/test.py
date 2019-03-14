import numpy as np
import argparse
import pickle
import gym
import quanser_robots
from quanser_robots import GentlyTerminating

from trpo import line_search
from policy import Policy
from sampling import sample_sp
from value_estimation import compute_discounted_rewards, GAE
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
parser.add_argument("--layers", type=int, default=[64, 64], nargs="+",
                    help="dimensions of layers in policy network and eventually of the value network")
parser.add_argument("--gae", action='store_true',
                    help="shall general advantage estimation be used?")
parser.add_argument("--lambd", type=float, default=0.9,
                    help="Parameter for general advantage estimation")
args = parser.parse_args()
if args.save is not None:
    settings_file = open("settings/%s.txt" %args.save, "w+")
    settings_file.write(str(args.__dict__))
    settings_file.close()

plotter = LearningCurvePlotter(args.iterations, args.save)
env = GentlyTerminating(gym.make(args.env))

# Load policy
if args.load is not None:
    input = open("policies/%s.pkl" %args.load, "rb")
    data = pickle.load(input)
    policy = data.get("policy")
else:
    policy = Policy(env.observation_space.shape[0], env.action_space.shape[0], args.layers)

if args.gae:
    gae = GAE(args.gamma, args.lambd, env.observation_space.shape[0], args.layers)

for i in range(args.iterations):
    print("Iteration ", i, ":")

    states, actions, rewards = sample_sp(env, policy, args.episodes)
    value_sample_estimate = np.concatenate([compute_discounted_rewards(r, args.gamma) for r in rewards])

    if args.gae:
        td_residuals = gae.compute_td_residuals(states, rewards)
        advantages = np.concatenate([gae.compute_advantages(tds) for tds in td_residuals])

        #for _ in range(15):
        gae.update_value(np.concatenate([s[:-1] for s in states]), value_sample_estimate, args.delta)

    else:
        #original TRPO:
        advantages = value_sample_estimate

    policy = line_search(args.delta, np.concatenate([s[:-1] for s in states]), np.concatenate(actions), advantages, policy)

    #print("STD: ", policy.model.log_std.exp())

    # Save in file
    policy.save_model(args.save)
    if args.gae:
        gae.value.save_model(args.save)
    # Plotting
    plotter.update(np.concatenate(rewards).mean(), np.concatenate(rewards).var())
