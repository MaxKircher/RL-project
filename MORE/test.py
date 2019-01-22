import numpy as np
import gym
import quanser_robots
from policy import *
from MORE_iteration import *
from MORE_iteration_ts import *
from training_states import*

env = gym.make('CartpoleStabShort-v0')
state_dim = env.observation_space.shape[0] # = 5
action_dim = env.action_space.shape[0] # = 1

degree = 2 #we assume that all degrees occur
#policy = PolynomialPolicy(state_dim, action_dim, degree)
#policy = NeuronalNetworkPolicy(state_dim, action_dim)
#policy = DebugPolicy(state_dim, action_dim)
policy = Rosenbrock(state_dim, action_dim)

print("(state_dim, action_dim) =  ", "(", state_dim, ",", action_dim, ")")
print("Number of model parameters: ", policy.get_number_of_parameters())

bound = .000001 * policy.get_number_of_parameters()
#N_per_theta, number_of_thetas, memory_size = 10000,20,150
N_per_theta, number_of_thetas, memory_size = 1, 1000, 1000 # Samples auf 1000 erhöhen?
# Without training states
iterator = More(bound, policy, env, N_per_theta, number_of_thetas, memory_size)
thetas = iterator.iterate()

# With training states
# training_states = TrainingStates(env)
# ts = training_states.generate_training_states(300)
# ts_iterator = MoreTS(bound, policy, env, ts)
# ts_thetas = ts_iterator.iterate()

print("MORE converged to the above soloution.")
