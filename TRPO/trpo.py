import numpy as np
import torch
import copy
from util import kl_normal_distribution, cg


'''
    Params:
     - states: Is a list of states that contains several single states. A Single
               state corresponds to x in the paper in chapter C.1

    Returns:
     - Jacobi_matrix: Jacobi-Matrix die nicht geaveraged wird! Da wir nur an der "Richtung" interssiert sind
'''
def compute_Jacobians(policy, states):
    '''
    Computes one Jacobi-Matrix per state sample

    :param policy: {NN} the used policy
    :param states: {numpy ndarray} the sampled states
    :return: {list of numpy matrix} a list of Jacobi matrixes
    '''
    states = torch.tensor(states, dtype = torch.float)
    action_expectations = policy.model(states)

    # Compute the coloumns of the Jacobi-Matrix
    number_cols = sum(p.numel() for p in policy.model.parameters())

    Jacobi_matrices = []

    # We compute the Jacobi matrix for each state in states
    for i in range(action_expectations.size(0)):
        Jacobi_matrix = np.matrix(np.zeros((policy.a_dim * 2, number_cols)))

        # reset gradients:
        policy.model.zero_grad()

        for k in range(policy.a_dim):
            action_expectations[i,k].backward(retain_graph=True)

            Jacobi_matrix[k,:] = policy.get_gradients()

            Jacobi_matrix[k + policy.a_dim, k] = policy.model.log_std.exp()[k]
            Jacobi_matrices += [Jacobi_matrix]

    return Jacobi_matrices

def compute_FIM_mean(policy):
    '''
    Computes the Fisher-Information Matrix (FIM)
    The Gaussian-Distribution is our distribution of intrest. Therfore,
    by Wiki https://de.wikipedia.org/wiki/Fisher-Information?oldformat=true we obtain
    a simple computable FIM w.r.t mean and variance

    :param policy: policy that gives the variance todo: maybe directly give variance
    :return: {numpy matrix} Fisher Information Matrix
    '''
    inverse_vars = policy.model.log_std.exp().pow(-2).detach().numpy()
    fim = np.matrix(np.diag(np.append(inverse_vars, 0.5 * np.power(inverse_vars, 2))))
    return fim

'''
Parameter:
 - beta: Step size
 - delta: KL constraint
 - s: search direction, i.e. A⁻1 * g
 - theta_old: old model parameter
'''
def line_search(delta, states, actions, Q, old_policy):
    '''
    Perform a linesearch to ensure, that the KL bound is not violated and the objective is improved
    :param delta: {float} bound on KL divergence betweeen original and updated policy
    :param states: {numpy ndarray} sampled states
    :param actions: {numpy ndarray} sampled actions
    :param Q: {numpy ndarray} sampled Q values
    :param old_policy: {NN} the old policy
    :return: {NN} the updated policy
    '''
    subsampled_states = states[0::10] # get every tenth state (see above App D)

    theta_old = old_policy.get_parameters().detach()

    # Compute natural gradient:
    JMs = compute_Jacobians(old_policy, subsampled_states)
    FIM = compute_FIM_mean(old_policy)
    g = compute_objective_gradients(states, actions, Q)
    s = cg(g, JMs, FIM, g)

    beta = compute_beta(delta, np.matrix(s), JMs, FIM)

    old_obj = objective_theta(old_policy.pi_theta, old_policy.pi_theta, states, actions, Q)
    log_std_old = old_policy.model.log_std.detach().numpy()
    mean_old = old_policy.model(torch.tensor(states, dtype = torch.float)).detach().numpy()

    for i in range(1, 20):
        theta_new = theta_old + beta * torch.tensor(s.T, dtype = torch.float)

        # Update the parameters of the model
        new_policy = copy.deepcopy(old_policy)
        new_policy.update_policy_parameter(theta_new)

        mean_new = new_policy.model(torch.tensor(states, dtype = torch.float)).detach().numpy()
        log_std_new = new_policy.model.log_std.detach().numpy()

        kl_change = kl_normal_distribution(mean_new, mean_old, log_std_new, log_std_old)

        if kl_change <= delta:
            obj = objective_theta(old_policy.pi_theta, new_policy.pi_theta, states, actions, Q)
            if obj >= old_obj:
                improvement = obj-old_obj
                print("beta = ", beta, "iteration = ", i)
                print("new objective: ", obj.detach().numpy(), " improved by ", improvement.detach().numpy())
                return new_policy

        # decrease beta:
        beta = beta * np.exp(-0.5 * i)

    print("Something went wrong!")
    return None



def objective_theta(pi_old, pi_new, states, actions, Q):
    '''
    Compute the objective, that shall be optimized.
    See formula 14
    :param pi_old: {NN} the old policy
    :param pi_new: {NN} the new policy
    :param states: {numpy ndarray} the sampled states
    :param actions: {numpy ndarray} the sampled actions
    :param Q: {numpy ndarray} the sampled Q values
    :return: {torch tensor} the objective value
    '''

    q = pi_old(states, actions).detach()
    fast_sum = (pi_new(states, actions) * torch.tensor(Q, dtype=torch.double) / q).sum()

    return fast_sum / actions.shape[0]

def compute_objective_gradients(policy, states, actions, Q):
    '''
    compute the policy gradient for the objective
    :param policy: {NN} the used policy
    :param states: {numpy ndarray} sampled states
    :param actions: {numpy ndarray} sampled actions
    :param Q: {numpy ndarray} sampled Q values
    :return: {torch tensor} the gradient
    '''
    policy.model.zero_grad()
    to_opt = objective_theta(policy.pi_theta, policy.pi_theta, states, actions, Q)
    to_opt.backward()

    g = policy.get_gradients()
    return g

def compute_beta(delta, s, JMs, FIM):
    '''
    Compute the analytical solution for the learning rate / step size
    :param delta: {float} KL bound
    :param s: {numpy matrix} natural gradient
    :param JMs: {list of numpy matrix} the Jacobi matrices
    :param FIM: {numpy matrix} the Fisher information matrix
    :return: the analytical solution for the stepsize
    '''
    sAs = 0
    for JM in JMs:
        sAs += (s.T @ (JM.T @ (FIM @ (JM @ s))))[0,0]
    return np.power((2 * delta) / sAs, 0.5)
