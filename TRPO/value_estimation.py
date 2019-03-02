import numpy as np
import torch
from network import NN
from util import conjugate_gradient

def compute_discounted_rewards(rewards, gamma):
    '''
    Compute discounted rewards for single episode
    param rewards: {numpy ndarray} sampled rewards
    param gamma: {float} discount factor
    return: {numpy ndarray} discounted rewards
    '''
    R = np.zeros(rewards.shape)
    R[-1] = rewards[-1]
    for i in range(rewards.shape[0] - 2, -1, -1):
        R[i] = gamma * R[i + 1] + rewards[i]

    return R


class GAE(object):
    '''
    Method proposed in high-dimensional continuous control using generalized advantage estimation
    (Schulman 2018)
    '''

    def __init__(self, gamma, lambd, state_dim, inter_dims):
        self.gamma = gamma
        self.lambd = lambd
        self.value = Value(state_dim, 1, inter_dims)


    def compute_td_residuals(self, states, rewards):
        '''
        Compute the temporal difference for multiple episodes
        param states: {list of numpy ndarray} sampled states
        param rewards: {list of numpy ndarray} sampled rewards
        return: {list of numpy ndarray} td residuals
        '''
        def compute_td_residual(state, next_state, reward):
            return reward + self.gamma * self.value(next_state) - self.value(state)

        all_td_residuals = []
        for episode in range(len(rewards)):
            episode_td_residuals = []
            for i in range(rewards[episode].shape[0]):
                episode_td_residuals += [compute_td_residual(states[episode][i], states[episode][i+1], rewards[episode][i])]
            all_td_residuals += [np.array(episode_td_residuals)]
        return all_td_residuals


    def compute_advantages(self, deltas):
        '''
        Compute advantage estimates for one episode
        param deltas: {numpy ndarray} td residuals
        return: {numpy ndarray} advantages
        '''
        A = np.zeros(deltas.shape)
        A[-1] = self.gamma * self.lambd * deltas[-1]
        for i in range(deltas.shape[0] - 2, -1, -1):
            A[i] = self.gamma * self.lambd * (deltas[i] + A[i+1])
        return A


    def update_value(self, states, discounted_rewards, delta):
        '''
        update the value function
        param states: {numpy ndarray} sampled states
        param discounted_rewards: {numpy ndarray} sampled discounted rewards
        param delta: {float} KL-bound for value update
        '''
        target = torch.tensor(discounted_rewards, dtype=torch.float)
        loss = (self.value(states) - target).pow(2).mean()
        loss.backward()
        g = self.value.get_gradients()
        Js = self.value.compute_Jacobians(states)
        s = conjugate_gradient(g, Js, 1, g)
        params = self.value.get_parameters()

        sHs = 0
        for J in Js:
            # todo divide by len(Js)?????
            sHs += (s.T @ (J.T @ (J @ s)))[0, 0]
        sHs = sHs / len(Js)
        alpha = np.power((2 * delta) / sHs, 0.5)
        #todo linesearch
        new_params = (params + alpha * torch.tensor(s.T)).view(-1)
        self.value.update_parameter(new_params)

class Value(NN):
    def compute_Jacobians(self, states):
        '''
        Computes one Jacobi-Matrix per state sample

        :param states: {numpy ndarray} the sampled states
        :return: {list of numpy matrix} a list of Jacobi matrixes
        '''
        states = torch.tensor(states, dtype=torch.float)
        predictions = self.model(states)

        # Compute the coloumns of the Jacobi-Matrix
        number_cols = sum(p.numel() for p in self.model.parameters())

        Jacobi_matrices = []

        # We compute the Jacobi matrix for each state in states
        for i in range(predictions.size(0)):
            Jacobi_matrix = np.matrix(np.zeros((1, number_cols)))

            # reset gradients:
            self.model.zero_grad()

            predictions[i].backward(retain_graph=True)

            Jacobi_matrix[0, :] = self.get_gradients().T
            Jacobi_matrices += [Jacobi_matrix]

        return Jacobi_matrices


    def __call__(self, *args, **kwargs):
        '''
        Compute values for a number of states
        param *args: {numpy ndarray} states
        return: {torch Tensor} values
        '''
        states = torch.tensor(*args, dtype=torch.float)
        return self.model(states, **kwargs)
