import numpy as np
import gym
import quanser_robots
from policy import *
import torch

class TRPO(object):

    def __init__(self, env, gamma, policy):
        self.env = env
        self.gamma = gamma
        self.policy = policy



    def sample_sp(self, policy, s0, T): # sp = single path, T = Anzahl Sample
        s = s0
        states = [s0] # leere Liste mit Actions
        actions = []
        rewards = []
        for i in range(T):
            a = policy.choose_a(s)
            s, r, done, info = self.env.step(a)
            # self.env.render()
            states  += [s]
            actions += [a]
            rewards += [r]
            # if done:
            #     T = i+1
            #     break

        # print("Rewards: ", np.array(rewards).sum())

        states = np.array(states) # Aus Liste Array machen
        actions = np.array(actions)
        Q = np.zeros(T+1)
        for i in range(T-1, -1, -1):
            Q[i] = self.gamma*Q[i+1] + rewards[i]

        return states, actions, Q

###### C.1aus dem Paper
    '''
        mu_a(x) bestimmen
        aus dem Netzwerk holen und dort auch die Gradientenableitungen
        die Gradienten dann in J Matrix speichern

        1. Wir übergeben pytorch für ein bestimmtes x' mu_a(x')
        2. Dann berechnet pytorch mit backward die Gradienten
        3. Wir speichern das ergebnis als J(x')
        4. Benötigen wir J(x*) für ein anderes x*, so starten wir bei 1. mit x* statt x'
    '''

    # state ist das x im paper
    # x = states
    def compute_Jacobian(self, states):
        policy_net = self.policy.model
        # our prediction - we will pass the network a matrix, i.e. a batch
        states = torch.tensor(states, dtype = torch.float)
        mu_states = policy_net(states)

        anzahl_spalten = sum(p.numel() for p in policy_net.parameters())
        print("anzahl_spalten" , anzahl_spalten)
        Jacobi_matrix = np.zeros((mu_states.size(1), anzahl_spalten)) # size mit runden oder eckigen Klammern?


        # .backward um die Gradienten zu bestimmen (2.)
        print(mu_states.size())
        avg_grad = 0
        for i in range(mu_states.size(0)):
            # zero-grad damit die Gradienten zurückgesetzt sind
            policy_net.zero_grad()

            # mehr dim action zweiter Eintrag 0 muss geändert werden
            mu_states[i,0].backward()
            # Abspeichern der thetas = weights
            thetas = list(policy_net.parameters())
            # print("policy_net.parameters() = ", policy_net.parameters())
            # for j in range(4):
            #     print("thetas[0] = ", thetas[j].size())
            # print("len(thetas) = ", len(thetas))
            # Macht man eine for-Schleife drum hat man die Jacobi-Matrix als Lsite aufgeschrieben

            j = 0
            for theta in thetas:
                print("grad.size = " , theta.grad.size())
                grad = theta.grad.view(-1)
                print("grad.size.view(-1) = ", grad.size(0))
                print("grad = ", grad)
                Jacobi_matrix[0,j: + grad.size(0)] += grad
                j += grad.size(0)


            # for j in range(len(thetas)):
            #     grad = thetas[j].grad
            #     # print("grad = ", grad)
            #     # ... J = Jacobi-Matrix TODO generalisieren für mu_states.size() > 1
            #     Jacobi_matrix[0,] += grad.reshape(-1)

        return Jacobi_matrix # evtl. noch averagen

    def compute_FIM(self):
        # Nach Wiki und Screenshot, siehe drive erhalten wir die Fisher-Information Matrix als
        # log-dev brauchen wir anch D
        fim = np.eye(self.policy.log_dev.size())*self.policy.log_dev.exp().pow(-2)
        return fim
