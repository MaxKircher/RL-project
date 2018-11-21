import numpy as np
import gym
import quanser_robots
import torch
import policy

'''
    mu_a(x) bestimmen
    aus dem Netzwerk holen und dort auch die Gradientenableitungen
    die Gradienten dann in J Matrix speichern

    1. Wir übergeben pytorch für ein bestimmtes x' mu_a(x')
    2. Dann berechnet pytorch mit backward die Gradienten
    3. Wir speichern das ergebnis als J(x')
    4. Benötigen wir J(x*) für ein anderes x*, so starten wir bei 1. mit x* statt x'
'''

# TODO generalisieren 3 = dim_state_space 1 = dim_action_space
policy_getter = policy.NN(3, 1)
policy_net = policy_getter.model

# state ist das x im paper
# x = ....

# our prediction - we will pass the network a matrix, i.e. a batch
mu_x = policy_net(x)

# .backward um die Gradienten zu bestimmen (2.)
mu_x.backward()

# Abspeichern der thetas
thetas = list(policy_net.parameters())

# Macht man eine for-Schleife drum hat man die Jacobi-Matrix als Lsite aufgeschrieben
grad = thetas[0].grad
# ... J = Jacobi-Matrix

# Nach Wiki und Screenshot, siehe drive erhalten wir die Fisher-Information Matrix als
# log-dev brauchen wir anch D
fim = np.eye(policy_getter.log_dev.size())*policy_getter.log_dev.exp().pow(-2)
