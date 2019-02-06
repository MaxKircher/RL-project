import numpy as np


def kl_normal_distribution(mu_new, mu_old, log_std_new, log_std_old):
    '''
    Computes the Kullback-Leiber divergence of two normal distributions
    :param mu_new: {numpy ndarray} expectation value for the new normal distribution
    :param mu_old: {numpy ndarray} expectation value for the old normal distribution
    :param log_std_new: {float} logarithm of standard deviation for the new normal distribution
    :param log_std_old: {float} logarithm of standard deviation for the old normal distribution
    :return: Kullback-Leiber Divergence
    '''
    var_new = np.power(np.exp(log_std_new), 2)
    var_old = np.power(np.exp(log_std_old), 2)

    kl = log_std_new - log_std_old + (var_old - np.power(mu_old - mu_new, 2)) / (2.0 * var_new) - 0.5
    # average over samples, sum over action dim
    kl = np.abs(kl.mean(0)).sum(0)
    # print("kl: ", kl)
    return kl

def conjugate_gradient(g, Js, M, x, k=10):

    '''
    Computes the soloution for Ax = g, where A is symmetric and positive definit
    :param g: {numpy ndarray} gradient
    :param Js: {list numpy matrix} a list of Jacobi Matrices
    :param M: {numpy matrix} Fisher Information Matrix
    :param x: {numpy ndarray} start value
    :param k: {int} number of iterations
    :return: {float} search direction x
    '''
    def fisher_vector_product(x):
        '''
        Computes the Fisher vector product for all Jacobi matrices
        :param x: {nump ndarray} current search direction
        :return: Monte Carlo estimate for Fisher vector product
        '''
        Ax = np.zeros(x.shape)
        for j in range(len(Js)):
            Ax += Js[j].T @ (M @ (Js[j] @ x))
        return Ax / len(Js)

    Ax = fisher_vector_product(x)
    r = g - Ax
    d = r

    r_norm_2 = r.T @ r

    for i in range(k):
        z = fisher_vector_product(d)

        dz = d.T @ z
        assert dz != 0
        alpha = r_norm_2 / dz

        x += alpha[0,0] * d
        r = r - alpha[0,0] * z

        r_kplus1_norm_2 = r.T @ r
        beta = r_kplus1_norm_2 / r_norm_2

        r_norm_2 = r_kplus1_norm_2

        d = r + beta[0,0] * d

        if r_norm_2[0,0] < 1e-10:
            break

    return x
