import numpy as np


"""
    Use, that only variances are given -> dimensions are independant
"""
def kl_normal_distribution(mu_new, mu_old, log_std_new, log_std_old):
    var_new = np.power(np.exp(log_std_new), 2)
    var_old = np.power(np.exp(log_std_old), 2)

    kl = log_std_new - log_std_old + (var_old - np.power(mu_old - mu_new, 2)) / (2.0 * var_new) - 0.5
    # average over samples, sum over action dim
    kl = np.abs(kl.mean(0)).sum(0)
    # print("kl: ", kl)
    return kl


def cg(g, Js, M, x, k=10):
    def fisher_vector_product(x):
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
