import numpy as np

class ConjugateGradient(object):
    def __init__(self, k):
        self.k = k


    '''
        x = x0
    '''
    def cg(self, g, J, M, x):
        Jx = J @ x
        MJx = M @ Jx
        JtMJx = J.T @ MJx

        Ax = JtMJx
        r = g - Ax
        d = r

        r_norm_2 = r.T @ r

        for i in range(self.k):
            Jd = J @ d
            MJd = M @ Jd
            JtMJd = J.T @ MJd

            Ad = JtMJd
            z = Ad

            dz = d.T @ z
            alpha = r_norm_2 / dz

            x += alpha[0,0] * d
            r = r - alpha[0,0] * z

            r_kplus1_norm_2 = r.T @ r
            beta = r_kplus1_norm_2 / r_norm_2

            r_norm_2 = r_kplus1_norm_2

            d = r + beta[0,0] * d

        return x
