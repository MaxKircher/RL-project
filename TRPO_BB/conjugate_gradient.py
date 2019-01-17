import numpy as np

class ConjugateGradient(object):
    def __init__(self, k):
        self.k = k

    '''
        x = x0
    '''
    def cg(self, g, Js, M, x):

        def fisher_vector_product(x):
            Ax = np.zeros(g.shape)
            for j in range(len(Js)):
                Jx = np.matrix(Js[j]) @ x
                MJx = M @ Jx
                Ax += np.matrix(Js[j]).T @ MJx # was JtMJx
            return Ax / len(Js)

        Ax = fisher_vector_product(x)
        r = g - Ax
        d = r

        #print("d: ", d.sum())

        r_norm_2 = r.T @ r

        for i in range(self.k):
            z = fisher_vector_product(d)

            #Ad = JtMJd
            #z = Ad

            #print("z: ", z.sum())
            #print("d: ", d.sum())

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
