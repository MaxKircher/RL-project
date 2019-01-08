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
        Ax = J.T @ MJx # was JtMJx

        #Ax = JtMJx
        r = g - Ax
        d = r

        #print("d: ", d.sum())

        r_norm_2 = r.T @ r

        for i in range(self.k):
            Jd = J @ d
            MJd = M @ Jd
            z = J.T @ MJd # was JtMJd

            #Ad = JtMJd
            #z = Ad

            #print("z: ", z.sum())
            #print("d: ", d.sum())

            dz = d.T @ z
            assert dz != 0
            alpha = r_norm_2 / dz

            x += alpha[0,0] * d
            r = r - alpha[0,0] * z

            new_norm_2 = r.T @ r
            beta = new_norm_2 / r_norm_2

            r_norm_2 = new_norm_2

            d = r + beta[0,0] * d

            #print("r norm: ", r_norm_2)
            if r_norm_2[0,0] < 1e-10:
                break

        return x



    def cg_torch(self, g, fvp, states, x):

        fvp_x = fvp(states, x)
        #print("fvp_x: ", fvp_x)
        r = g - fvp_x
        d = r

        r_norm_2 = r.pow(2).sum()

        #print("rnorm: ", r_norm_2.size())

        for i in range(self.k):
            z = fvp(states, d).detach()

            #print("z: ", z)
            #print("d: ", d)

            dz = (d * z).sum()
            #print("dz: ", dz.size())
            assert dz != 0
            alpha = r_norm_2 / dz
            #print("alpha: ", alpha.size())


            x += alpha[0,0] * d
            r = r - alpha[0,0] * z

            #print("r: ", r.size())


            new_norm_2 = r.pow(2).sum()
            beta = new_norm_2 / r_norm_2

            r_norm_2 = new_norm_2

            d = r + beta[0,0] * d

            #print("r norm: ", r_norm_2)
            if r_norm_2[0,0] < 1e-10:
                break

        return x
