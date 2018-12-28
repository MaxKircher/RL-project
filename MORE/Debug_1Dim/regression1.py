import numpy as np

def __phi__(theta):
    return np.asarray([theta**2, theta, 1])

'''
    Linear regression is of format: rewards = X*beta

    X computes the X matrix for multpile linear regression
     - thetas: List of sampled thetas

    Returns:
     - X = [[__phi__(theta_1).T], ..., [__phi__(theta_n).T]] as npumpy array
'''
def __X__(thetas):
    # init X with first theta from thetas (i.e. the list with all thetas)
    phi_theta_0 = __phi__(thetas[0])

    # make theta to matrix _m
    phi_theta_0_m = np.array([phi_theta_0])
    X = phi_theta_0_m

    for i in range(1, len(thetas)):

        # make theta to matrix _m
        phi_theta_i = __phi__(thetas[i])
        phi_theta_i_m = np.array([phi_theta_i])
        X = np.append(X, phi_theta_i_m, 0)
    #print("X: " ,X.shape)
    return X

'''
    Input:
     - thetas: A list of parameter wich corresponds to rewards
     - rewards: A list of rewards wich corresponds to theta

    Computes regression Matrix X

    Returns:
     - beta_hat, which contains the parameter for our quadratic surrogate model R
'''
def linear_regression(thetas, rewards): # X as usual in a linear regression and rewards is the y value
    X = __X__(thetas)
    beta_hat = np.linalg.lstsq(X, rewards, rcond=None)[0]
    return beta_hat

'''
    Reconstruct R Matrix from beta_hat
    NB: for polynomial_policy Grad 2: R = np.eye(11) (falls np.eye(d) ein Fehler wirft)
'''
def compute_quadratic_surrogate(beta_hat):

    R = beta_hat[0]
    r = beta_hat[1]
    r0 = beta_hat[2]

    return R, r, r0
