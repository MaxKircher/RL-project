import numpy as np
import torch

'''
    Phi is a feature function that
     - (takes a vector) theta:
        Wir nehmen an das jeder Grad vor kommt!
        - parameter vector of dimension d := deg(polynom) * state_dim + 1
        - is in list format, i.e. untransformed unlike the thetas in policy.py for the dot product
     - Returns:
        The feature vector (result_vector) w.r.t. to the linear regression rewards = phi(theta)*beta
        where:
         -> beta = (R_11, R_12, ..., R_1d, R_22, R_23, ..., R_2d, ..., R_dd, r_1, r_2, ..., r_d, r_0)

        1. A bias term r_0 (dimension = 1)
        2. A linear term r (dimension = d)
        3. A quadratic term R (symmetric (state_dim x state_dim)-Matrix) (dimension = d*(d+1)/2)
     - Has dimension  1 + d + d*(d+1) / 2

    Convention on result_vector that contains 1. - 3.
     - Entries:
        (a): 1               to d*(d+1)/2        -> R_11, R_12, ..., R_1d, R_22, R_23, ..., R_2d, ..., R_dd
        (b): d*(d+1)/2 + 1   to d*(d+1)/2 + d    -> r_1, r_2, ..., r_d
        (c): d*(d+1)/2 + d+1 to d*(d+1)/2 + d+1  -> r_0
'''
def __phi__(theta):
    # dimension of theta
    d = theta.shape[0]

    # dimension of phi
    dim_phi =  1 + d + d*(d+1) / 2

    # construct (d x d)-Matrix
    theta_matrix = theta * np.array([theta]).T
    result_vector = np.array([])
    for i in range(d):
        # Take first row beginning from i for the coefficients of (a)
        matrix_row = theta_matrix[i,i:]
        result_vector = np.append(result_vector, matrix_row)

    # Add theta for the coefficients of (b)
    result_vector = np.append(result_vector, theta)

    # Add bias term for the coefficient of (c)
    result_vector = np.append(result_vector, [1])
    return result_vector

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

    # coefficient_of_determination(rewards, thetas, beta_hat)

    return beta_hat

'''
    Reconstruct R Matrix from beta_hat
    NB: for polynomial_policy Grad 2: R = np.eye(11) (falls np.eye(d) ein Fehler wirft)
'''
def compute_quadratic_surrogate(beta_hat, d):

    R_param = beta_hat[:int(d*(d+1)/2) ]
    r = beta_hat[int(d*(d+1)/2) : int(d*(d+1)/2 + d)]
    #r0 = beta_hat[int(d * (d+1)/2 + d)]
    # Construct R matrix
    R_param = np.asarray(R_param)
    R = np.eye(d) # siehe Kommentar für polynomial policy von Grad 2

    j = 0
    for i in range(d):
        R[i, i:] = R_param[j:j+d-i]
        R[i:, i] = R_param[j:j+d-i]
        j += d-i


    # print(R - R.T) # if it's not a Null-Matrix, something went wrong

    return R, r




def coefficient_of_determination(true_rewards, thetas, beta_hat):
    avg_true_rewards = sum(true_rewards)/len(true_rewards)

    true_rewards = np.array(true_rewards)

    X = __X__(thetas)
    estimated_rewards = X @ beta_hat

    sqr = ((true_rewards - estimated_rewards)**2).sum()
    sqt = ((true_rewards - avg_true_rewards)**2).sum()

    assert sqt != 0

    print("coefficient_of_determination: ", 1 - (sqr / sqt))
