from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

def __phi__(thetas):
    '''
    Feature map for computing the quadratic surrogate model
    :param thetas: {list of numpy ndarray} parameters of the policy
    :return: {numpy ndarray} feature matrix
    '''
    poly = PolynomialFeatures(2)
    return poly.fit_transform(thetas)

def compute_quadratic_surrogate(thetas, rewards, weights):
    '''
    Computes the quadratic surrogate model
    :param thetas: {numpy ndarray} A list of policy parameter
    :param rewards: {numpy ndarray}  A list of rewards
    :param weights: {numpy ndarray} A list of weights
    :return:
    '''
    d = thetas.shape[1]

    # Do linear regression
    features = __phi__(thetas)
    reg = Ridge(fit_intercept=False).fit(features, rewards, weights)
    beta_hat = reg.coef_

    #r0 = beta_hat[0]
    r = beta_hat[1 : d+1]
    R_param = beta_hat[d+1:]

    # Construct R matrix
    R = np.zeros((d,d))
    j = 0
    for i in range(d):
        R[i, i:] += R_param[j:j+d-i]
        R[i:, i] += R_param[j:j+d-i]
        j += d-i

    R = R / 2

    return R, r
