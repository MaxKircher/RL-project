from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

def __phi__(thetas):
    poly = PolynomialFeatures(2)
    return poly.fit_transform(thetas)

def linear_regression(thetas, rewards):
    features = __phi__(thetas)
    reg = LinearRegression(fit_intercept=False).fit(features, rewards)
    return reg.coef_


def compute_quadratic_surrogate(thetas, rewards, d):
    beta_hat = linear_regression(thetas, rewards)
    # print("beta_hat: ", beta_hat)
    r = beta_hat[1 : d+1]
    #r0 = beta_hat[0]

    R_param = beta_hat[d+1:]
    # Construct R matrix

    R = np.zeros((d,d)) # siehe Kommentar für polynomial policy von Grad 2

    j = 0
    for i in range(d):
        R[i, i:] += R_param[j:j+d-i]
        R[i:, i] += R_param[j:j+d-i]
        j += d-i

    R = R / 2

    return R, r


# Only for testing:
def compare(thetas, rewards, d):
    features = __phi__(thetas)
    #print("features: ", features)
    reg = LinearRegression().fit(features, rewards)

    thetas = np.array(thetas)

    beta_hat = linear_regression(thetas, rewards)
    # print("beta_hat: ", beta_hat)
    r = beta_hat[1 : d+1]
    r0 = beta_hat[0]

    R_param = beta_hat[d+1:]
    # Construct R matrix

    R = np.eye(d) # siehe Kommentar für polynomial policy von Grad 2

    j = 0
    for i in range(d):
        R[i, i:] = R_param[j:j+d-i]
        #R[i:, i] = R_param[j:j+d-i]
        j += d-i

    #print("R: ", R.shape, " thetas: ", thetas.shape, " r: ", r.shape)
    our_pred = thetas @ R @ thetas.T + thetas @ r + r0
    #print("R: ", (thetas @ R @ thetas.T).shape, " thetas: ", (thetas @ r).shape)

    correct_pred = reg.predict(features)

    # print("ours: ", np.diag(our_pred))
    # print("correct: ", correct_pred)
    # print("diff: ", np.diag(our_pred) - correct_pred)
