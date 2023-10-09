import numpy as np 
from math import log
from scipy.optimize import minimize


def grimshaw(peaks:np.array, threshold:float, num_candidates:int=10, epsilon:float=1e-8):
    ''' The Grimshaw's Trick Method

    The trick of thr Grimshaw's procedure is to reduce the two variables 
    optimization problem to a signle variable equation. 

    Args:
        peaks: peak nodes from original dataset. 
        threshold: init threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform

    Returns:
        gamma: estimate
        sigma: estimate
    '''
    min = peaks.min()
    max = peaks.max()
    mean = peaks.mean()

    if abs(-1 / max) < 2 * epsilon:
        epsilon = abs(-1 / max) / num_candidates

    a = -1 / max + epsilon
    b = 2 * (mean - min) / (mean * min)
    c = 2 * (mean - min) / (min ** 2)

    candidate_gamma = solve(function=lambda t: function(peaks, threshold), 
                            dev_function=lambda t: dev_function(peaks, threshold), 
                            bounds=(a + epsilon, -epsilon), 
                            num_candidates=num_candidates
                            )
    candidate_sigma = solve(function=lambda t: function(peaks, threshold), 
                            dev_function=lambda t: dev_function(peaks, threshold), 
                            bounds=(b, c), 
                            num_candidates=num_candidates
                            )
    candidates = np.concatenate([candidate_gamma, candidate_sigma])

    gamma_best = 0
    sigma_best = mean
    log_likelihood_best = cal_log_likelihood(peaks, gamma_best, sigma_best)

    for candidate in candidates:
        gamma = np.log(1 + candidate * peaks).mean()
        sigma = gamma / candidate
        log_likelihood = cal_log_likelihood(peaks, gamma, sigma)
        if log_likelihood > log_likelihood_best:
            gamma_best = gamma
            sigma_best = sigma
            log_likelihood_best = log_likelihood

    return gamma_best, sigma_best


def function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    return u * v - 1


def dev_function(x, threshold):
    s = 1 + threshold * x
    u = 1 + np.log(s).mean()
    v = np.mean(1 / s)
    dev_u = (1 / threshold) * (1 - v)
    dev_v = (1 / threshold) * (-v + np.mean(1 / s ** 2))
    return u * dev_v + v * dev_u


def obj_function(x, function, dev_function):
    m = 0
    n = np.zeros(x.shape)
    for index, item in enumerate(x):
        y = function(item)
        m = m + y ** 2
        n[index] = 2 * y * dev_function(item)
    return m, n


def solve(function, dev_function, bounds, num_candidates):
    step = (bounds[1] - bounds[0]) / (num_candidates + 1)
    x0 = np.arange(bounds[0] + step, bounds[1], step)
    optimization = minimize(lambda x: obj_function(x, function, dev_function), 
                            x0, 
                            method='L-BFGS-B', 
                            jac=True, 
                            bounds=[bounds]*len(x0)
                            )
    x = np.round(optimization.x, decimals=5)
    return np.unique(x)


def cal_log_likelihood(peaks, gamma, sigma):
    if gamma != 0:
        tau = gamma/sigma
        log_likelihood = -peaks.size * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * peaks)).sum()
    else: 
        log_likelihood = peaks.size * (1 + log(peaks.mean()))
    return log_likelihood



def pot(data:np.array, risk:float=1e-4, init_level:float=0.98, num_candidates:int=10, epsilon:float=1e-8) -> float:
    ''' Peak-over-Threshold Alogrithm

    References: 
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        risk: detection level
        init_level: probability associated with the initial threshold
        num_candidates: the maximum number of nodes we choose as candidates
        epsilon: numerical parameter to perform
    
    Returns:
        z: threshold searching by pot
        t: init threshold 
    '''
    # Set init threshold
    t = np.sort(data)[int(init_level * data.size)]
    peaks = data[data > t] - t

    # Grimshaw
    gamma, sigma = grimshaw(peaks=peaks, 
                            threshold=t, 
                            num_candidates=num_candidates, 
                            epsilon=epsilon
                            )

    # Calculate Threshold
    r = data.size * risk / peaks.size
    if gamma != 0:
        z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
    else: 
        z = t - sigma * log(r)

    return z, t
    


def dspot(data:np.array, num_init:int, depth:int, risk:float=1e-4):
    ''' Streaming Peak over Threshold with Drift

    Reference:
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory." 
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge 
    Discovery and Data Mining. 2017.

    Args:
        data: data to process
        num_init: number of data point selected to init threshold
        depth: number of data point selected to detect drift
        risk: detection level

    Returns: 
        logs: 'threshold' threshold with dataset length; 'a' anomaly datapoint index
    '''
    logs = {'threshold': [], 'a': []}

    base_data = data[:depth]
    init_data = data[depth:depth + num_init]
    rest_data = data[depth + num_init:]

    for i in range(num_init):
        temp = init_data[i]
        init_data[i] -= base_data.mean()
        np.delete(base_data, 0)
        np.append(base_data, temp)

    z, t = pot(init_data)
    k = num_init
    peaks = init_data[init_data > t] - t
    logs['threshold'] = [z] * (depth + num_init)

    for index, x in enumerate(rest_data):
        temp = x
        x -= base_data.mean()
        if x > z:
            logs['a'].append(index + num_init + depth)
        elif x > t:
            peaks = np.append(peaks, x - t)
            gamma, sigma = grimshaw(peaks=peaks, threshold=t)
            k = k + 1
            r = k * risk / peaks.size
            z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
            np.delete(base_data, 0)
            np.append(base_data, temp)
        else:
            k = k + 1
            np.delete(base_data, 0)
            np.append(base_data, temp)

        logs['threshold'].append(z)
    
    return logs