import numpy as np


def logistic(x, x0=0.0, k=1.0, L=1.0):
    """
    
    Parameters
    ----------
    : 
        TODO
    : 
        TODO
    Returns
    -------
    :
        TODO
    """
    return L / (1.0 + np.exp(-k * x - x0))


def IS(x, l, u, alp):
    """
    
    Parameters
    ----------
    : 
        TODO
    : 
        TODO
    Returns
    -------
    :
        TODO
    """
    return (u - l) + 2.0 / alp * (l - x) * (x < l) + 2.0 / alp * (x - u) * (x > u)


def smooth_IS(x, l, u, alp):
    """
    
    Parameters
    ----------
    : 
        TODO
    : 
        TODO
    Returns
    -------
    :
        TODO
    """
    width = u - l
    return (
        (u - l)
        + 2.0 / alp * (l - x) * logistic(-x, x0=-l, k=2.0 / width)
        + 2.0 / alp * (x - u) * logistic(x, x0=u, k=2.0 / width)
    )


def WIS(x, q, x_q, norm=False, log=False, smooth=False):
    """
    
    Parameters
    ----------
    : 
        TODO
    : 
        TODO
    Returns
    -------
    :
        TODO
    """
    # todo sort q and x_q based on q
    K = len(q) // 2
    alps = np.array([1 - q[-i - 1] + q[i] for i in range(K)])
    Fs = np.array([[x_q[i], x_q[-i - 1]] for i in range(K)])
    m = x_q[K + 1]
    w0 = 0.5
    wk = alps / 2.0
    if smooth:
        ret = (
            1.0
            / (K + 1.0)
            * (
                w0 * 2 * np.abs(x - m)
                + np.sum(wk * smooth_IS(x, Fs[:, 0], Fs[:, 1], alps))
            )
        )
    else:
        ret = (
            1.0
            / (K + 1.0)
            * (w0 * 2 * np.abs(x - m) + np.sum(wk * IS(x, Fs[:, 0], Fs[:, 1], alps)))
        )
    if norm:
        ret /= x
    if log:
        ret = np.log(ret)
    return ret
