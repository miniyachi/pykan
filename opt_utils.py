import torch
from pyhessian.utils import group_product


# def normalization_aug(v):
#     """
#     normalization of a list of parameters v[0] and a scalar v[1]
#     return: normalized parameters and scalar v = [v[0], v[1]]
#     """
#     s = group_product(v[0], v[0]) + v[1]**2
#     s = s**0.5
#     s = s.cpu().item()
#     v = [[vi[0] / (s + 1e-6) for vi in v], v[1] / (s + 1e-6)]
#     return v

def group_scalar(v, alpha):
    """
    multiply a list of vectors by a scalar
    """
    return [alpha*vi for vi in v]