import torch
import numpy as np

class dADMM:
    def __init__(self, rho, alpha, beta, gamma, max_iter):
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter