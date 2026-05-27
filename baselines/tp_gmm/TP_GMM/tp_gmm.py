from . import gmm
from transformations import lintrans, lintrans_cov
from quaternion_metric import process_quaternions
from utils import get_mean_cov_hats
import numpy as np

class TP_GMM():
    def __init__(self, data_in_all_rfs, rfs, nb_states, nb_dim):
        self.data_in_all_rfs = data_in_all_rfs
        self.rfs = rfs
        self.n_rfs = len(rfs)
        self.nb_states = nb_states
        self.nb_dim = nb_dim
        self.gmms = {}
        for rf in self.rfs:
            self.gmms[rf] = gmm.GMM(self.nb_states, self.nb_dim)

    def train(self, reg=1e-8, maxiter=200, verbose=False, kmeans_init = False):
        for i, rf in enumerate(self.rfs):
            data_rf = self.data_in_all_rfs[i]
            if kmeans_init:
                self.gmms[rf].em(data_rf, reg, maxiter, verbose = verbose, kmeans_init = kmeans_init, random_init=False)
            else:
                self.gmms[rf].em(data_rf, reg, maxiter, verbose=verbose)

    def predict(self, t, HTs, objs):
        mus = []
        sigmas = []
        for i, obj in enumerate(objs):
            H = HTs[i]
            mu, sigma = self.gmms[obj].condition(t[:, None], dim_in=slice(0, 1), dim_out=slice(1, self.nb_dim))
            new_mu = lintrans(np.array(mu), H)  # Linear
            n_dims = len(mu[0])
            if n_dims != 3:
                quats = new_mu[:, -4:]
                quats_new = process_quaternions(quats, sigma=None)
                new_mu[:, -4:] = quats_new
            new_sigma = lintrans_cov(sigma, H)
            mus.append(new_mu)
            sigmas.append(new_sigma)
        mu_mean, sigma_mean = get_mean_cov_hats(mus, sigmas)
        if n_dims != 3:
            quats = mu_mean[:, -4:]
            quats_new = process_quaternions(quats, sigma=None)
            mu_mean[:, -4:] = quats_new
        return mu_mean, sigma_mean