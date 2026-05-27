from . import utils
from . import promp_gmm as promp
from . import promp_gaussian as promp_gaussian
import numpy as np
import random
from transformations import lintrans, lintrans_cov, lintrans_position, lintrans_quat, lintrans_cov_position, lintrans_cov_quat
from quaternion_metric import  process_quaternions
from utils import get_mean_cov_hats

class PMP():
    def __init__(self, data_in_all_rfs, times, dof, rfs, dim_basis_fun = 25, sigma = 0.035, full_basis = None, n_components = 1, covariance_type = 'block_diag', max_iter = 100, n_init = 1, gmm = False):
        self.sigma = sigma
        self.dof = dof
        if full_basis == None:
            self.full_basis = {
            'conf': [
                {"type": "sqexp", "nparams": 22, "conf": {"dim": 21}},
                {"type": "poly", "nparams": 0, "conf": {"order": 3}}
            ],
            'params': [np.log(self.sigma), 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65
                , 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        }
        else:
            self.full_basis = full_basis
        self.dim_basis_fun = dim_basis_fun
        self.data_in_all_rfs = data_in_all_rfs
        self.rfs = rfs
        self.n_rfs = len(rfs)
        self.times = times
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.inv_whis_mean = lambda Sigma: utils.make_block_diag(Sigma, self.dof * self.n_rfs)
        self.prior_Sigma_w = {'v': self.dim_basis_fun * self.dof, 'mean_cov_mle': self.inv_whis_mean}
        self.prior_mu_w = {'k0':2, 'm0':0}
        self.prior_mu_w = None
        self.max_iter = max_iter
        self.n_init = n_init
        self.n_demos = len(self.times)
        self.gmm = gmm
        if self.gmm:
            self.pmp = promp.FullProMP(basis=self.full_basis, n_dims= self.n_rfs * self.dof, n_rfs = self.n_rfs, n_components=self.n_components,
                                        covariance_type=self.covariance_type)
        else:
            self.pmp = promp_gaussian.FullProMP(basis=self.full_basis, n_dims=self.n_rfs * self.dof, n_rfs=self.n_rfs)

    def train(self, print_lowerbound = True, no_Sw = False):

        train_summary = self.pmp.train(self.times, data=self.data_in_all_rfs, print_lowerbound=print_lowerbound, no_Sw=no_Sw,
                                        max_iter=self.max_iter, prior_Sigma_w=self.prior_Sigma_w, prior_mu_w = self.prior_mu_w,
                                        n_init=self.n_init)
    def refine(self, max_iter = None):
        if max_iter is None:
            max_iter = self.max_iter
        self.pmp.refine(max_iter)

    def bic(self):
        return self.pmp.bic()


    def select_mode(self):
        modes = np.arange(self.n_components)
        selected_mode_ind = random.choices(modes, weights = self.promp.alpha, k = 1)[0]
        return selected_mode_ind

    def marginal_w(self, t):
        selected_mode_ind = self.select_mode()
        model = self.pmp
        mu,sigma = model.marginal_w(t, selected_mode_ind)

        return mu, sigma

    def condition(self, t, T, q, Sigma_q=None, ignore_Sy = True):
        self.pmp.condition(t, T, q, Sigma_q, ignore_Sy)

    def predict(self, t, HTs, objs, mode_selected = 0):
        mus = []
        sigmas = []
        self.mus = {}
        self.sigmas = {}
        if self.gmm:
            mu_all_rfs, sigma_all_rfs = self.pmp.marginal_w(t, mode_selected)
        else:
            mu_all_rfs, sigma_all_rfs = self.pmp.marginal_w(t)
        for i, obj in enumerate(objs):
            H = HTs[i]
            ind = self.rfs.index(obj)
            mu = mu_all_rfs[:, ind * self.dof: (ind + 1) * self.dof]
            sigma = sigma_all_rfs[:, ind * self.dof: (ind + 1) * self.dof, ind * self.dof: (ind + 1) * self.dof]
            new_mu = lintrans(np.array(mu), H)  # Linear

            n_dims = len(mu[0])
            if n_dims != 3:
                quats = new_mu[:, -4:]
                quats_new = process_quaternions(quats, sigma=None)
                new_mu[:, -4:] = quats_new
            new_sigma = lintrans_cov(sigma, H)
            self.mus[obj] = new_mu
            self.sigmas[obj] = new_sigma
            mus.append(new_mu)
            sigmas.append(new_sigma)
        mu_mean, sigma_mean = get_mean_cov_hats(mus, sigmas)
        return mu_mean, sigma_mean


    def predict_position(self, t, As, bs, objs, mode_selected=0):
        mus = []
        sigmas = []
        self.mus = {}
        self.sigmas = {}
        if self.gmm:
            mu_all_rfs, sigma_all_rfs = self.pmp.marginal_w(t, mode_selected)
        else:
            mu_all_rfs, sigma_all_rfs = self.pmp.marginal_w(t)
        for i, obj in enumerate(objs):
            A = As[i]
            b = bs[i]
            position_dim = A.shape[0]
            ind = self.rfs.index(obj)
            mu_position = mu_all_rfs[:, ind * self.dof: ind * self.dof + position_dim]
            sigma_position = sigma_all_rfs[:, ind * self.dof: ind * self.dof + position_dim, ind * self.dof: ind * self.dof + position_dim ]
            self.mus[obj] = mu_position
            self.sigmas[obj] = sigma_position
            new_mu_position = lintrans_position(mu_position, A, b)
            new_sigma_position = lintrans_cov_position(sigma_position, A)
            mus.append(new_mu_position)
            sigmas.append(new_sigma_position)
        mu_mean, sigma_mean = get_mean_cov_hats(mus, sigmas)
        return mu_mean, sigma_mean

    def predict_quat(self, t, As, objs, mode_selected=0):
        mus = []
        sigmas = []
        if self.gmm:
            mu_all_rfs, sigma_all_rfs = self.pmp.marginal_w(t, mode_selected)
        else:
            mu_all_rfs, sigma_all_rfs = self.pmp.marginal_w(t)
        for i, obj in enumerate(objs):
            A = As[i]
            position_dim = A.shape[0]
            ind = self.rfs.index(obj)
            mu_quat = mu_all_rfs[:, ind * self.dof + position_dim: (ind + 1) * self.dof]
            sigma_quat = sigma_all_rfs[:, ind * self.dof + position_dim: (ind + 1) * self.dof,
                         ind * self.dof + position_dim: (ind + 1) * self.dof]
            new_mu_quat = lintrans_quat(mu_quat, A)
            new_sigma_quat = lintrans_cov_quat(sigma_quat, A)
            mus.append(new_mu_quat)
            sigmas.append(new_sigma_quat)
        mu_mean, sigma_mean = get_mean_cov_hats(mus, sigmas)
        return mu_mean, sigma_mean


