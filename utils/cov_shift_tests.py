import torch
from torch.distributions.normal import Normal
import numpy as np
from scipy import stats

from models.gps import ExactGP, SVGP, train_model, setup_model
from copy import deepcopy
from scipy.linalg import sqrtm
from utils.utils import single_x_f, single_x_f_OOD, plot_ID_OOD, two_x_f


def print_dist(x_test, prior_mean, prior_covar, updated_mean, updated_covar):
    """
    Print the distances between the prior and updated distributions
    :param x_test:
    :param y_test:
    :param prior_mean:
    :param prior_covar:
    :param updated_mean:
    :param updated_covar:
    :return:
    """
    print("Prior mean:", prior_mean.item().__round__(3), "Predictive mean:", updated_mean.item().__round__(3),
          "True value:", two_x_f(x_test, noise=False).item().__round__(3))
    print("Prior variance:", prior_covar.item().__round__(3), "Predictive variance:", updated_covar.item().__round__(3))
    print("KL divergence:", gauss_kl_div(prior_mean, updated_mean, prior_covar, updated_covar).item().__round__(3))
    print("JS divergence:", gauss_js_div(prior_mean, updated_mean, prior_covar, updated_covar).item().__round__(3))
    print("Wasserstein distance:",
          gauss_wass_dist(prior_mean, updated_mean, prior_covar, updated_covar).item().__round__(3))
    print("KS statistic:", ks_statistic(prior_mean, updated_mean, prior_covar, updated_covar).item().__round__(3))


def train_test_split(x, y, val_size=0.2, random_state=42):
    """
    Split the data into train and test sets
    :param x:
    :param y:
    :param test_size:
    :param random_state:
    :return:
    """
    n = x.size(0)
    n_test = int(n * val_size)
    n_train = n - n_test

    # Set the random seed and shuffle the data
    np.random.seed(random_state)
    idx = np.random.permutation(n)

    # Split the data
    x_train, x_val = x[idx[:n_train]], x[idx[n_train:]]
    y_train, y_val = y[idx[:n_train]], y[idx[n_train:]]

    # Reshape if X has only one feature
    if x_train.dim() == 1:
        x_val = x_val.unsqueeze(-1)

    return x_train, y_train, x_val, y_val


def mahalanobis_distance(x, mu, sigma):
    """
    Compute the Mahalanobis distance between a point x and a univariate Gaussian distribution.
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    # Get all the dimension to be the same
    x = x.squeeze()
    mu = mu.squeeze()
    sigma = sigma.squeeze()

    # Compute the Mahalanobis distance
    diff = x - mu
    sigma_inv = 1.0 / sigma
    dm = torch.sqrt(diff ** 2 * sigma_inv)

    return dm


def ks_statistic(prior_mean, post_mean, prior_cov, post_cov):
    """
    Given two distributions as stats.
    :param prior_mean:
    :param post_mean:
    :param prior_cov:
    :param post_cov:
    """

    # Squeeze the dimensions
    prior_mean, post_mean = prior_mean.squeeze(), post_mean.squeeze()
    prior_cov, post_cov = prior_cov.squeeze(), post_cov.squeeze()

    # Set the distributional variables
    prior_dist = Normal(prior_mean, torch.sqrt(prior_cov))
    post_dist = Normal(post_mean, torch.sqrt(post_cov))

    # Compute CDFs for a range of values
    x = torch.linspace(-10, 10, 1000)
    prior_cdf = prior_dist.cdf(x)
    post_cdf = post_dist.cdf(x)

    # Compute the KS distance
    ks_stat = torch.max(torch.abs(prior_cdf - post_cdf))

    return ks_stat


def gauss_kl_div(mu1, mu2, std1, std2):
    """
    Compute the KL divergence between two univariate Gaussian distributions.
    :param mu1:
    :param std1:
    :param mu2:
    :param std2:
    :return:
    """
    # Squeeze the dimensions
    mu1, mu2 = mu1.squeeze(), mu2.squeeze()
    std1, std2 = std1.squeeze(), std2.squeeze()

    p = Normal(mu1, std1)
    q = Normal(mu2, std2)
    kl_div = torch.distributions.kl.kl_divergence(q, p)
    return kl_div


def gauss_js_div(mu1, mu2, std1, std2):
    """
    Compute the JS divergence between two univariate Gaussian distributions.
    :param mu1:
    :param std1:
    :param mu2:
    :param std2:
    :return:
    """
    # Squeeze the dimensions
    mu1, mu2 = mu1.squeeze(), mu2.squeeze()
    std1, std2 = std1.squeeze(), std2.squeeze()

    p = Normal(mu1, std1)
    q = Normal(mu2, std2)
    m = Normal(0.5 * (mu1 + mu2), 0.5 * (std1 + std2))
    js_div = 0.5 * (torch.distributions.kl.kl_divergence(q, m) + torch.distributions.kl.kl_divergence(p, m))
    return js_div


def gauss_wass_dist(mu1, mu2, std1, std2):
    """
    Compute the Wasserstein distance between two Gaussian distributions.

    Parameters:
    mu1, mu2: Mean vectors of the two distributions (PyTorch tensors)
    sigma1, sigma2: Covariance matrices of the two distributions (PyTorch tensors)
    """
    # Squeeze the dimensions
    mu1, mu2 = mu1.squeeze(), mu2.squeeze()
    std1, std2 = std1.squeeze(), std2.squeeze()

    w2 = (mu2 - mu1) ** 2 + (std2 - std1) ** 2
    return torch.sqrt(w2)


def multi_gauss_wass_dist(mu1, mu2, sigma1, sigma2):
    """
    Compute the Wasserstein distance between two multivariate normal distributions.

    Parameters:
    mu1, mu2: Mean vectors of the two distributions (PyTorch tensors)
    sigma1, sigma2: Covariance matrices of the two distributions (PyTorch tensors)
    """
    # Compute the squared Euclidean distance between the means
    diff = mu1 - mu2
    mean_term = torch.dot(diff, diff)

    # Compute the term involving the covariances
    sigma1_sqrt = torch.tensor(sqrtm(sigma1.detach().numpy()))
    sigma = torch.mm(torch.mm(sigma1_sqrt, sigma2), sigma1_sqrt)
    sigma_sqrt = torch.tensor(sqrtm(sigma.detach().numpy()))
    trace_term = torch.trace(sigma1 + sigma2 - 2*sigma_sqrt)

    # Return the Wasserstein distance
    return mean_term + trace_term


def multi_gauss_kl_div(mu1, mu2, sigma1, sigma2):
    """
    Compute the KL divergence between two multivariate normal distributions.

    Parameters:
    mu1, mu2: Mean vectors of the two distributions (PyTorch tensors)
    sigma1, sigma2: Covariance matrices of the two distributions (PyTorch tensors)
    """
    # Compute the trace of the inverse of sigma2 times sigma1
    sigma2_inv = torch.inverse(sigma2)
    trace_term = torch.trace(torch.mm(sigma2_inv, sigma1))

    # Compute the term involving the means
    diff = mu2 - mu1
    mean_term = torch.dot(diff, torch.mv(sigma2_inv, diff))

    # Compute the log determinant ratio
    log_det_ratio = torch.logdet(sigma2) - torch.logdet(sigma1)

    # Return the KL divergence
    return 0.5 * (trace_term + mean_term + log_det_ratio - mu1.shape[0])


def multi_gauss_js_div(mu1, mu2, sigma1, sigma2):
    """
    Compute the JS divergence between two multivariate normal distributions.

    Parameters:
    mu1, mu2: Mean vectors of the two distributions (PyTorch tensors)
    sigma1, sigma2: Covariance matrices of the two distributions (PyTorch tensors)
    """
    # Compute the mean and covariance of the mixture distribution
    mu_mixture = 0.5 * (mu1 + mu2)
    sigma_mixture = 0.5 * (sigma1 + sigma2)

    # Compute the KL divergences from the mixture distribution to the original distributions
    kl1 = multi_gauss_kl_div(mu1, mu_mixture, sigma1, sigma_mixture)
    kl2 = multi_gauss_kl_div(mu2, mu_mixture, sigma2, sigma_mixture)

    # Return the JS divergence
    return 0.5 * (kl1 + kl2)



def pred_error(pred, y):
    """
    Compute the l2 prediction error.
    :param pred:
    :param y:
    :return:
    """
    return (pred - y)**2


class OOD_detector:
    """
    Wrapper class for the OOD detection methods
    """
    def __init__(self, input_dim, model='ExactGP', lr=0.1, num_ind_pts=100):
        """
        Initialize the OOD detector
        :param input_dim:
        :param model:
        :param lr:
        :param num_ind_pts:
        """
        self.gp_dict = None
        self.X_train = None
        self.y_train = None

        self.input_dim = input_dim
        self.model = model
        self.lr = lr
        self.num_ind_pts = num_ind_pts

        # self.gp_model = gp_dict['model']
        # self.likel = gp_dict['likelihood']
        # self.optim = gp_dict['optimizer']
        alpha_dict = {0.1: [], 0.05: [], 0.01: []}
        self.th_dict = {'eucl_dist': deepcopy(alpha_dict), 'zscore': deepcopy(alpha_dict),
                        'js_div': deepcopy(alpha_dict), 'wass_dist': deepcopy(alpha_dict)}

    def predict_posterior_xi(self, x_test):
        """
        Closed-form update of the posterior distribution over the training data.
        :args:
        - x (torch.Tensor): The training inputs.
        - y (torch.Tensor): The training outputs.
        - x_test (torch.Tensor): The test inputs.
        - likel (gpytorch.likelihoods): The likelihood function, to get the noise
        """
        pred = self.gp_dict['model'](x_test)

        prior_mean = pred.mean
        prior_cov = torch.tensor(self.gp_dict['model'].covar_module(x_test, x_test).numpy()).diag()

        pred_mean = pred.mean
        pred_cov = pred.variance

        return prior_mean, pred_mean, prior_cov, pred_cov

    def entropy(self, x):
        """
        Given a point x, compute the entropy of the GP posterior likelihood.
        :param x:
        :return:
        """
        pred = self.gp_dict['model'](x)
        likel = self.gp_dict['likelihood']
        entropy = likel(pred).entropy()

        return entropy

    def maha_dist(self, x, mean, cov):
        """
        Compute the Mahalanobis distance between the prior and the updated posterior, given new point x_test
        :param prior_mean:
        :param post_mean:
        :param prior_cov:
        :param post_cov:
        :return:
        """
        # Compute the entropy of the prior
        maha_dist_xi = mahalanobis_distance(x, mean, cov)

        return maha_dist_xi

    def kl_div(self, prior_mean, post_mean, prior_cov, post_cov):
        """
        Compute the information gain between the prior and the updated posterior, given new point x_test
        :param prior_mean:
        :param post_mean:
        :param prior_cov:
        :param post_cov:
        :return:
        """
        # Compute the entropy of the prior
        info_gain_xi = gauss_kl_div(prior_mean, post_mean, prior_cov, post_cov)

        return info_gain_xi

    def js_div(self, prior_mean, post_mean, prior_cov, post_cov):
        """
        Compute the information gain between the prior and the updated posterior, given new point x_test
        :param prior_mean:
        :param post_mean:
        :param prior_cov:
        :param post_cov:
        :return:
        """
        # Compute the entropy of the prior
        info_gain_xi = gauss_js_div(prior_mean, post_mean, prior_cov, post_cov)

        return info_gain_xi

    def compute_ks_statistic(self, prior_mean, post_mean, prior_cov, post_cov):
        """
        Compute the KS test statistics given train (x,y) and test x_test
        :param prior_mean:
        :param post_mean:
        :param prior_cov:
        :param post_cov:
        :return:
        """
        ks_stat = ks_statistic(prior_mean, post_mean, prior_cov, post_cov)

        return ks_stat

    def compute_wass_dist(self, prior_mean, post_mean, prior_cov, post_cov):
        """
        Compute the Wasserstein distance given train (x,y) and test x_test
        :param prior_mean:
        :param post_mean:
        :param prior_cov:
        :param post_cov:
        :return:
        """
        wass_dist = gauss_wass_dist(prior_mean, post_mean, prior_cov, post_cov)

        return wass_dist

    def train_OOD(self, X, y, K=5, val_size=0.2, training_iterations=100, rng=42, verbose=False):
        """
        Train the OOD model.
        Given (X, y), for K-folds split (X, y) into (X_train, y_train) and (X_val, y_val). Train the model on
        (X_train, y_train) and compute the thresholds on (X_val, y_val). Update the thresholds in the th_dict.
        :param X:
        :param y:
        :param K:
        :param val_size:
        :param training_iterations:
        :param rng:
        :param verbose:
        :return:
        """

        # Split the data into K-folds
        for k in range(K):

            print(f"Training fold {k+1}/{K}...")

            # Split the data into train and validation
            seed_k = rng * k
            X_train, y_train, X_val, y_val = train_test_split(X, y, val_size=val_size, random_state=seed_k)

            self.X_train = X_train
            self.y_train = y_train

            # Initialize model
            if self.model == 'SVGP':
                gp_mod = SVGP(self.input_dim, self.num_ind_pts)
                likel, obj_fun, optim = setup_model(gp_mod, y_train, learning_rate=0.1)
                self.gp_dict = {'model': gp_mod, 'likelihood': likel, 'objective_function': obj_fun, 'optimizer': optim}
            elif self.model == 'ExactGP':
                gp_mod = ExactGP(X_train, y_train)
                likel, obj_fun, optim = setup_model(gp_mod, y_train, learning_rate=0.1)
                self.gp_dict = {'model': gp_mod, 'likelihood': likel, 'objective_function': obj_fun, 'optimizer': optim}
            else:
                raise ValueError("Model not recognized.")

            # Train the model on train set
            train_model(self.gp_dict, X_train, y_train, training_iterations=training_iterations, verbose=verbose)

            # Use the validation set to compute the statistics
            self.gp_dict['model'].eval()
            self.gp_dict['likelihood'].eval()

            # Compute posterior
            prior_mean, post_mean, prior_cov, post_cov = self.predict_posterior_xi(X_val)
            eucl_dist, z_score, js_div, wass_dist = [], [], [], []
            for i in range(X_val.shape[0]):

                prior_mean_i, post_mean_i = prior_mean[i], post_mean[i]
                prior_cov_i, post_cov_i = prior_cov[i], post_cov[i]

                eucl_dist_i = torch.norm(X_val[i] - self.X_train.mean(dim=0))
                z_score_i = torch.abs((X_val[i] - self.X_train.mean(dim=0)) / torch.sqrt(self.X_train.var(dim=0)))
                js_div_i = self.js_div(prior_mean_i, post_mean_i, prior_cov_i, post_cov_i)
                wass_dist_i = self.compute_wass_dist(prior_mean_i, post_mean_i, prior_cov_i, post_cov_i)

                eucl_dist.append(eucl_dist_i)
                z_score.append(z_score_i)
                js_div.append(js_div_i)
                wass_dist.append(wass_dist_i)

            eucl_dist, z_score = torch.stack(eucl_dist), torch.stack(z_score)
            js_div, wass_dist = torch.stack(js_div), torch.stack(wass_dist)

            # Compute the thresholds as the \alpha quantile of the statistics and populate the th_dict
            for alpha in [0.1, 0.05, 0.01]:
                self.th_dict['eucl_dist'][alpha].append(eucl_dist.quantile(1 - alpha).item())
                self.th_dict['zscore'][alpha].append(z_score.quantile(1 - alpha).item())
                self.th_dict['js_div'][alpha].append(js_div.quantile(alpha).item())
                self.th_dict['wass_dist'][alpha].append(wass_dist.quantile(alpha).item())

        # Save only the 5th percentile across folds for each alpha
        for alpha in [0.1, 0.05, 0.01]:
            self.th_dict['eucl_dist'][alpha] = np.quantile(self.th_dict['eucl_dist'][alpha], 0.95)
            self.th_dict['zscore'][alpha] = np.quantile(self.th_dict['zscore'][alpha], 0.95)
            self.th_dict['js_div'][alpha] = np.quantile(self.th_dict['js_div'][alpha], 0.05)
            self.th_dict['wass_dist'][alpha] = np.quantile(self.th_dict['wass_dist'][alpha], 0.05)

    def predict_OOD(self, X, alpha=0.05):
        """
        Predict if the test set (X, y) is OOD.
        :param X:
        :param alpha:
        :return:
        """

        prior_mean, post_mean, prior_cov, post_cov = self.predict_posterior_xi(X)
        eucl_dist, z_score, js_div, wass_dist = [], [], [], []

        for i in range(X.shape[0]):

            X_i = X[i]

            # Slice mean and cov
            prior_mean_i, post_mean_i = prior_mean[i], post_mean[i]
            prior_cov_i, post_cov_i = prior_cov[i], post_cov[i]

            eucl_dist_i = torch.norm(X_i - self.X_train.mean(dim=0))
            z_score_i = torch.abs((X_i - self.X_train.mean(dim=0)) / torch.sqrt(self.X_train.var(dim=0)))
            js_div_i = self.js_div(prior_mean_i, post_mean_i, prior_cov_i, post_cov_i)
            wass_dist_i = self.compute_wass_dist(prior_mean_i, post_mean_i, prior_cov_i, post_cov_i)

            eucl_dist.append(eucl_dist_i)
            z_score.append(z_score_i)
            js_div.append(js_div_i)
            wass_dist.append(wass_dist_i)

        eucl_dist, z_score = torch.stack(eucl_dist), torch.stack(z_score)
        js_div, wass_dist = torch.stack(js_div), torch.stack(wass_dist)

        # Compute the thresholds as the 1-\alpha quantile of the statistics and populate the th_dict
        th_eucl_dist = self.th_dict['eucl_dist'][alpha]
        th_zscore = self.th_dict['zscore'][alpha]
        th_js_div = self.th_dict['js_div'][alpha]
        th_wass_dist = self.th_dict['wass_dist'][alpha]

        # Predict OOD (it is lower or equal because ID training points have the largest update and thus
        # statistics - OOD have lower statistics)
        pred_OOD_eucl_dist = eucl_dist > th_eucl_dist
        pred_OOD_zscore = z_score > th_zscore
        pred_OOD_js = js_div < th_js_div
        pred_OOD_wass_dist = wass_dist < th_wass_dist

        # For eucl and zscore, if any of the P(x_i) > th, then OOD
        pred_OOD_zscore = pred_OOD_zscore.any(dim=1)


        return pred_OOD_eucl_dist, pred_OOD_zscore, pred_OOD_js, pred_OOD_wass_dist

