import torch
import numpy as np
from copy import deepcopy

from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import entropy as js_dist
from scipy.spatial.distance import jensenshannon as js_dist
from scipy.spatial.distance import mahalanobis

from sklearn.model_selection import train_test_split


def wass_dist(p, q):
    # Compute the cumulative distribution functions
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)

    # Compute the squared difference of the CDFs
    diff = (cdf_p - cdf_q) ** 2

    # Compute the 2-Wasserstein distance
    distance = np.sqrt(np.sum(diff))

    return distance



def tv_div(p, q):
    """
    Compute the total variation distance between two probability distributions p and q.
    :param p:
    :param q:
    :return:
    """
    return 0.5 * np.sum(np.abs(p - q))


class OOD_detec_class:
    """
    Wrapper class for the OOD detection methods
    """

    def __init__(self, num_classes=3):
        """
        Initialize the OOD detector
        :param input_dim:
        :param model:
        :param lr:
        :param num_ind_pts:
        """
        self.X_train = None
        self.y_train = None

        self.model = None
        self.kernel = 1.0 * RBF(1.0)

        self.ref_dist = None
        alpha_dict = {0.1: [], 0.05: [], 0.01: []}
        self.th_dict = {'eucl_dist' : deepcopy(alpha_dict), 'zscore': deepcopy(alpha_dict),
                        'max_softmax': deepcopy(alpha_dict), 'maha_dist': deepcopy(alpha_dict),
                        'js_dist': deepcopy(alpha_dict), 'wass_dist': deepcopy(alpha_dict)}

    def train_OOD(self, X, y, K=5, val_size=0.2, rng=42):
        """
        Train the OOD model.
        Given (X, y), for K-folds split (X, y) into (X_train, y_train) and (X_test, y_val). Train the model on
        (X_train, y_train) and compute the thresholds on (X_test, y_val). Update the thresholds in the th_dict.
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

            print(f"Training fold {k + 1}/{K}...")

            # Split the data into train and validation
            seed_k = rng * k
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=val_size, random_state=seed_k)

            self.X_train = X_train
            self.y_train = y_train

            # Define the reference distribution as the raw prevalence of the classes in the y training set
            # self.ref_dist = np.bincount(y_train) / len(y_train)
            self.ref_dist = np.array([1 / len(np.unique(y_train))] * len(np.unique(y_train)))

            # Initialize model
            self.model = MLPClassifier(hidden_layer_sizes=(32,), random_state=seed_k, max_iter=100, verbose=False,
                                       activation='relu', solver='lbfgs')
            self.model.fit(X_train, y_train)
            y_prob = np.array(self.model.predict_proba(X_test))

            # Compute the KL divergence
            eucl_diff_, zscore_, maxsoft_, maha_dist_, js_dist_, wass_dist_ = [], [], [], [], [], []
            for i in range(X_test.shape[0]):
                y_prob_i = y_prob[i]

                eucl_diff_i = torch.max(torch.norm(X_test[i] - X_train, dim=0))
                zscore_i = torch.max(torch.abs((X_test[i] - X_train.mean(dim=0)) / X_train.std(dim=0)))
                maxsoft_i = np.max(y_prob_i)
                maha_dist_i = mahalanobis(X_test[i], X_train.mean(dim=0), torch.cov(X_train.T))
                js_dist_i = js_dist(y_prob_i, self.ref_dist)
                wass_dist_i = wass_dist(y_prob_i, self.ref_dist)

                eucl_diff_.append(eucl_diff_i)
                zscore_.append(zscore_i)
                maxsoft_.append(maxsoft_i)
                maha_dist_.append(maha_dist_i)
                js_dist_.append(js_dist_i)
                wass_dist_.append(wass_dist_i)

            eucl_diff_, zscore_ = np.array(eucl_diff_), np.array(zscore_)
            maxsoft_, maha_dist_ = np.array(maxsoft_), np.array(maha_dist_)
            js_dist_, wass_dist_ = np.array(js_dist_), np.array(wass_dist_)

            # Compute the thresholds as the \alpha quantile of the statistics and populate the th_dict
            for alpha in [0.1, 0.05, 0.01]:
                self.th_dict['eucl_dist'][alpha].append(np.quantile(eucl_diff_, 1 - alpha))
                self.th_dict['zscore'][alpha].append(np.quantile(zscore_, 1 - alpha))
                self.th_dict['max_softmax'][alpha].append(np.quantile(maxsoft_, 1 - alpha))
                self.th_dict['maha_dist'][alpha].append(np.quantile(maha_dist_, 1 - alpha))
                self.th_dict['js_dist'][alpha].append(np.quantile(js_dist_, 1 - alpha))
                self.th_dict['wass_dist'][alpha].append(np.quantile(wass_dist_, 1 - alpha))

        # Save only the 95th percentile across folds for each alpha
        for alpha in [0.1, 0.05, 0.01]:
            self.th_dict['eucl_dist'][alpha] = np.quantile(self.th_dict['eucl_dist'][alpha], 0.95)
            self.th_dict['zscore'][alpha] = np.quantile(self.th_dict['zscore'][alpha], 0.95)
            self.th_dict['max_softmax'][alpha] = np.quantile(self.th_dict['max_softmax'][alpha], 0.95)
            self.th_dict['maha_dist'][alpha] = np.quantile(self.th_dict['maha_dist'][alpha], 0.95)
            self.th_dict['js_dist'][alpha] = np.quantile(self.th_dict['js_dist'][alpha], 0.95)
            self.th_dict['wass_dist'][alpha] = np.quantile(self.th_dict['wass_dist'][alpha], 0.95)

    def predict_OOD(self, X, alpha=0.05):
        """
        Predict if the test set (X, y) is OOD.
        :param X:
        :param y:
        :param alpha:
        :return:
        """
        eucl_diff_, zscore_, maxsoft_, maha_dist_, js_dist_, wass_dist_ = [], [], [], [], [], []
        y_prob = np.array(self.model.predict_proba(X))

        for i in range(X.shape[0]):
            y_prob_i = y_prob[i]

            eucl_diff_i = torch.max(torch.norm(X[i] - self.X_train, dim=0))
            zscore_i = torch.max(torch.abs((X[i] - self.X_train.mean(dim=0)) / self.X_train.std(dim=0)))
            maxsoft_i = np.max(y_prob_i)
            maha_dist_i = mahalanobis(X[i], self.X_train.mean(dim=0), torch.cov(self.X_train.T))
            js_dist_i = js_dist(y_prob_i, self.ref_dist)
            wass_dist_i = wass_dist(y_prob_i, self.ref_dist)

            eucl_diff_.append(eucl_diff_i)
            zscore_.append(zscore_i)
            maxsoft_.append(maxsoft_i)
            maha_dist_.append(maha_dist_i)
            js_dist_.append(js_dist_i)
            wass_dist_.append(wass_dist_i)

        eucl_diff_, zscore_ = np.array(eucl_diff_), np.array(zscore_)
        maxsoft_, maha_dist_ = np.array(maxsoft_), np.array(maha_dist_)
        js_dist_, wass_dist_ = np.array(js_dist_), np.array(wass_dist_)

        print("JS distance: ", js_dist_[-1])
        print("Wasserstein distance: ", wass_dist_[-1])

        # Compute the thresholds as the 1-\alpha quantile of the statistics and populate the th_dict
        # Predict OOD
        pred_OOD_eucldiff = eucl_diff_ > self.th_dict['eucl_dist'][alpha]
        pred_OOD_zscore = zscore_ > self.th_dict['zscore'][alpha]
        pred_OOD_maxsoft = maxsoft_ > self.th_dict['max_softmax'][alpha]
        pred_OOD_maha_dist = maha_dist_ > self.th_dict['maha_dist'][alpha]
        pred_OOD_js_dist = js_dist_ > self.th_dict['js_dist'][alpha]
        pred_OOD_wass_dist = wass_dist_ > self.th_dict['wass_dist'][alpha]

        return pred_OOD_eucldiff, pred_OOD_zscore, pred_OOD_maha_dist, pred_OOD_maxsoft, pred_OOD_js_dist, pred_OOD_wass_dist

