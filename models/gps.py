import math
import torch
import gpytorch

from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class ExactGP(gpytorch.models.ExactGP):
    """
    Exact Gaussian Process model for the dynamics of the environment.
    :args:
    - input_dim (int): The dimension of the input space.
    - output_dim (int): The dimension of the output space.
    """

    def __init__(self, train_x, train_y):
        super(ExactGP, self).__init__(train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1)),
        #                                                  num_dims=train_x.size(-1))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def update_posterior_xi(self, x_train, x_test, y_train, y_test, likel):
        """
        Closed-form update of the posterior distribution over the training data.
        :args:
        - x (torch.Tensor): The training inputs.
        - y (torch.Tensor): The training outputs.
        - x_test (torch.Tensor): The test inputs.
        - likel (gpytorch.likelihoods): The likelihood function, to get the noise
        """
        self.eval()
        with torch.no_grad():

            # Before concatenating the data, if any of them is one-dimensional, unsqueeze it on first dimension
            if len(x_test.shape) == 1:
                x_test = x_test.unsqueeze(1)
            if len(y_test.shape) == 0:
                y_test = y_test.unsqueeze(0)

            # Concatenate training and test data
            # x_cat = torch.cat([x_train, x_test], dim=0)
            # y_cat = torch.cat([y_train, y_test], dim=0)

            # Get the mean functions
            # mean_x_star = self.mean_module(x_test)  # shape: (m,)
            # mean_x_cat = self.mean_module(x_cat)  # shape: (n+m,)

            mean_x_star = self.forward(x_test).mean
            mean_x_n = self.forward(x_train).mean

            # Get the covariance matrices
            k_xx_cat = self.covar_module(x_train, x_train)  # shape: (n+m, n+m)
            k_x_star_cat = self.covar_module(x_train, x_test)  # shape: (n+m, m)
            k_star_star = self.covar_module(x_test, x_test)  # shape: (m, m)

            # Add the observation noise to the covariance matrix
            noise = likel.noise
            k_xx_cat_noise = k_xx_cat + noise * torch.eye(k_xx_cat.shape[0])  # shape: (n+m, n+m)
            k_xx_cat_noise = torch.tensor(k_xx_cat_noise.numpy())

            # Get updated mean and covariance
            post_mean = mean_x_star + k_x_star_cat.t().matmul(torch.inverse(k_xx_cat_noise)).matmul(y_test - mean_x_n)
            post_covar = k_star_star - k_x_star_cat.t().matmul(torch.inverse(k_xx_cat_noise)).matmul(k_x_star_cat)

            # Get only diagonal of the covariance matrix
            post_covar = torch.diag(torch.tensor(post_covar.numpy()))

        return post_mean, post_covar


class SVGP(gpytorch.models.ApproximateGP):
    """
    Sparse Variational Gaussian Process model for the dynamics of the environment.
    :args:
    - input_dim (int): The dimension of the input space.
    - output_dim (int): The dimension of the output space.
    - num_ind_pts (int): The number of inducing points to use in the model.
    """

    def __init__(self, input_dim, num_ind_pts):
        inducing_points = torch.rand(num_ind_pts, input_dim)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super(SVGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim),
                                                         num_dims=input_dim)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def update_posterior_xi(self, x_train, x_test, y_train, y_test, likel):
        """
        Closed-form update of the posterior distribution over the training data.
        :args:
        - x (torch.Tensor): The training inputs.
        - y (torch.Tensor): The training outputs.
        - x_test (torch.Tensor): The test inputs.
        - likel (gpytorch.likelihoods): The likelihood function, to get the noise
        """
        self.eval()
        with torch.no_grad():

            # Before concatenating the data, if any of them is one-dimensional, unsqueeze it on first dimension
            if len(x_test.shape) == 1:
                x_test = x_test.unsqueeze(1)
            if len(y_test.shape) == 0:
                y_test = y_test.unsqueeze(0)

            # Concatenate training and test data
            x_cat = torch.cat([x_train, x_test], dim=0)
            y_cat = torch.cat([y_train, y_test], dim=0)

            # Get the mean functions
            # mean_x_star = self.mean_module(x_test)  # shape: (m,)
            # mean_x_cat = self.mean_module(x_cat)  # shape: (n+m,)

            mean_x_star = self.forward(x_test).mean
            mean_x_n = self.forward(x_train).mean

            # Get the covariance matrices
            k_xx_cat = self.covar_module(x_train)  # shape: (n+m, n+m)
            k_x_star_cat = self.covar_module(x_train, x_test)  # shape: (n+m, m)
            k_star_star = self.covar_module(x_test)  # shape: (m, m)

            # Add the observation noise to the covariance matrix
            noise = likel.noise
            k_xx_cat_noise = k_xx_cat + noise * torch.eye(k_xx_cat.shape[0])  # shape: (n+m, n+m)
            k_xx_cat_noise = torch.tensor(k_xx_cat_noise.numpy())

            # Get updated mean and covariance
            post_mean = mean_x_star + k_x_star_cat.t().matmul(torch.inverse(k_xx_cat_noise)).matmul(y_test - mean_x_n)
            post_covar = k_star_star - k_x_star_cat.t().matmul(torch.inverse(k_xx_cat_noise)).matmul(k_x_star_cat)

        return post_mean, torch.tensor(post_covar.numpy())


# Setup likelihood, objective function, and optimizer
def setup_model(gp_model, train_y, learning_rate=0.1):

    if isinstance(gp_model, SVGP):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        objective_function = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=train_y.numel())
        optimizer = torch.optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=learning_rate)
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        objective_function = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
        optimizer = torch.optim.Adam(gp_model.parameters(), lr=learning_rate)

    return likelihood, objective_function, optimizer


# Train model
def train_model(gp_model, X, y, training_iterations=50, verbose=True):
    # Train
    gp_model['model'].train()
    gp_model['likelihood'].train()
    for _ in range(training_iterations):
        output = gp_model['model'](X)
        loss = -gp_model['objective_function'](output, y)
        loss.backward()
        gp_model['optimizer'].step()
        gp_model['optimizer'].zero_grad()

        if verbose and _ % 10 == 0:
            print(f'Iter {_:d}/{training_iterations:d} - Loss: {loss.item():.3f}')


# Predict next state
def predict_model(gp_model, likelihood, state_action):
    gp_model.eval()
    likelihood.eval()
    with torch.no_grad():

        f_dist = gp_model(state_action)
        y_dist = likelihood(f_dist)
        y_covar = y_dist.variance

        mean = f_dist.mean
        covar = f_dist.variance
        f_lower, f_upper = f_dist.confidence_region()

    return mean, covar, y_covar, f_lower, f_upper



def update_GP_model(gp_dict, x_train, x_test, y_train, y_test):
    """
    Update the model with new data.
    :args:
    - gp_model (ExactGP or SVGP): The model to update.
    - x_train (torch.Tensor): The training inputs.
    - y_train (torch.Tensor): The training outputs.
    - x_test (torch.Tensor): The test inputs.
    - y_test (torch.Tensor): The test outputs.
    - likel (gpytorch.likelihoods): The likelihood function, to get the noise
    """
    # Concatenate training and test data
    x_cat = torch.cat([x_train, x_test], dim=0)
    y_cat = torch.cat([y_train, y_test], dim=0)

    print(x_cat.shape)
    print(y_cat.shape)

    # Update the training data
    gp_dict['model'].set_train_data(x_cat, y_cat, strict=False)

    # Update the model
    likelihood, objective_function, optimizer = setup_model(gp_dict['model'], y_cat)
    gp_dict['likelihood'] = likelihood
    gp_dict['objective_function'] = objective_function
    gp_dict['optimizer'] = optimizer

    # Re-train for just a few iterations
    gp_dict['model'].train()
    gp_dict['likelihood'].train()
    train_model(gp_dict, x_cat, y_cat, training_iterations=50)

    # Update the posterior distribution
    gp_dict['model'].eval()
    gp_dict['likelihood'].eval()
    post_x = gp_dict['model'](x_test)
    post_mean = post_x.mean
    post_covar = post_x.variance

    return post_mean, post_covar


