import torch
import matplotlib.pyplot as plt

# From scikit learn compute the ROC AUC, tpr, fpr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix

# Import from sklearn to generate classification data
from sklearn.datasets import make_classification


def create_ordinal(N=500, P=5):

    # Create 3 unbalanced classes
    X, y = make_classification(n_samples=N, n_features=P, n_informative=P, n_redundant=0, n_clusters_per_class=1,
                               n_classes=3, weights=[0.6, 0.1, 0.3], class_sep=1.5, random_state=42, scale=1.0, flip_y=0.1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def ordinal_shift(y, shift_type='Far'):

    # Deep copy of y
    y_OOD = y.clone()

    # Switch most of the 2s to 0s and vice versa, but not all
    if shift_type == 'Far':
        y_OOD[y == 0] = 2
        y_OOD[y == 2] = 0
    elif shift_type == 'Near':
        y_OOD[y == 0] = 1
        y_OOD[y == 1] = 0
    else:
        raise ValueError('shift_type must be either "Far" or "Near"')

    return y_OOD


def two_x_f(x, noise=True):

    signal = x[:, 0] + x[:, 1] + 1.0*torch.cos(x[:, 0])
    signal.squeeze()
    var_eps = torch.randn_like(signal) * .5

    if noise:
        y = signal + var_eps
    else:
        y = signal

    return y


def single_x_f(x, noise=True):

    signal = x + 1.0*torch.cos(x)
    var_eps = torch.randn_like(x) * .5

    if noise:
        y = signal + var_eps
    else:
        y = signal

    return y


def single_x_f_OOD(x, OOD_type='Far', noise=True):

    if OOD_type == 'Far':
        signal = 1.5 + x + 1.0*torch.cos(x)
        var_eps = torch.randn_like(x) * .5
    elif OOD_type == 'Near':
        signal = 0.5 + x + 1.0*torch.cos(x)
        var_eps = torch.randn_like(x) * 1.0
    else:
        raise ValueError('OOD_type must be either "Far" or "Near"')

    if noise:
        y = signal + var_eps
    else:
        y = signal

    return y


def plot_ID_OOD(x_test_ID, x_test_OOD, y_test_ID, y_test_OOD, type='Semantic', OOD_type='Far'):
    """
    Generate a plot of the ID and OOD data using the 2 functions above
    :param x_test_ID: 
    :param x_test_OOD: 
    :return: 
    """

    # Plot model
    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Plot ID and OOD data
    ax.scatter(x_test_ID.numpy(), y_test_ID, c='r', marker='.', label='ID', alpha=0.5)
    ax.scatter(x_test_OOD.numpy(), y_test_OOD, c='b', marker='.', label='OOD', alpha=0.5)

    # Plot the underlying functions
    x = torch.linspace(torch.min(torch.min(x_test_ID), torch.min(x_test_OOD)),
                       torch.max(torch.max(x_test_ID), torch.max(x_test_OOD)), 1000)
    f_ID = single_x_f(x, noise=False).reshape(-1)

    if type == 'Semantic':

        if OOD_type == 'Far':
            f_OOD = single_x_f_OOD(x, noise=False, OOD_type='Far').reshape(-1)
        else:
            f_OOD = single_x_f_OOD(x, noise=False, OOD_type='Near').reshape(-1)
        ax.plot(x, f_ID, color='r', linestyle='dashed', alpha=0.7)
        ax.plot(x, f_OOD, color='b', linestyle='dashed', alpha=0.7)
    else:
        ax.plot(x, f_ID, color='r', linestyle='-')

    # Set grid
    ax.legend(loc="best")
    ax.grid(True, which='both', alpha=0.3)

    # Set tight layout
    plt.tight_layout()
    ax.set_title('ID and OOD Data', fontsize=15)
    plt.show()


def plot_gp_fit(gp_model, likelihood, x, y, title, filename='./logs/plots/gp_fit.pdf'):

    gp_model.eval()
    likelihood.eval()

    with torch.no_grad():
        f_dist = gp_model(x)
        mean = f_dist.mean.detach()
        f_lower, f_upper = f_dist.confidence_region()
        y_dist = likelihood(f_dist)
        y_lower, y_upper = y_dist.confidence_region()

    # Plot model
    # Create subplots
    x = x[:, 0]

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    line, = ax.plot(x, mean.detach(), color='b')
    ax.fill_between(x.reshape(-1), f_lower.detach(), f_upper.detach(), color=line.get_color(), alpha=0.3, label="p(f)")
    ax.fill_between(x.reshape(-1), y_lower, y_upper, color=line.get_color(), alpha=0.1, label="p(y)")
    ax.scatter(x.numpy(), y, c='k', marker='.')
    ax.legend(loc="best")
    ax.set(xlabel="", ylabel="y")

    # Draw vertical line at -3 and 3 to denote training data
    ax.axvline(x=-3, color='r', linestyle='--', linewidth=1)
    ax.axvline(x=3, color='r', linestyle='--', linewidth=1)

    # Set grid
    ax.grid(True, which='both', alpha=0.3)

    # Set y axis to be between -2.05 and 4.05
    # ax.set_ylim([-3.0, 4.0])

    # Set tight layout
    plt.tight_layout()
    ax.set_title(title, fontsize=15)
    plt.show()

    # Save plot
    fig.savefig(filename)


def compute_metrics(OOD_pred, OOD_labels):
    """
    Compute the AUROC, TPR, FPR, F1 score for each method
    :param OOD_pred_95:
    :param OOD_labels:
    :return:
    """

    # Compute confusion matrix
    cm = confusion_matrix(OOD_labels, OOD_pred)
    tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])

    # Compute AUROC
    auroc = roc_auc_score(OOD_labels, OOD_pred)

    return auroc, tpr, fpr
