# Standard Data Science Helpers
import  matplotlib
matplotlib.use('TKAgg')
import numpy as np
import pandas as pd
import scipy
import random


from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)


from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import roc_auc_score

import cufflinks as cf
cf.go_offline(connected=True)
cf.set_config_file(colorscale='plotly', world_readable=True)

# Extra options
pd.options.display.max_rows = 30
pd.options.display.max_columns = 25

# Show all code cells outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import os
from IPython.display import Image, display, HTML
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from sklearn.base import clone



#Some utility functions.

class DataContainer():
    def __init__(self):
        pass

    def set_x(self, X):
        self.X = X

    def set_y(self, y):
        self.y = y





from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def classification_assessment(X_test, y_test, y_test_predicted, clf, data_container=None):
    print(classification_report(y_test, y_test_predicted))
    cnf_matrix = confusion_matrix(y_test, y_test_predicted, labels=[1,0])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['incident=1','no incident=0'],normalize= False,
                          title='Confusion matrix')
    ns_probs = [0 for _ in range(len(y_test))]

    # plot ROC
    if data_container is None:
        plt.figure()
    lr_probs = clf.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    #     print(lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, ns_thresh = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, lr_thresh = roc_curve(y_test, lr_probs)
    # plot the roc curve for the model
    if data_container is None:
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
    else:
        roc_ax = data_container.roc_ax
        roc_ax.plot(lr_fpr, lr_tpr, marker='.', label='Logistic, auc:{:0.2f}'.format(lr_auc))
        roc_ax.legend()
    return lr_probs


def dilute_class(X, y, class_tag, dilute_factor):
    y_indices = y==class_tag
    num_class_entries = sum(y_indices)
    diluted_ys_indices =  random.sample(range(num_class_entries), int(num_class_entries*dilute_factor))
    diluted_ys = y[y_indices][diluted_ys_indices]
    diluted_X_other_classes = X[~y_indices]
    diluted_X_class = X[y_indices][diluted_ys_indices]
    diluted_X = np.concatenate([diluted_X_other_classes, diluted_X_class])
    diluted_y = np.concatenate([ y[~y_indices], diluted_ys])

    return diluted_X, diluted_y






data_container = DataContainer()
# mu = -2
# sigma = 2
# k = 1.5
# x = np.arange(mu+0.05, mu+10*sigma, 0.05);
# F = 1-np.exp(-np.power((-(mu-x)/sigma), k));

mu = 5
sigma = 7
xi = 2.5

x = np.arange(-100, 100, 0.05)
F= np.exp(-np.power((1+xi*((x-mu)/sigma)), -1/xi))

num_repetitions = 10000;
xx = np.array([x]*num_repetitions);
xx = xx.reshape(1, num_repetitions*len(x))[0];

u1 = np.random.rand(len(xx));#
# u2 = np.random.rand(len(x))
# F_inverse_u = xx - sigma*np.power((-np.log(1-u1)), 1/k);#
F_inverse_u = sigma/xi*(np.power(-np.log(u1), -xi) - 1) + xx
# print(F_inverse_u.shape);
# F_inverse_u
x_ones = xx[F_inverse_u < 0];
x_zeros = xx[F_inverse_u >= 0];
n1, bins1,_ = plt.hist(x_ones, bins=np.append(x, x[-1] + 0.05), color='r', label= "ones", alpha=0.5, density=False);
n0, bins0,_ = plt.hist(x_zeros, bins=np.append(x, x[-1] + 0.05), color='b', label= "zeros", alpha=0.5, density=False);
empirical_probs = list(map(lambda tup : tup[0]/(tup[0]+tup[1]), zip(n1,n0) ));

# F_0 = 1-np.exp(-np.power((-(-x)/sigma), k));
# plt.plot(x, F_0, color='y');
plt.plot(x, empirical_probs, color='g');
F_sig = 1/(1+np.exp(-x/(0.5*sigma)));
plt.plot(x, F_sig, color='k');
plt.legend();


def generate_x_y_gev(n0, n1, bins, total_num_samples):
    X = np.array([]);
    y = np.array([]);
    binsize = bins[-1]-bins[-2];
    #Here I assume that the number of samples in each bin are more or less equal.
    dilute_factor = (sum(n0) + sum(n1))/total_num_samples

    for tupp in zip(n0, n1, bins ):
        #         print(tupp)
        x_zeros = tupp[2] + binsize*np.random.rand(int(tupp[0]/dilute_factor));
        x_ones = tupp[2] + binsize*np.random.rand(int(tupp[1]/dilute_factor));
        X = np.concatenate([X, x_zeros, x_ones]);
        y = np.concatenate([y, np.zeros(len(x_zeros)), np.ones(len(x_ones)) ]);
    #         set_trace()
    return X,y

total_num_samples = 200000
X,y = generate_x_y_gev(n0, n1, bins1, total_num_samples);

plt.figure();
X_ones = X[y==1];
X_zeros = X[y==0];
plt.hist(X_ones, bins=bins1, color = 'r', alpha = 0.4);
plt.hist(X_zeros, bins=bins1, color = 'b', alpha = 0.4);
plt.show();

data_container.set_x(X.reshape(len(X), 1));
data_container.set_y(y);


X_train, X_test, y_train, y_test = train_test_split(data_container.X, data_container.y, test_size=0.2);
data_container.X_train = X_train;
data_container.X_test = X_test;
data_container.y_train = y_train;
data_container.y_test = y_test;
probs_fig, probs_axs = plt.subplots(2,1);
data_container.probs_ax, data_container.roc_ax = probs_axs;
data_container.probs_ax.set_title("Classifier's probabilities over test set");
data_container.probs_ax.set_xlabel("x");
data_container.probs_ax.set_ylabel("prob(y(x)=1)");
data_container.probs_fig = probs_fig

#Overlay the empirical probabilities (taken from the actual "real" distributions).
def calc_bin_prob(bn_df):
    #     set_trace()
    bin_x = bn_df[0].left
    df = bn_df[1]
    prob_1 = np.nan if len(df) == 0 else df['y'].sum()/len(df)
    return bin_x, prob_1

X = data_container.X
y = data_container.y
X_y_df = pd.DataFrame({'X':X.T[0], 'y':y});
#Extend range otherwise inerval may be nan.
bin_size = 0.05
X_y_df['x_binned'] = pd.cut(X_y_df['X'], bins=np.arange(X.min()-bin_size, X.max() + bin_size, bin_size));
x_binned = []
prob_one_binned = []
for bn_df in X_y_df.groupby('x_binned'):
    bin_x, prob_1 = calc_bin_prob(bn_df)
    x_binned.append(bin_x)
    prob_one_binned.append(prob_1)
prob_one_binned = np.array(prob_one_binned)
data_container.probs_ax.plot(x_binned, prob_one_binned, 'r', label="empirical P(y=1)")
plt.show(block=False)

# derivative = (prob_one_binned[1:]-prob_one_binned[:-1])/bin_size
# data_container.probs_ax.plot(x_binned[:-1], derivative, 'y', label="derivative")
# data_container.probs_ax.plot(x_binned, prob_one_binned/(1-prob_one_binned), 'r+', label="empirical P(y=1)/P(y=0)")
# data_container.probs_ax.plot(x_binned, np.log(prob_one_binned/(1-prob_one_binned)), 'rx', label="empirical log(P(y=1)/P(y=0))")

data_container.probs_ax.legend();
#Roc
x_test_binned = X_y_df.set_index('X').loc[X_test.T[0],'x_binned'].values;
# if np.nan in x_test_binned:
#     print("Found NAN!!")
#     for i,intr in enumerate(x_test_binned):
#         if isinstance(intr, float):
#             print(f"The nan is at index {i}, corresponding to x_test {X_test[i]}")
def get_left(interval):
    return interval.left




############## GEV #######################
# import scipy.optimize.fmin_cg
def gev(x, tau):
    return np.exp( -np.power(1 + tau*x, -1/tau))


class GevRegressionCV():

    def __init__(self):
        self.curr_iter_x_dot_theta = None
        self.curr_iter_pi = None
        self.tau_ = 0
        self.intercept_ = 0
        self.coefs_ = np.array([])

    #gev_instance is an attempt to avoid the double calculations, but I don't know which of the methods (grad, err_func) are
    # accessed first, so it requires a more sophisticated mechanism.
    @staticmethod
    def err_func(tau_theta, X, y, reg_param, gev_instance = None):

        tau=tau_theta[0]
        theta = tau_theta[1:]
        x_dot_theta = np.dot(X, theta)
        data_container.probs_ax.scatter(X[:,1], gev(x_dot_theta, tau), s=0.5 )
        plt.show(block=False)
        print(f"tau:{tau}. theta: {theta}. tau*X*theta: {tau*x_dot_theta}")
        pi = gev(x_dot_theta, tau)
        inds = ~((np.isnan(pi)) | (pi == 0) )
        pi = pi[inds]
        y = y[inds]
        h_theta = -1/len(y) *( np.dot(y, np.log(pi)) + np.dot(1-y, np.log(1 - pi))) + \
                  reg_param/(2*len(y))*np.dot(tau_theta, tau_theta)
        #Save for the gradient invocation, to save repeated evaluation.
        #         gev_instance.curr_iter_x_dot_theta =  x_dot_theta
        #         gev_instance.curr_iter_pi = pi
        print(f"h_theta: {h_theta}")
        return h_theta

    @staticmethod
    def grad(tau_theta, X, y, reg_param, gev_instance = None):
        #These two are calculated twice - once in this function, and one in err_func. Make this more efficient later if necessary.
        tau=tau_theta[0]
        theta = tau_theta[1:]
        x_dot_theta = np.dot(X, theta)
        pi = gev(x_dot_theta, tau)
        inds = ~((np.isnan(pi)) | (pi == 0) )
        pi = pi[inds]
        y = y[inds]
        x_dot_theta = x_dot_theta[inds]
        #         x_dot_theta = gev_instance.curr_iter_x_dot_theta
        #         pi = gev_instance.curr_iter_pi
        mult_vec = np.log(pi)*(y-pi)/( (1+tau*x_dot_theta)*(1-pi) )
        grad_theta = -1/len(y)*np.dot(X[inds].T, mult_vec)
        #Regularization
        grad_theta[1:] += reg_param / len(y) * theta[1:] #Don't regularize the intercept.

        u = 1/(tau*tau)*np.log((1+tau*x_dot_theta)) - x_dot_theta/( tau*(1+tau*x_dot_theta))
        v = (y-pi)*np.log(pi)/(1-pi)
        grad_tau = -1/len(y)*np.dot(u,v)  + reg_param / len(y)*tau

        return np.append(grad_tau, grad_theta )

    def fit(self, X,y):
        def constr_fun(tau_theta):
            return tau_theta[0]*X.dot(tau_theta[1:])

        def constr_jac(tau_theta):
            num_rows = X.shape[0]
            df_dtau = X.dot(tau_theta[1:]).reshape(num_rows, 1)
            df_dtheta = tau_theta[0]*X
            return np.concatenate((df_dtau, df_dtheta), axis=1)

        def constr_hess(tau_theta, v):
            first_row = np.append(np.array([0]), v.dot(X)).reshape(1, len(tau_theta))
            first_col = v.dot(X).reshape(len(tau_theta)-1,1)
            sub_mat1 = np.concatenate((first_col, np.zeros([len(first_col), len(first_col)])) , axis=1)
            hess = np.concatenate((first_row, sub_mat1))
            return hess

        regularization_param = 1 #TODO: optimize on this!!
        #This function requires two separate functions calls - one for the log-likelihood,
        # and the other for the gradient calculation. I'm saving the
        #Add intercept
        first_col = np.array([np.ones(X.shape[0])]).reshape(X.shape[0], 1)
        X = np.concatenate((first_col, X), axis=1)
        #[\tau, \theta_0, \theta_1 ... \theta_k]
        # initial_beta0 = np.log(-np.log(np.mean(y)))
        # initial_tau = -1/initial_beta0 + (0.1 if initial_beta0 > 0 else -0.1) #Satisfy constraint
        # ## Just for debugging
        # initial_tau = 1
        # initial_guess = np.append([initial_tau, initial_beta0], np.zeros(X.shape[1] - 1))
        initial_guess = np.array([0.8, -1.375, 1.25])
        lb = -1*np.ones(len(X))
        ub = np.inf*np.ones(len(X))
        constr = scipy.optimize.NonlinearConstraint(constr_fun, lb, ub, jac=constr_jac,
                                                    hess=constr_hess)
        opt_res = scipy.optimize.minimize(self.err_func, initial_guess,
                                                jac=self.grad, args=(X, y, regularization_param), method='SLSQP',
                                                constraints=constr)
        opt_theta_tau = opt_res.x
        self.tau_ = opt_theta_tau[0]
        self.intercept_ = opt_theta_tau[1]
        self.coefs_ = opt_theta_tau[2:]

#     def predict(self, X_to_classify):

corrected_clf = GevRegressionCV().fit(data_container.X, data_container.y)
x=0

#plt.show()


