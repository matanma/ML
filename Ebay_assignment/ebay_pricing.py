#!C:\Users\Matan\anaconda3\python.exe
##!/usr/bin/env python
import pandas as pd
import argparse
import sys, os
import scipy
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt




def calc_theta_score(theta_vec, X, Y, R,  num_consumers, num_features, regularization_lambda ):
    theta = theta_vec.reshape( num_consumers, num_features)
    j_theta = 1 / 2 * (((X.dot(theta.T)-Y)*R)**2).sum() + regularization_lambda/2*(theta**2).sum()
    return j_theta

def calc_theta_grad(theta_vec, X, Y, R,  num_consumers, num_features, regularization_lambda ):
    theta = theta_vec.reshape(num_consumers, num_features)
    return (((X.dot(theta.T) - Y)*R).T.dot(X) + regularization_lambda*theta).flatten()


class EbayPricingDemo():
    pricing_strategies = ['static', 'adjust_by_stds']

    def __init__(self, products_filepath, num_consumers, commision_ratio, propensity_range = (0,5), initial_density = 0.35):

        self.product_names, self.avg_product_prices, self.products_stds, self.products_features = self.load_product_prices_and_features(products_filepath)
        self.propensity_range = propensity_range
        self.commision_ratio = commision_ratio
        self.propensity_purchase_distribution = self.init_propensity_purchase_distribution()
        self.base_pricing_matrix = self.init_pricing_matrix(self.avg_product_prices, num_consumers)
        self.dynamic_pricing_matrix = np.copy(self.base_pricing_matrix)
        self.propensity_matrix = self.init_propensity_matrix(initial_density, self.base_pricing_matrix.shape )
        self.dynamic_probs = np.zeros(self.base_pricing_matrix.shape)



    def  load_product_prices_and_features(self, products_filepath="mercari-price-suggestion-challenge/train_small.csv"):
        extension = products_filepath.split(".")[-1]
        raw_data_df = pd.read_csv(products_filepath, sep=',' if extension == "csv" else "\t" )
        #Extract some features.
        def split_cat(text):
            try:
                return text.split("/")
            except:
                return ("No Label", "No Label", "No Label")

        raw_data_df['general_cat'], raw_data_df['subcat_1'], raw_data_df['subcat_2'] = \
            zip(*raw_data_df['category_name'].apply(lambda x: split_cat(x)))
        sub_df = raw_data_df[['general_cat']]
        enc = OneHotEncoder().fit(sub_df)
        products_features = enc.transform(sub_df).A
        #TODO: Add textual features

        #Approximations for the standard deviations of the product's prices.
        raw_data_df['stds'] = np.ones(len(raw_data_df))
        tmp = raw_data_df.set_index(["category_name", 'shipping'])
        for name_ship, df in raw_data_df.groupby(["category_name", 'shipping']):
            tmp.loc[name_ship, 'stds'] = max(1, df['price'].std())
        stds_vec = tmp.set_index('train_id')['stds'].values
        return raw_data_df['name'].values, raw_data_df['price'], stds_vec, products_features


    #Returns a matrix in which column j is the probability distribution corresponding to propencity j (in ascending order)
    def init_propensity_purchase_distribution(kernel = np.arange(5,0,-1)/100):
        kernel = np.arange(5, 0, -1) / 100
        return kernel[:, np.newaxis].dot(np.arange(0, 6)[np.newaxis, :])


    #Initially - have all consumer prices identical
    def init_pricing_matrix(self, avg_product_prices, num_consumers):
        return avg_product_prices[:, np.newaxis].dot(np.ones(num_consumers)[ np.newaxis, :])


    def perform_conent_based_filtering(self, products_features, known_propensity_matrix):
        regularization_lambda = 1
        num_consumers = known_propensity_matrix.shape[-1]
        num_features = products_features.shape[-1]
        theta_initial = np.random.rand(num_consumers, num_features)
        R = known_propensity_matrix >= 0 #Valid propensities.
        y_mean_tmp = np.nan_to_num( [np.mean(known_propensity_matrix[i,R[i,:]]) for i in
                                     range(known_propensity_matrix.shape[0]) ])
        y_mean = np.array(y_mean_tmp)[:, np.newaxis]
        known_propensity_matrix_normed = known_propensity_matrix - y_mean
        optimal_theta = scipy.optimize.minimize( calc_theta_score, theta_initial,
                                                 args=( products_features, known_propensity_matrix_normed, R,  num_consumers, num_features, regularization_lambda ),
                                                 method='CG', jac = calc_theta_grad,
                                                 tol=1e-19
                                                 )
        regression_theta_mat = optimal_theta.x.reshape(*(theta_initial.shape))
        regressed_propensity_matrix = products_features.dot(regression_theta_mat.T) + y_mean
        regressed_propensity_matrix_int = np.ceil(regressed_propensity_matrix).astype(int)
        #Amend regression noises
        regressed_propensity_matrix_int[regressed_propensity_matrix_int < self.propensity_range[0]] = self.propensity_range[0]
        regressed_propensity_matrix_int[regressed_propensity_matrix_int > self.propensity_range[-1]] =  self.propensity_range[-1]
        return regressed_propensity_matrix_int



    def init_propensity_matrix(self, initial_density, shape):
        S = scipy.sparse.random(*shape, density=initial_density)
        # Initialize the propensity matrix to contain initial_density entries in the propensity_range, and the others set to -1
        self.known_propensity_matrix =  np.ceil(S.A*(self.propensity_range[-1]+1)).astype(int) - 1
        #Perform content based filtering to estimate the empty entries.
        return self.perform_conent_based_filtering(self.products_features, self.known_propensity_matrix)


    def update_pricing(self, pricing_strategy):
        nproducts, nconsumers = self.base_pricing_matrix.shape
        if pricing_strategy == 'static':
            for i in np.arange(nproducts):
                for j in np.arange(nconsumers):
                    #The purchase probability is the propensity induced probability for purchasing when the price is the average one.
                    self.dynamic_probs[i,j] = self.propensity_purchase_distribution[self.propensity_purchase_distribution.shape[0]//2 , self.propensity_matrix[i,j] ]
        elif pricing_strategy == 'adjust_by_stds':
            stds_coefs = np.array([-2, -1, 0, 1, 2])
            for i in np.arange(nproducts):
                for j in np.arange(nconsumers):
                    #Todo - try to have this more efficient (see https://stackoverflow.com/questions/41164305/numpy-dot-product-with-max-instead-of-sum)
                    purchase_distribution_vec = self.propensity_purchase_distribution[:, self.propensity_matrix[i,j]  ]
                    prices_plus_stds_vec =  self.base_pricing_matrix[i,j] + self.products_stds[i]*stds_coefs #TODO: This should be calculated offline (ndarray)
                    max_rev_index = np.argmax(purchase_distribution_vec * prices_plus_stds_vec)
                    self.dynamic_pricing_matrix[i,j] = prices_plus_stds_vec[max_rev_index]
                    self.dynamic_probs[i,j] = purchase_distribution_vec[max_rev_index]
        else:
            raise Exception("We don't support pricing strategy {}".format(pricing_strategy))


    def perform_sells(self):
        bought_indices = np.random.uniform(size = self.dynamic_pricing_matrix.shape) < self.dynamic_probs
        ebay_revenue =  self.commision_ratio*self.dynamic_pricing_matrix[bought_indices].sum()

        # print("performed sells: {}".format(self.dynamic_pricing_matrix[bought_indices]))
        print("sum sells: {}".format(self.dynamic_pricing_matrix[bought_indices].sum()))
        # print("pricing matrix:")
        # print(self.dynamic_pricing_matrix)
        # print("dynamic probs:")
        # print(self.dynamic_probs)
        return ebay_revenue

    def run_dynamics(self, time_horizon, time_step, pricing_strategies ):
        timestamps = np.arange(0, time_horizon, time_step)
        daily_sells = pd.DataFrame({}, columns=pricing_strategies, index = timestamps)
        for pricing_strategy in pricing_strategies:
            print(f"############ applying strategy {pricing_strategy} #############")
            for t in timestamps:
                print(f"############ day{t} #############")
                self.update_pricing(pricing_strategy)
                curr_revenue = self.perform_sells()
                daily_sells.loc[t, pricing_strategy] = curr_revenue
                print("#" * 40)
        daily_sells.cumsum().plot()
        plt.title("Ebay's cumulative revenues (US$)")
        plt.xlabel("Day")
        plt.ylabel("Revenue")

        plt.show(block=False)
        input("Hit Enter To Close")
        plt.close()
        return daily_sells.sum().values



def main(args):
    description = "Run pricing dynamics and calculate accumulated revenues."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', dest="products_filepath", help="The path to the Kaggle-format products file")
    parser.add_argument('-n', dest="num_consumers", help="Number of consumers to use in the simulation")
    parser.add_argument('-g', dest="commission_ratio", default=0.02, help="The commision gained by Ebay per sell, as a float between 0 and 1")
    parser.add_argument('--initial-propensity-density', dest="initial_density", default=0.35,
                        help="The fraction of the propensity matrix entries for which we initially have knowledge of. ")
    parser.add_argument('--horizon', dest="time_horizon_in_days", default=365, help="The examined overall period in days.")
    parser.add_argument('-t', dest="timestep_in_days", default=1, help="timestep in days")
    parser.add_argument('--pricing-strategy', dest="pricing_strategy", choices = EbayPricingDemo.pricing_strategies, nargs='+',
                        default="static", help="The priding strategy to use in the simulation.")




    options = parser.parse_args(args)
    products_filepath = os.path.expanduser(options.products_filepath)
    if not os.path.exists(products_filepath):
        raise Exception("Couldn't find products filepath (-i options) {}".format(products_filepath))
    num_consumers = int( options.num_consumers)
    if num_consumers < 1:
        raise Exception("I'll need at least a single consumer to work with, but I've received {}".format(num_consumers))
    commission_ratio = options.commission_ratio
    if (commission_ratio <= 0) or (commission_ratio >= 1):
        raise Exception("The commission ratio should be greater than 0 and less than 1. I've received {}".format(commission_ratio))
    time_horizon_in_days = int(options.time_horizon_in_days)
    if time_horizon_in_days < 1:
        raise Exception(
            "The time_horizon_in_days should be at least 1 day. I've received {}".format(time_horizon_in_days))
    timestep_in_days = int(options.timestep_in_days)
    if timestep_in_days < 1:
        raise Exception(
            "The timestep should be at least 1 day. I've received {}".format(timestep_in_days))
    initial_density = float(options.initial_density)
    if (initial_density <= 0.05) or (initial_density > 1):
        raise Exception(
            "The initial_density should be at 0.05 and at most 1. I've received {}".format(initial_density))
    pricing_strategies = options.pricing_strategy
    pricing_demo = EbayPricingDemo( products_filepath, num_consumers, commission_ratio, initial_density = initial_density)
    accumulated_revenues = pricing_demo.run_dynamics( time_horizon_in_days, timestep_in_days, pricing_strategies)
    print("Accumulated revenues for pricing strategy {}: {}".format(pricing_strategies, accumulated_revenues))


if __name__ == "__main__":
    main(sys.argv[1:])


