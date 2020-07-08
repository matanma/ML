#!C:\Users\Matan\anaconda3\python.exe
##!/usr/bin/env python
import pandas as pd
import argparse
import sys, os
import sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
from pandas_summary import DataFrameSummary
import pandas_profiling as pp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import random
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import xgboost as xgb
# %matplotlib inline



class EbayPricingDemo():
    pricing_strategies = ['static', 'adjust_by_stds']

    def __init__(self, products_filepath, num_consumers, commision_ratio, propensity_range_max = 5):
        self.product_names, self.avg_product_prices, self.products_stds = self.load_product_prices(products_filepath)
        self.commision_ratio = commision_ratio
        self.propensity_purchase_distribution = self.init_propensity_purchase_distribution()
        self.base_pricing_matrix = self.init_pricing_matrix(self.avg_product_prices, num_consumers)
        self.dynamic_pricing_matrix = np.copy(self.base_pricing_matrix)
        # self.gains_array = self.init_gains_array(self.products_stds, self.propensity_purchase_distribution)
        #Initially, the propensity of any consumer to by any product is neutral.
        self.propensity_matrix = np.ones(self.base_pricing_matrix.shape, dtype=int)*(propensity_range_max//2 + 1)
        self.dynamic_probs = np.zeros(self.base_pricing_matrix.shape)



    def  load_product_prices(self, products_filepath="mercari-price-suggestion-challenge/train_small.csv"):
        raw_data_df = pd.read_csv(products_filepath, sep=',' )
        return raw_data_df['name'].values, raw_data_df['price'], np.ones(len(raw_data_df)) #!!! TODO: extract stds from actual prices.

    #Returns a matrix in which column j is the probability distribution corresponding to propencity j (in ascending order)
    def init_propensity_purchase_distribution(kernel = np.arange(5,0,-1)/100, generation_method = 'exp'):
        kernel = np.arange(5, 0, -1) / 100
        return kernel[:, np.newaxis].dot(np.arange(1, 6)[np.newaxis, :])

    #Initially - have all consumer prices identical
    def init_pricing_matrix(self, avg_product_prices, num_consumers):
        return avg_product_prices[:, np.newaxis].dot(np.ones(num_consumers)[ np.newaxis, :])

    # def init_gains_array(self, products_stds, propensity_purchase_distribution):
    #     gains_array = np.zeros((propensity_purchase_distribution.shape[0], *products_stds.shape))
    #     for i in [-2, -1 ,0, 1, 2]:
    #         stds_mat =


    def update_pricing(self, pricing_strategy):
        nproducts, nconsumers = self.base_pricing_matrix.shape
        if pricing_strategy == 'static':
            for i in np.arange(nproducts):
                for j in np.arange(nconsumers):
                    #The purchase probability is the propensity induced probability for purchasing when the price is the average one.
                    self.dynamic_probs[i,j] = self.propensity_purchase_distribution[self.propensity_purchase_distribution.shape[0]//2, self.propensity_matrix[i,j] -1 ]
        elif pricing_strategy == 'adjust_by_stds':
            stds_coefs = np.array([-2, -1, 0, 1, 2])

            for i in np.arange(nproducts):
                for j in np.arange(nconsumers):
                    #Todo - try to have this more efficient (see https://stackoverflow.com/questions/41164305/numpy-dot-product-with-max-instead-of-sum)
                    purchase_distribution_vec = self.propensity_purchase_distribution[:, self.propensity_matrix[i,j] -1 ]
                    prices_plus_stds_vec =  self.base_pricing_matrix[i,j]+self.products_stds[i]*stds_coefs #TODO: This should be calculated offline (ndarray)
                    max_rev_index = np.argmax(purchase_distribution_vec * prices_plus_stds_vec)
                    self.dynamic_pricing_matrix[i,j] = prices_plus_stds_vec[max_rev_index]
                    self.dynamic_probs[i,j] = purchase_distribution_vec[max_rev_index]
        else:
            raise Exception("We don't support pricing strategy {}".format(pricing_strategy))


    def perform_sells(self):
        bought_indices = np.random.uniform(size = self.dynamic_pricing_matrix.shape) < self.dynamic_probs
        ebay_revenue =  self.commision_ratio*self.dynamic_pricing_matrix[bought_indices].sum()

        print("performed sells: {}".format(self.dynamic_pricing_matrix[bought_indices]))
        print("pricing matrix:")
        print(self.dynamic_pricing_matrix)
        print("dynamic probs:")
        print(self.dynamic_probs)
        return ebay_revenue

    def run_dynamics(self, time_horizon, time_step, pricing_strategy ):
        total_revenue = 0
        for t in np.arange(0, time_horizon, time_step):
            print(f"############ day{t} #############")
            self.update_pricing(pricing_strategy)
            curr_revenue = self.perform_sells()
            total_revenue += curr_revenue
            print("#" * 40)
        return total_revenue





def main(args):
    description = "Run pricing dynamics and calculate accumulated revenues."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', dest="products_filepath", help="The path to the Kaggle-format products file")
    parser.add_argument('-n', dest="num_consumers", help="Number of consumers to use in the simulation")
    parser.add_argument('-g', dest="commission_ratio", default=0.02, help="The commision gained by Ebay per sell, as a float between 0 and 1")
    parser.add_argument('--horizon', dest="time_horizon_in_days", default=365, help="The examined overall period in days.")
    parser.add_argument('-t', dest="timestep_in_days", default=1, help="timestep in days")
    parser.add_argument('--pricing-strategy', dest="pricing_strategy", choices = EbayPricingDemo.pricing_strategies,
                        default="static", help="The priding strategy to use in the simulation.")




    options = parser.parse_args(args)
    products_filepath =  os.path.expanduser(options.products_filepath)
    if not os.path.exists(products_filepath):
        raise Exception("Couldn't find products filepath (-i options) {}".format(products_filepath))
    num_consumers = int( options.num_consumers)
    if num_consumers < 1:
        raise Exception("I'll need at least a single consumer to work with, but I've received {}".format(num_consumers))
    commission_ratio = options.commission_ratio
    if (commission_ratio <= 0) or (commission_ratio >= 1):
        raise Exception("The commission ratio should be greater than 0 and less than 1. I've received {}".format(commission_ratio))

    time_horizon_in_days = int(options.time_horizon_in_days)
    timestep_in_days = int(options.timestep_in_days)
    if timestep_in_days < 1:
        raise Exception(
            "The timestep should be at least 1 day. I've received {}".format(timestep_in_days))

    pricing_strategy = options.pricing_strategy

    pricing_demo = EbayPricingDemo( products_filepath, num_consumers, commission_ratio)
    accumulated_revenues = pricing_demo.run_dynamics( time_horizon_in_days, timestep_in_days, pricing_strategy)
    print("Accumulated revenues for pricing strategy {}: {}".format(pricing_strategy, accumulated_revenues))


if __name__ == "__main__":
    main(sys.argv[1:])


##TODO: beautify code to resemble Java.