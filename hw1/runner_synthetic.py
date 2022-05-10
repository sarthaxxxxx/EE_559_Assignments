##################################################
# Author:   Sarthak Kumar Maharana
# Email:    maharana@usc.edu
# Date:     01/29/2022
# Course:   EE 559
# Project:  Homework 1
# Instructor: Prof. B Keith Jenkins
##################################################

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from data.plotDecBoundaries import plotDecBoundaries


ROOTDIR = '~/Desktop/spring_22/EE_559/hw1/codes_2/data/'
train_file = 'synthetic2_train.csv'
test_file = 'synthetic2_test.csv'


class Homework1_ab:
    """
    Nearest Means Classifier
    """
    def __init__(self, 
                train_file,
                test_file,
                mode
                ):
        self.train_file = train_file
        self.test_file = test_file
        self.mode = mode


    def _read_csv_get_features(self, 
                              filename
                              ):
        """ Read csv files and return features, labels, and dataframe. """
        df = pd.read_csv(os.path.join(ROOTDIR, filename), 
                        header = None
                        )
        x, y = df.iloc[:, : -1].values, df.iloc[:, -1].values
        return x, y, df


    def _load(self):
        """ Utility function to load data. """
        self.train_x, self.train_y, _ = self._read_csv_get_features(self.train_file)
        self.test_x, self.test_y, _ = self._read_csv_get_features(self.test_file)
        return self.train_x, self.train_y, self.test_x, self.test_y


    def _plot_data(self):
        """ Plot for visualization. """
        plt.scatter(self.train_x[:, 0], 
                    self.train_x[:, 1], 
                    c = self.train_y, 
                    s = 50, 
                    cmap = 'viridis'
                    )
        plt.show()


    @staticmethod
    def L2distance(x, y):
        """ Compute L2 (Euclidean) distance between two vectors. """
        return np.sqrt(np.sum((x - y)**2))


    def _sample_mean(self, 
                    data_x, 
                    data_y
                    ):
        """ Compute the sample mean for the data. """
        c_1_mean = np.mean(data_x[data_y == 1], axis = 0) # mean of class 1
        c_2_mean = np.mean(data_x[data_y == 2], axis = 0) # mean of class 2 
        if len(np.unique(data_y)) >= 3:
            c_3_mean = np.mean(data_x[data_y == 3], axis = 0) # mean of class 3, if it exists
            self.sample_mean = np.vstack((c_1_mean, c_2_mean, c_3_mean))
            return self.sample_mean
        self.sample_mean = np.vstack((c_1_mean, c_2_mean))
        return self.sample_mean


    def _error_rate(self, 
                    data_x, 
                    data_y, 
                    mean
                    ):
        """ Compute error rate, based on the data and sample mean that are passed. """
        self.error = 0.0
        for idx in range(len(data_x)):
            e_feat_1 = self.L2distance(data_x[idx], 
                                      mean[0]
                                      ) # label 1 error
            e_feat_2 = self.L2distance(data_x[idx], 
                                      mean[1]
                                      ) # label 2 error
            if e_feat_1 > e_feat_2 and data_y[idx] == 1:
                self.error += 1
            if e_feat_1 < e_feat_2 and data_y[idx] == 2:
                self.error += 1
        return round(self.error / len(data_x), 3) # total error rate


    def _solver(self):
        """ Solver for the problem. """
        self.train_x, self.train_y, self.test_x, self.test_y = self._load()  # load data
        means = self._sample_mean(self.train_x, self.train_y) # train the "classifier" by computing the sample mean.
        plotDecBoundaries(self.train_x, self.train_y, means) # plot decision boundaries and regions of the training data
        plotDecBoundaries(self.test_x, self.test_y, means) # plot decision boundaries and regions of the test data
        return self._error_rate(self.test_x, self.test_y, means) if self.mode == 'test' \
        else self._error_rate(self.train_x, self.train_y, means) # return error rate on training data or test data based on user config. 



if __name__ == '__main__':
    """
    Execute the .py as follows:
    python3 [name of the file].py --mode train  [EXAMPLE]    // to test on training data

    Available "mode" options: train, test [default: train (Train the classifier and plot the decision boundaries)]
    """
    parser = argparse.ArgumentParser(description = 'CLI args for running the script.')
    parser.add_argument('--mode', type = str, default = 'train',
                    help = 'evaluate the classifier on train or test.')
    args = parser.parse_args()
    hw1 = Homework1_ab(train_file, test_file, mode = args.mode)
    print(f'Error rate on the {hw1.mode} set:\
    {hw1._solver()}'
    )

