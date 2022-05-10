import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# from data.h5w7_pr3_python_files.h5_w7_helper_functions import *

np.random.seed(0)

ROOTDIR = './data/h5w7_pr3_python_files/'
FILENAME = 'h5w7_data.csv'


class NonLinearMapping:
    def __init__(self, filepath):
        self.filepath = filepath

    def _get_features(self, path):
        """ Obtain the feature matrix and the corresponding labels.
        
        Parameters
        ----------
        path : str
            Path to the dataset.
        
        Returns
        -------
        x : ndarray
            Feature matrix of dimension N x D.
        y : ndarray
            Labels of dimension N x 1.
        """
        df = pd.read_csv(path, header = None)[1 : ]
        x, y = df.iloc[:, :-1].values.astype(float),\
               df.iloc[:, -1].values.astype(float)
        return x, y
    
    def _plot_feature_space(self, x, y):
        """
        Visualize the original feature space.

        Parameters
        ----------
        x : ndarray
            Feature matrix of dimension N x D.
        y : ndarray
            Labels of dimension N x 1.

        Returns
        -------
        None
        """
        class_names = ['Class ' + str(int(c)) for c in np.unique(y)]
        classes = np.unique(y)
        style = ['rx', 'bo']
        for idx in range(len(classes)):
            plt.plot(x[y == classes[idx], 0],
                    x[y == classes[idx], 1],
                    style[idx],
                    label = class_names[idx])
        plt.legend(loc = 'upper right')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Feature space (non-augmented)')
        plt.tight_layout()
        plt.show()
        

    def _feature_space_expansion(self, x):
        """
        Expand the feature space using the feature vectors.
        [x1, x2] -> [x1, x2, x1x2, x1x1, x2x2]

        Parameters
        ----------
        x : ndarray
            Feature matrix of dimension N x D.
        
        Returns
        -------
        expanded_x : ndarray
            Expanded feature matrix of dimension N x D + 3.
        """

        x1x2 = np.multiply(x[:, 0], x[:, 1]).reshape(-1, 1)
        x1x1 = np.multiply(x[:, 0], x[:, 0]).reshape(-1, 1)
        x2x2 = np.multiply(x[:, 1], x[:, 1]).reshape(-1, 1)
        expanded_x = np.hstack((x, x1x2, x1x1, x2x2))
        return expanded_x

    def _runner(self):
        """
        Runner module for the class.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # 3(a)
        # Load the data
        x, y = self._get_features(self.filepath)
        # Plot the feature space (non-augmented)
        self._plot_feature_space(x, y)


        # 3(b)
        # Fit the perceptron model
        model = Perceptron(fit_intercept = False).fit(x, y)
        # Obtain weights
        perceptron_weights = model.coef_[0]
        print(f'Classification score on the feature space: {model.score(x, y)}')

        # 3(c)
        # Plot the decision boundary
        plot_perceptron_boundary(
            x, y, perceptron_weights, linear_decision_function
        )   


        # 3(d)
        # Quadratic feature expansion
        expanded_x = self._feature_space_expansion(x)
        # Fit the perceptron model
        model.fit(expanded_x, y)
        # Obtain weights
        weights_newspace = model.coef_[0]
        # Classication accuracy
        score_newspace = model.score(expanded_x, y)
        print('Weight vector in the new feature space:', weights_newspace)
        print(f'Classification score on the expanded feature space: {score_newspace}')
        
        # 3(e)
        # Obtain absolute weights of the expanded feature space
        abs_weights = np.abs(weights_newspace)
        # Get locations of the sorted absolute weights
        indices_locs = np.flip(np.argsort(abs_weights))
        # Consider relevant features according to the weights 
        relevant_feat_1, relevant_feat_2 = expanded_x[:, indices_locs[0]].reshape(-1, 1), \
                                           expanded_x[:, indices_locs[1]].reshape(-1, 1)
        # New feature space (2 dims) of relevant features
        phi_X_best = np.hstack((relevant_feat_1, relevant_feat_2))
        # Weights of the relevant features
        weights_best_2 = np.array([abs_weights[indices_locs[0]], abs_weights[indices_locs[1]]])
        # Plot the decision boundary
        # plot_perceptron_boundary(
        #     phi_X_best, y, weights_best_2, linear_decision_function
        # )

        # 3(f)
        # Nonlinear mapping (decision boundary)
        # plot_perceptron_boundary(
        #     x, y, perceptron_weights, nonlinear_decision_function
        # )
        

        
if __name__ == '__main__':
    filepath = os.path.join(ROOTDIR, FILENAME)
    hw = NonLinearMapping(filepath)
    hw._runner()