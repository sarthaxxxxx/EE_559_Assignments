import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOTDIR = '/home/sarthak/Desktop/spring_22/EE_559/midterm/data/Pr1_dataset2/'
TRAIN_FILE = 'train_2.csv'
VAL_FILE = 'val_2.csv'
TEST_FILE = 'test_2.csv'

class Kernel:
    def __init__(self, 
                root, 
                train, 
                test,
                val,
                dataset):
        self.train_file = os.path.join(root, train)
        self.val_file = os.path.join(root, val)
        self.test_file = os.path.join(root, test)
        self.curr_dataset = dataset
        self.labels = [1.0, 2.0] if self.curr_dataset == 'dataset1' else [0.0, 1.0]

    def _get_features(self, path):
        df = pd.read_csv(path, header = None)
        x, y = df.iloc[:, :-1].values.astype(float),\
               df.iloc[:, -1].values.astype(float)
        return x, y

    def _get_data(self):
        self.train_x, self.train_y = self._get_features(self.train_file)
        self.test_x, self.test_y = self._get_features(self.test_file)
        self.val_x, self.val_y = self._get_features(self.val_file)
        return self.train_x, self.train_y, self.val_x, self.val_y, \
               self.test_x, self.test_y

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

    @staticmethod
    def _rbf(x1, x2, gamma = 0.01):
        L2dist = np.sqrt(np.sum((x1 - x2)**2))
        return np.exp(-gamma * L2dist)

    @staticmethod
    def _linear_kernel(x1, x2):
        return np.dot(x1.T, x2)

    def _fit_term1(self, x, y, gamma = None):
        for curr_class in self.labels:
            curr_x = x[y == curr_class]
            outer_prod_store_1 = 0.0
            for idx in range(len(curr_x)):
                prod_store = 0.0
                for jdx in range(len(curr_x)):
                    if gamma:
                        prod_store += self._rbf(curr_x[idx], curr_x[jdx], gamma)
                    else:
                        prod_store += self._linear_kernel(curr_x[idx], curr_x[jdx])
                outer_prod_store_1 += (prod_store)
            if curr_class == self.labels[0]:
                term_1_1 = -outer_prod_store_1 / (len(x[y == self.labels[0]]) ** 2) 
            else:
                term_1_2 = -outer_prod_store_1 / (len(x[y == self.labels[1]]) ** 2) 
        return [term_1_1, term_1_2]

    def _fit_term2(self, curr_data, x, y, gamma = None):
        for curr_class in self.labels:
            curr_x = x[y == curr_class]
            outer_prod_store_2 = 0.0
            for idx in range(len(curr_x)):
                if gamma:
                    outer_prod_store_2 += self._rbf(curr_x[idx], curr_data, gamma)
                else:
                    outer_prod_store_2 += self._linear_kernel(curr_x[idx], curr_data)
            if curr_class == self.labels[0]:
                term_2_1 = (2 / len(x[y == self.labels[0]])) * outer_prod_store_2
            else:
                term_2_2 = (2 / len(x[y == self.labels[1]])) * outer_prod_store_2  
        return [term_2_1, term_2_2]

    def _g_x(self, kernel_term1, x, gamma = None):
        kernel_term2 = self._fit_term2(x, self.train_x, self.train_y, gamma)
        g_x_1 = np.dot(x.T, x) - (kernel_term1[0] + kernel_term2[0])
        g_x_2 = np.dot(x.T, x) - (kernel_term1[1] + kernel_term2[1])
        return np.array([g_x_1, g_x_2])

    def _optimal_gamma(self, x, y):
        val_k_error = []
        for k in np.linspace(-2, 2, 100):
            gamma = (10**k)
            error = 0.0
            kernel_term1 = self._fit_term1(self.train_x, self.train_y, gamma)
            for idx in range(len(x)):
                g_x = self._g_x(kernel_term1, x[idx], gamma)
                if g_x[0] > g_x[1] and y[idx] == self.labels[0]:
                    error += 1
                if g_x[0] < g_x[1] and y[idx] == self.labels[1]:
                    error += 1
            val_k_error.append((k , error / len(x)))
        return val_k_error

    def _classify(self, x, y, opt_gamma = None):
        predictions, error= [], 0.0
        kernel_term1 = self._fit_term1(self.train_x, self.train_y, opt_gamma)
        for idx in range(len(x)):
            g_x = self._g_x(kernel_term1, x[idx], opt_gamma)
            if g_x[0] > g_x[1] and y[idx] == self.labels[0]:
                error += 1
            if g_x[0] < g_x[1] and y[idx] == self.labels[1]:
                error += 1            
            preds = np.argmin(g_x, axis = 0 ) + 1.0 if self.curr_dataset == 'dataset1'\
                    else np.argmin(g_x, axis = 0)
            predictions.append(preds)
        return error / len(x), predictions

    def _plot_k_errors(self, dataset_k_error):
        plt.plot([x[0] for x in dataset_k_error], [x[1] for x in dataset_k_error])
        plt.ylabel('Validation error')
        plt.xlabel('k (gamma = 10 ^ k)')
        plt.tick_params(axis = "x")
        plt.tick_params(axis = "y")
        plt.title(f'Validation error vs k (gamma = 10 ^ k) for {self.curr_dataset}')
        plt.tight_layout()
        plt.show()
    
    def _plotter(self, x_data, y_data, opt_gamma = None, kernel = None):
        # Find max and min values of both the features         
        max_x, min_x = np.ceil(max(x_data[:, 0])) + 1, np.floor(min(x_data[:, 0])) - 1
        max_y, min_y = np.ceil(max(x_data[:, 1])) + 1, np.floor(min(x_data[:, 1])) - 1

        inc = 0.05
        # Calculate the range of values for x and y
        range_x = np.arange(min_x, max_x + inc/100, inc)
        range_y = np.arange(min_y, max_y + inc/100, inc)

        # Create a mesh grid of values
        xx, yy = np.meshgrid(range_x, range_y)

        # Predict the values on the mesh grid
        _, grid_preds = np.array(self._classify(np.c_[xx.ravel(), \
                     yy.ravel()], yy.ravel(), opt_gamma))
        preds = np.array(grid_preds).reshape(xx.shape) # matrix of classifications

        # Obtain data points of both the features
        x_1, x_2 = x_data[:,0], x_data[:,1]
        
        _, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 200)

        # Plot the filled contours (decision regions)
        ax.contourf(xx, yy, preds, alpha = 0.25)
        # Plot the decision boundary.
        ax.contour(xx, yy, preds, colors = 'k', linewidths = 0.8, linestyles = 'dashed')
        # Plot the data points (scatter plot)
        ax.scatter(x_1, x_2, c = y_data, edgecolors = 'k')

        ax.grid(False)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        if opt_gamma:
            ax.set_title(f'Feature space w/ decision boundary and regions of {self.curr_dataset} using the {kernel} kernel with gamma = {opt_gamma}')
        else:
            ax.set_title(f'Feature space w/ decision boundary and regions of {self.curr_dataset} using the {kernel} kernel')   
        # plt.savefig(f'{self.curr_dataset}_{mode}.png')
        plt.tight_layout()
        plt.show()

    def _runner(self):
        print(f'*** RUNNING SCRIPTS FOR {str(self.curr_dataset)} ***')
        self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = self._get_data()

        # 1(d)
        k_errors = self._optimal_gamma(self.val_x, self.val_y)

        # 1(e)
        optimal_k = k_errors[np.argmin([x[1] for x in k_errors])][0]
        print(f'Optimal gamma = {10 ** optimal_k}')
        self._plot_k_errors(k_errors)
        
        # 1(f)
        rbf_error, _ = self._classify(self.test_x, self.test_y, 10 ** optimal_k)
        linear_error, _ = self._classify(self.test_x, self.test_y)
        print(f'Test set error using the RBF and linear kernels = {rbf_error}, {linear_error}, respectively')

        # 1(g)
        print(f"*** Plotting the decision regions for the linear kernel ***")
        self._plotter(self.train_x, self.train_y, kernel = 'linear')

        # 1(h)
        print(f"*** Plotting the decision regions for the RBF kernel ***")
        self._plotter(self.train_x, self.train_y, 10 ** optimal_k, kernel = 'rbf')

        # 1(i)
        for new_gamma_step in [0.01, 0.1, 0.3, 3, 10, 100]:
            print(f"*** Plotting the decision regions for the RBF kernel with gamma = {new_gamma_step * (10**optimal_k)} ***")
            gamma = new_gamma_step * (10**optimal_k)
            self._plotter(self.train_x, self.train_y, gamma, kernel = 'rbf')

        print(f'*** DONE RUNNING SCRIPTS FOR {str(self.curr_dataset)} ***')
       

if __name__ == '__main__':
    current_dataset = (ROOTDIR.split('/')[-2]).split('_')[1]
    hw = Kernel(ROOTDIR, TRAIN_FILE, TEST_FILE, VAL_FILE, current_dataset)
    hw._runner()
