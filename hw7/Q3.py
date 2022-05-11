import numpy as np


class KNNRegressor:
    def __init__(self, x, y, true_target, q = 'a', nbd = 4):
        self.x = x
        self.y = y
        self.nbd = nbd
        self.q = q
        self.true = true_target

    def euclidean_dist(self, x1, x2):
        return np.sqrt(np.sum(x1 - x2) ** 2)
    
    def quadratic_poly(self, x1, x2, **kwargs):
        return 1 - (self.euclidean_dist(x1, x2) / kwargs['dmax'])
        
    
    def evaluate(self, test):
        distances = [self.euclidean_dist(test, self.x[pts]) for pts in range(len(self.x))]
        close_pts = np.argsort(distances)
        dmax = distances[close_pts[self.nbd]]
        w = [1 if self.q == 'a' else self.quadratic_poly(test, idx, dmax = dmax)\
             for idx in self.x[close_pts[:self.nbd]]]
        # op = np.sum(self.y[close_pts[:self.nbd]]) / np.sum(w)
        op = np.average(self.y[close_pts[:self.nbd]], weights = w)
        mse = (self.true(test) - op)**2
        
        return op, mse
        


if __name__ == '__main__':
    x = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    y = np.array([0.81, 0.49, 0.25, 0.09, 0.01, 0.01, 0.09, 0.25, 0.49, 0.81])
    knn = KNNRegressor(x, y, lambda x:x**2)
    test = np.array([0, 0.4])
    total_mse = 0.0
    for pts in test:
        op, mse_error = knn.evaluate(pts)
        print(f"Output for x_test {pts} is {np.round(op, 4)}")
        total_mse += mse_error
    print(f"Total mse for (a): {np.round(total_mse / len(test), 4)}")
    
    
    total_mse = 0.0
    knn = KNNRegressor(x, y, lambda x:x**2, q = 'b')
    for pts in test:
        op, mse_error = knn.evaluate(pts)
        print(f"Output for x_test {pts} is {np.round(op, 4)}")
        total_mse += mse_error
    print(f"Total mse for (a): {np.round(total_mse / len(test), 4)}")
    
    