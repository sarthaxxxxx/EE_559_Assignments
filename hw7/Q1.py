import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def generate_distributions(mean_1, mean_2, cov):
    px_s1 = stats.multivariate_normal(mean_1, cov)
    px_s2 = stats.multivariate_normal(mean_2, cov)
    x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    plt.contour(x, y, px_s1.pdf(np.dstack((x, y))), cmap = 'viridis')
    plt.contour(x, y, px_s2.pdf(np.dstack((x, y))), cmap = 'viridis')
    plt.title('Class conditional probabilities')
    plt.show()

def plotter(mean1, mean2, cov, prior1, prior2):
    x_max = 10
    y_max = 10
    x_min = -10
    y_min = -10
    
    inc = 0.05
    
    (x, y) = np.meshgrid(np.arange(x_min, x_max + inc/100, inc),
                        np.arange(y_min, y_max + inc/100, inc))
    
    xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'),
                    y.reshape(y.shape[0] * y.shape[1], 1, order='F')))
    
    prod1 = (stats.multivariate_normal(mean1, cov)).pdf(xy) * prior1
    prod2 = (stats.multivariate_normal(mean2, cov)).pdf(xy) * prior2
    
    dec_region = ((prod1 - prod2) > 0).reshape(x.shape, order = 'F')
    
    plt.imshow(dec_region, extent = [x_min, x_max, y_min, y_max], origin = 'lower')
    plt.title(f'Decision regions and boundary in 2D non-augmented space for priors = {prior1} and {prior2} respectively')
    plt.xlabel('$x_1$'), plt.ylabel('$x_2$')
    plt.show()

if __name__ == '__main__':
    mean_1 = [1, -1]
    mean_2 = [2, 3]
    covariance = [[2, 3],[3, 6.5]]
    generate_distributions(mean_1, mean_2, covariance)
    plotter(mean_1, mean_2, covariance, 0.5, 0.5)
    plotter(mean_1, mean_2, covariance, 0.1, 0.9)