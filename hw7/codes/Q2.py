import numpy as np
import matplotlib.pyplot as plt


def prob(idx, h, d, pts):
    if idx >= (pts - h) and idx <= (pts + h):
        return d
    else:
        return 0

def kde(x, h):
    d = 1/(2*h*len(x))
    y = []
    for idx in np.arange(-3, 10, 0.01):
        probability = 0.0
        for pts in x:
            probability += prob(idx, h, d, pts)
        y.append(probability)
    return np.array(y)


def kde_estimate(data, h):
    priors = {k: len(v) for k, v in data.items()}
    prod_1 = priors['S1']*kde(data['S1'], h)
    prod_2 = priors['S2']*kde(data['S2'], h)
    density = np.zeros(np.arange(-3, 10, 0.01).shape)
    density[np.where((prod_1 == prod_2))] = -1
    density[np.where((prod_1 > prod_2))] = 1
    density[np.where((prod_1 < prod_2))] = 0
    return density
    
        
if __name__ == '__main__':
    x1 = [0, 0.4, 0.9, 1, 6, 8]
    x2 = [2.0, 4.0, 4.5, 5.0, 5.8, 6.7, 7.0]
    x_axis = np.arange(-3, 10, 0.01)
    for h in [0.5, 1, 2]:
        density = kde_estimate({'S1': x1, 'S2': x2}, h)
        plt.plot(x_axis, kde(x1, h), 'r')
        plt.title(f'KDE estimates for $S_1$ w/ h = {str(h)}'), plt.xlabel('x'), plt.ylabel('P(x|$S_1$)')
        plt.show()
        plt.plot(x_axis, kde(x2, h), 'b')
        plt.title(f'KDE estimates for $S_2$ w/ h = {str(h)}'), plt.xlabel('x'), plt.ylabel('P(x|$S_2$)')
        plt.show()
        plt.plot(x_axis[density == 1], np.zeros(x_axis[density == 1].shape), 'ro', label = '$S_1$')
        plt.plot(x_axis[density == 0], np.zeros(x_axis[density == 0].shape), 'bo', label = '$S_2$')
        plt.legend()
        plt.title(f'Decision boundary and region for h = {str(h)}')
        plt.show()