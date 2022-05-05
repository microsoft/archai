import numpy as np

from archai.nas.nas_utils.pareto_frontier import compute_crowding_distance





def main():
    # generate some random numbers from a mixture of gaussians
    cov = np.diag([1, 1])
    size = 100
    cluster_1 = np.random.Generator.multivariate_normal([1, 1], cov, size=size)
    cluster_2 = np.random.Generator.multivariate_normal([2, 1], cov, size=size)


    # plot them

    # compute the crowding distance

    # sort by descending order of crowding distance

    # pick the order that is 







if __name__ == '__main__':
    main()