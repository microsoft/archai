import numpy as np

import plotly.graph_objects as go

from archai.nas.nas_utils.pareto_frontier import compute_crowding_distance





def main():
    # generate some random numbers from a mixture of gaussians
    cov = np.diag([1, 1])
    size = 1000
    cluster_1 = np.random.default_rng().multivariate_normal([1, 1], cov, size=size)
    cluster_2 = np.random.default_rng().multivariate_normal([5, 1], cov, size=size)
    all_points = np.vstack((cluster_1, cluster_2))

    # plot them
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=all_points[:,0], y=all_points[:,1], 
                    mode='markers', 
                    name='random points'))

    # compute the crowding distance for each point
    c_dists = compute_crowding_distance(all_points)

    # keep points which have high crowding distance
    num_keep = 300
    ids = np.argsort(-c_dists, axis=None) # descending sort
    ids_to_keep = ids[:num_keep]
    top_uncrowded = all_points[ids_to_keep,:]
    
    # plot top k uncrowded
    fig.add_trace(go.Scatter(x=top_uncrowded[:,0], y=top_uncrowded[:,1], 
                            mode='markers', 
                            name=f'Top {num_keep} uncrowded',
                            marker_color='red'))

    fig.write_html("crowding_test.html")






if __name__ == '__main__':
    main()