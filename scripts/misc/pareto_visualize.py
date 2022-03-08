import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from numpy.random import default_rng
from typing import List


def find_pareto_points(all_points:np.array,
                        is_decreasing:bool=True)->List[int]:
    '''Takes in a list of points, 
    one per row, returns the list of row indices
    which are pareto-frontier points. Assumes that 
    lower values on every dimension are better.'''

    # for each point see if there exists 
    # any other point which dominates it on all dimensions
    # if that is true, then it is not a pareto point
    # and vice-versa.

    # input should be two dimensional array
    assert len(all_points.shape) == 2

    pareto_inds = []

    dim = all_points.shape[1]

    for i in range(all_points.shape[0]):
        this_point = all_points[i,:]
        is_pareto = True
        for j in range(all_points.shape[0]):
            if j == i:
                continue
            other_point = all_points[j,:]
            if is_decreasing:
                diff = this_point - other_point
            else:
                diff = other_point - this_point
            if sum(diff>0) == dim:
                # other point is smaller/larger on all dimensions
                # so we have found at least one dominating point
                is_pareto = False
        if is_pareto:
            pareto_inds.append(i)

    return pareto_inds



def main():

    # generate 2D random points
    rng = default_rng()
    points = rng.standard_normal((1000,2))
    p_inds = find_pareto_points(points, is_decreasing=False)
    p_points = points[p_inds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=points[:,0], y=points[:,1], mode='markers', marker_color='blue'))
    fig.add_trace(go.Scatter(x=p_points[:,0], y=p_points[:,1], mode='markers', marker_color='red'))
    fig.show()

    # generate 3D random points
    points3 = rng.standard_normal((1000,3))
    p_inds3 = find_pareto_points(points3, is_decreasing=False)
    p_points3 = points3[p_inds3]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter3d(x=points3[:,0], y=points3[:,1], z=points3[:,2], mode='markers', marker_color='blue'))
    fig3.add_trace(go.Scatter3d(x=p_points3[:,0], y=p_points3[:,1], z=p_points3[:,2], mode='markers', marker_color='red'))
    fig3.update_layout(scene = dict(xaxis_title = "blah", yaxis_title = "blih", zaxis_title = 'bloh'))
    fig3.show()


    print('dummy')
 






if __name__ == '__main__':
    main() 
