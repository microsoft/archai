from typing import List, Sequence
import numpy as np
import copy


def prune(model_matrix:np.ndarray, vertex_ops:List[str]):
    """Prune the extraneous parts of the graph.

    General procedure:
        1) Remove parts of graph not connected to input.
        2) Remove parts of graph not connected to output.
        3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """

    shape = np.shape(model_matrix)
    num_vertices = shape[0]
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError('model_matrix must be square')
    if shape[0] != len(vertex_ops):
      raise ValueError('length of vertex_ops must match model_matrix dimensions')
    if not _is_upper_triangular(model_matrix):
      raise ValueError('model_matrix must be upper triangular')

    # DFS forward from input
    visited_from_input = set([0])
    frontier = [0]
    while frontier:
        top = frontier.pop()
        for v in range(top + 1, num_vertices):
            if model_matrix[top, v] and v not in visited_from_input:
                visited_from_input.add(v)
                frontier.append(v)

    # DFS backward from output
    visited_from_output = set([num_vertices - 1])
    frontier = [num_vertices - 1]
    while frontier:
        top = frontier.pop()
        for v in range(0, top):
            if model_matrix[v, top] and v not in visited_from_output:
                visited_from_output.add(v)
                frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 2 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 2:
        raise RuntimeError(f'Cannot build model because there are {extraneous} vertices which are larger than total vertices {num_vertices}-2')

    model_matrix = copy.deepcopy(model_matrix)
    model_matrix = np.delete(model_matrix, list(extraneous), axis=0)
    model_matrix = np.delete(model_matrix, list(extraneous), axis=1)

    vertex_ops = copy.deepcopy(vertex_ops)
    for index in sorted(extraneous, reverse=True):
        del vertex_ops[index]

    return model_matrix, vertex_ops


def _is_upper_triangular(model_matrix:np.ndarray):
    # TODO: just use np.allclose(mat, np.triu(mat))
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(model_matrix)[0]):
        for dst in range(0, src + 1):
            if model_matrix[src, dst] != 0:
                return False

    return True