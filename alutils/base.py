# Typing
from __future__ import annotations
from typing import Callable, Any, Tuple, NewType, Literal

# Numpy
import numpy as np
from numpy.typing import NDArray

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

def dehomogenized(vectors: NDArray) -> NDArray:
    """
    Dehomogenize vectors stored in matrix (..., d + 1), scaling down by the last
    element of each vector and returning a d-dimensional homogeneous vectors in
    matrix of size (..., d).

    Inputs
    - vectors: `NDArray(..., d + 1)` homogeneous input vectors

    Returns
    - dehomogenized_vectors: `NDArray(..., d)` dehomogenized vectors
    """
    return vectors[..., :-1] / vectors[..., -1:]

def homogenized(vectors: NDArray, fill_value: Any = 1) -> NDArray:
    """
    Homogenize d-dimensional vectors stored in matrix (..., d), returning
    homogeneous vectors in matrix of size (..., d + 1) by appending a `1` to the
    last dimension. If specified, the `fill_value` will be used instead of `1`.

    Inputs
    - vectors: `NDArray(..., d)` input vectors

    Optional Inputs
    - fill_value: `Any` value to fill the last dimension of the vectors. Default
                   is `1`.

    Returns
    - homogenized_vectors: `NDArray(..., d + 1)` homogenized vectors
    """
    return np.concatenate(
        (vectors, np.full(vectors.shape[:-1],
                          fill_value,
                          dtype=vectors.dtype)[..., np.newaxis]),
        axis=-1,
        dtype=vectors.dtype
    )


def normalized(x: NDArray, axis: int = -1, norm: Literal['L1', 'L2'] = 'L2') \
    -> NDArray:
    """
    Normalize an array along the provided axis. By default the normalization is
    perfomed along the last axis.

    Inputs
    - x: `NDArray` input array

    Optional Inputs
    - axis: `int` axis along which to normalize the array. Default is `-1`.
    - norm: `Literal['L1', 'L2']` type of normalization to perform. Default is
            `L2`.

    Returns
    - normalized_x: `NDArray` normalized array
    """
    if np.any(np.linalg.norm(x, axis=axis, keepdims=True) == 0):
        raise ValueError("Error in vector normalization as norm is zero.")
    if norm == 'L1':
        return x / np.linalg.norm(x, ord=1, axis=axis, keepdims=True)
    elif norm == 'L2':
        return x / np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    else:
        raise TypeError(f"Invalid normalization argument '{norm}.' " +
                        f"Choose 'L1' or 'L2'.")


Model = NewType("Model", Any)
def ransac(
    model_fct: Callable[[NDArray], Model],
    error_fct: Callable[[NDArray, Model], NDArray],
    data: NDArray,
    n_datapoints: int,
    threshold: float,
    outlier_ratio: float = None,
    n_iterations: int = None
    ) -> Tuple[Any, NDArray]:
    """
    RANSAC method finding the best model while rejecting outlier data points.

    Inputs
    - model_fct:    `Callable[[NDArray], Model]` function that finds a model
                     given some data, i.e `model_fct(x: NDArray) -> Model`.
    - error_fct:     `Callable[[NDArray, Model], NDArray] function that computes
                     the error for every datapoint given a model,
                     i.e. `error_fct(x: NDArray, model: Model) -> NDArray`.
    - data:          `NDArray(N, ...)` the N data points used in the problem.
    - n_datapoints:  `int` the minimum number of data points needed to estimate
                     the model.
    - threshold:     `float` threshold value that determines if the model fits a
                     datapoint well or not.
    - outlier_ratio: `float` the estimated outlier ratio. If unspecified or
                     `None`, the `n_iterations is employed.
    - n_iterations:  `int` the number of iterations that will be performed if
                     the `outlier_ratio` is unspecified or `None`. If `None`, an
                     adaptive RANSAC method will be employed.
    Returns
    - best_model:   `Model` the model that fits the data the best
    - inlier_mask:  `NDArray(N, )` boolean array mask for the determined inliers
    """

    if len(data) < n_datapoints:
        logger.error("Fewer datapoints than the size of the subsets needed " +
                      "to estimate the model.")
        raise ValueError("Fewer datapoints than the size of the subsets " +
                         "needed to estimate the model.")

    if outlier_ratio != None:
        prob_success = 0.99
        n_iterations = int(np.ceil(
            np.log(1 - prob_success) /\
            np.log(1 - (1 - outlier_ratio)**n_datapoints)
        ))
    elif n_iterations == None:
        logger.error("Adapative RANSAC is not implemented.")
        raise NotImplementedError("Adapative RANSAC is not implemented.")

    max_n_inliers = 0
    best_model, best_inlier_mask = None, None
    for _ in range(n_iterations):

        # Pick a subset
        subset_idxs = np.random.choice(len(data), size=n_datapoints,
                                       replace=False)
        data_sub = data[subset_idxs]

        # Determine the model, get the error and set the inlier mask
        model = model_fct(data_sub)     # Model
        error = error_fct(data, model)  # (N, )
        inlier_mask = error < threshold # (N, )

        # Count number of inliers and update the best model if necessary
        n_inliers = np.sum(inlier_mask)
        if n_inliers > max_n_inliers:
            max_n_inliers = n_inliers
            best_model, best_inlier_mask = model, inlier_mask

    return best_model, best_inlier_mask
