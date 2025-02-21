# Typing
from typing import NewType, Tuple, Optional

# Numpy
import numpy as np
from numpy.typing import NDArray

# Scipy
try:
    import scipy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Types
State = NewType("State", NDArray)
Covariance = NewType("Covariance", NDArray)

def prior_update(x_m: State, P_m: Covariance,
                 A: NDArray, b: NDArray, Q: Covariance) \
                   -> Tuple[State, Covariance]:
   """
   Prior update of the Kalman Filter, also referred to as the prediction step.

   The dynamics are described by:
     x_k = A * x_{k-1} + b + v,
     where v is a zero-mean Gaussian RV with covariance Q.

   Inputs
   - x_m: `(n, 1)` or `(n,)` a posteriori estimate of the state,
          i.e. x_{k-1|k-1}.
   - P_m: `(n, n)` a posteriori covariance matrix of the state,
           i.e. P_{k-1|k-1}.
   - A:   `(n, n)` state transition matrix.
   - b:   `(n, 1)` or `(n,)` bias vector.
   - Q:   `(n, n)` process noise covariance matrix.

   Outputs
   - x_p: `(n, 1)` or `(n,)` predicted state estimate at the next time step
          when applying the dynamics, i.e. x_{k|k-1}.
   - P_p: `(n, n)` covariance matrix of the state at the next time step, when
          applying the dynamics, i.e. P_{k|k-1}.
   """

   n = x_m.shape[0]
   assert (x_m.shape == (n, 1) or x_m.shape == (n,)) and \
          P_m.shape == (n, n) and \
          A.shape == (n, n) and \
          b.shape == x_m.shape and \
          Q.shape == (n, n) and \
          "Incorrect input shapes."

   # Prior update
   x_p = A @ x_m + b
   P_p = A @ P_m @ A.T + Q

   return x_p, P_p

def measurement_update(x_p: State, P_p: Covariance,
                       z: NDArray, H: NDArray, R: Covariance,
                       KALMAN_GAIN_FORM: Optional[bool] = True,
                       JOSEPH_FORM: Optional[bool] = False,
                       symmetry_tol: Optional[float] = 1e-8,
                       invertibility_eps: Optional[float] = 1e-20) \
                         -> Tuple[State, Covariance]:
   """
   Measurement update of the Kalman Filter, also referred to as the a posteriori
   update.

   The measurement equation is given by:
     z_k = H * x_k + w,
     where w is a zero-mean Gaussian RV with covariance R.

   Inputs
   - x_p: `(n, 1)` or `(n,)` prior estimate of the state, i.e. x_{k|k-1}.
   - P_p: `(n, n)` prior covariance matrix of the state, i.e. P_{k|k-1}.
   - z:   `(m, 1)` or `(m,)` observed measurement, i.e. z_k.
   - H:   `(m, n)` observation matrix.
   - R:   `(m, m)` measurement noise covariance matrix, symmetric definite
          postive.
   - symmetry_eps: `float` tolerance for checking symmetry of matrices. Default
                    is `1e-8`.
   - invertibility_eps: `float` small tolerance for checking invertibility of
                        matrices. Default is `1e-20`.
   Optional inputs
   - KALMAN_GAIN_FORM: `bool` flag to use the Kalman gain form. Default is
                       `True`.
   - JOSEPH_FORM:      `bool` flag to use the Joseph form for the covariance
                       update. This only works when using the `KALMAN_GAIN_FORM`
                       form and improves numerical stability. Default is
                       `False`.

   Outputs
   - x_m: `(n, 1)` or `(n,)` a posteriori state estimate after employing the
           measurement z_k, i.e. x_{k|k}.
   - P_m: `(n, n)` a posteriori covariance matrix of the state after employing
           the measurement z_k, i.e. x_{k|k}.
   """

   n, m = x_p.shape[0], z.shape[0]
   assert (x_p.shape == (n, 1) or x_p.shape == (n,)) and \
          P_p.shape == (n, n) and \
          (z.shape == (m, 1) or z.shape == (m,)) and \
          z.ndim == x_p.ndim and \
          H.shape == (m, n) and \
          R.shape == (m, m) and \
          "Incorrect input shapes."

   # Assertions
   if np.abs(P_p.T - P_p).max() > symmetry_tol:
       raise ValueError("The provided prior covariance matrix P_p is not " +
                        "symmetric.")

   if np.abs(R.T - R).max() > symmetry_tol:
       raise ValueError("The provided measurement noise covariance matrix R " +
                        " is not symmetric.")

   if np.linalg.det(P_p) < invertibility_eps:
       raise ValueError("The provided prior covariance matrix P_p is not " +
                        "is not invertible.")

   if np.linalg.det(R) < invertibility_eps:
       raise ValueError("The provided measurement noise covariance matrix R " +
                        " is not invertible.")

   # Kalman gain form
   if KALMAN_GAIN_FORM:

       # Use Cholesky decomposition if scipy is available
       if SCIPY_AVAILABLE:
           HPHR_and_lower = scipy.linalg.cho_factor(H @ P_p @ H.T + R,
                                                    check_finite=False)
           K = scipy.linalg.cho_solve(HPHR_and_lower, b=(P_p @ H.T).T,
                                      check_finite=False).T
       else:
           K = P_p @ H.T @ np.linalg.inv(H @ P_p @ H.T + R)

       x_m = x_p + K @ (z - H @ x_p)

       # Jospeh form (for numerical stability)
       if JOSEPH_FORM:
           P_m = (np.eye(n) - K @ H) @ P_p @ (np.eye(n) - K @ H).T + K @ R @ K.T
       else:
           P_m = (np.eye(n) - K @ H) @ P_p

   # Direct form without computing the Kalman gain
   else:
       R_inv = np.linalg.inv(R)
       P_m_inv = H.T @ R_inv @ H + np.linalg.inv(P_p)
       P_m = np.linalg.inv(P_m_inv)

       x_m = x_p + P_m @ H.T @ R_inv @ (z - H @ x_p)

   return x_m, P_m