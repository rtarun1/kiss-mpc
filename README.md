# KissMPC: Keep It Simple and Straightforward MPC Formulation

This repository implements a velocity-based Model Predictive Control (MPC) planner using CasADi. The optimization formulation minimizes trajectory tracking error, penalizes undesired motion (like negative or high angular velocity), and obeys dynamics and velocity constraints.

---

##  Optimization Problem Formulation

We solve the following nonlinear program at each planning step:

###  Objective funtion

Minimize tracking error and control effort:

$$
\begin{aligned}
\min_{X, U} \quad 
& \underbrace{\sum_{t=1}^{N} \left[ 
    W_x (x_t - x_g)^2 + 
    W_y (y_t - y_g)^2 + 
    W_\theta (\theta_t - \theta_g)^2
\right]}_{\text{Goal Tracking}} \\
& + \underbrace{\sum_{t=0}^{N-1} \left[
    W_v^- \cdot \min(0, v_t)^2 + 
    W_v^+ \cdot \max(0, v_t)^2 +
    W_\omega \cdot \omega_t^2
\right]}_{\text{Control Penalty (forward + reverse)}}
\end{aligned}
$$

---

### Constraints

#### Initial state constraint:

$$
x_0 = x_I, \quad y_0 = y_I, \quad \theta_0 = \theta_I
$$

#### Kinematic model:

For all $t \in \{0, 1, \dots, N-1\}$:

$$
\begin{aligned}
x_{t+1} &= x_t + v_t \cdot \cos(\theta_t) \cdot T \\
y_{t+1} &= y_t + v_t \cdot \sin(\theta_t) \cdot T \\
\theta_{t+1} &= \theta_t + \omega_t \cdot T
\end{aligned}
$$

#### Control bounds:

For all $t \in \{0, 1, \dots, N-1\}$:

$$
v_L \leq v_t \leq v_U, \quad
\omega_L \leq \omega_t \leq \omega_U
$$

#### State bounds:

For all $t \in \{0, 1, \dots, N\}$:

$$
x_L \leq x_t \leq x_U, \quad
y_L \leq y_t \leq y_U
$$

---

### Variable Definitions

* $X = \{x_t, y_t, \theta_t\}_{t=0}^{N}$: state trajectory
* $U = \{v_t, \omega_t\}_{t=0}^{N-1}$: control inputs
* $P = \{x_I, y_I, \theta_I, x_G, y_G, \theta_G\}$: initial and goal states

---

## Formulation in code yet to test

$$
\begin{aligned}
& & & \forall i \in \\{1, \dots, O\\}, \forall t \in \{1, \dots, N\}, \quad \text{dist}(x_t, o_i) \geq I\\
\end{aligned}
$$
---

## Credit

This work is inspired by the [Casadi-MPC](https://github.com/Smart-Wheelchair-RRC/casadi-mpc) project.
