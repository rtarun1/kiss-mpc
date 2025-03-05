# Model Predictive Control written in Casadi.
https://github.com/user-attachments/assets/79f7ee61-ca49-416c-8fa6-676e0a2919f8

## Equations
$$
\begin{aligned}
& \underset{X}{\text{min}}& &f(X; P) = \sum_{t=1}^{N} \omega_x (x_t - x_g)^2 + \omega_y (y_t - y_g)^2 + \omega_\theta (\theta_t - \theta_g)^2 \\
& \underset{U}{\text{min}}& &f(U) = \sum_{t=1}^{N} \omega_a a_{t-1}^2 + \omega_{\alpha} \alpha_{t-1}^2 \\ 
& \text{subject to :}& & x_0 - x_I = 0; \quad \text{and} \quad y_0 - y_I = 0; \quad \text{and} \quad \theta_0 - \theta_I = 0 \\
& & & \forall t \in \\{1, \dots, N\\}, \quad x_t - (x_{t-1} + (v_I + \sum_{k=1}^{t} a_{k-1}T) \cos(\theta_{t-1}) T) = 0 \\
& & & \forall t \in \\{1, \dots, N\\}, \quad y_t - (y_{t-1} + (v_I + \sum_{k=1}^{t} a_{k-1}T) \sin(\theta_{t-1}) T) = 0 \\
& & & \forall t \in \\{1, \dots, N\\}, \quad \theta_{t} - (\theta_{t-1} + (\omega_I + \sum_{k=1}^{t} \alpha_{k-1}T) T) = 0 \\
& & & \forall i \in \\{1, \dots, O\\}, \forall t \in \{1, \dots, N\}, \quad \text{dist}(x_t, o_i) \geq I\\
& & & \forall t \in \\{1, \dots, N\\}, \quad v_L \leq v_i + \sum_{k=1}^{t} a_{k-1}T \leq v_U\\
& & & \forall t \in \\{1, \dots, N\\}, \quad \omega_L \leq \omega_i + \sum_{k=1}^{t} \alpha_{k-1}T \leq \omega_U \\
& & & \forall t \in \\{1, \dots, N+1\\}, \quad l_L \leq x_{t-1} \leq l_U \\
& & & \forall t \in \\{1, \dots, N\\}, \quad u_L \leq a_{t-1} \leq u_U \quad \text{and} \quad \alpha_L \leq \alpha_{t-1} \leq \alpha_U \\
& \text{where :}& & X = \\{ x_0, \dots, x_N, \quad y_0, \dots, y_N, \quad \theta_0, \dots, \theta_N \\} \\
& & & U = \\{ a_0, \dots, a_{N-1}, \quad \alpha_0, \dots, \alpha_{N-1} \\} \\
& & & P = \\{ x_I, y_I, \theta_I, x_G, y_G, \theta_G \\}
\end{aligned}
$$
