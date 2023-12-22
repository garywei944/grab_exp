import cvxpy as cp
import numpy as np

# Problem data.
m = 30
np.random.seed(1)
A = np.random.randn(m, m)
A = A.T @ A
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(m)
# z = cp.Variable(2, boolean=True)
# minimize x^Tb+x^TAx+MSE(x^Tx, 1)
objective = cp.Minimize(x.T @ b + cp.quad_form(x, A))
# https://stackoverflow.com/a/65003841
# constraints = [cp.sum(z) == 1, [1, -1] @ z == x]
constraints = [cp.sum_squares(x) >= 20, m, x <= 1, x >= -1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)
