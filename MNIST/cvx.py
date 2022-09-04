import cvxpy as cp
import numpy as np

# Problem data.
n = 100

np.random.seed(1)

# Construct the problem.
p = cp.Variable(3)
b = cp.Parameter(1)
b.value = [10]
#b_star = cp.Constant(10)

objective = cp.Minimize((1/2)*cp.sum_squares(p/(cp.power(2,b) - 1)))
constraints = [0 <= b, 0 <= p]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
print(p.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)