import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
m = len(y)

def hypothesis (x,  theta):
    return theta[0] + theta[1] * x

def compute_cost (x, y, theta):
    m = len(y)
    predictions = hypothesis (x,  theta)
    errors = predictions - y
    cost = (1/ (2 * m)) * np.sum(errors ** 2)
    return cost

theta1 = np.array([0.0, 0.0])
theta2 = np.array([0.0, 1.0])
theta3 = np.array([1.0, 0.0])


print("cost for theta1 [0,0] :", compute_cost(x, y, theta1))
print("cost for theta2 [0,1] :", compute_cost(x, y, theta2))
print("cost for theta3 [1,0] :", compute_cost(x, y, theta3))

assert abs (compute_cost(x, y, theta1)- 7/3) < 1e-9
assert abs (compute_cost(x, y, theta2) - 0.0) < 1e-9
assert abs (compute_cost(x, y, theta3) - 5/6) < 1e-9
print("All tests passed")
