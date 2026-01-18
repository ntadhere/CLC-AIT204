from model import LinearRegression

learning_rate = 5
n_iterations = 100

m = LinearRegression(learning_rate,n_iterations)

l_rate, n_iter = m.test_init()

print(f"Learing Rate: {l_rate}\nn_Iterations: {n_iter}")