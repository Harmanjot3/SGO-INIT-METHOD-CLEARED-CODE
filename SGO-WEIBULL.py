import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.special import gamma

class Person:
    def __init__(self, fitness, dim, minx, maxx, scale_param, shape_param):
        self.position = self.initialize_position(dim, minx, maxx, scale_param, shape_param)
        self.fitness = fitness(self.position)

    def initialize_position(self, dim, minx, maxx, scale_param, shape_param):
        return minx + weibull_min.rvs(c=shape_param, size=dim, scale=scale_param) * (maxx - minx)

def f1(position):  # rastrigin
    return np.sum([(xi * xi) - (10 * np.cos(2 * np.pi * xi)) + 10 for xi in position])

def f2(position):
    return np.sum(np.square(position))

def f3(position):  # rosenbrock
    fitness_value = 0.0
    for i in range(len(position) - 1):
        xi = position[i]
        xi1 = position[i + 1]
        fitness_value += 100 * (xi1 - xi ** 2) ** 2 + (1 - xi) ** 2
    return fitness_value

def f4(position):  # griewank
    fitness_value = 0.0
    sum_term = 0.0
    prod_term = 1.0
    for i in range(len(position)):
        xi = position[i]
        sum_term += xi ** 2 / 4000
        prod_term *= np.cos(xi / np.sqrt(i + 1))
    fitness_value = 1 + sum_term - prod_term
    return fitness_value

def f5(position):  # ackley
    dim = len(position)
    sum_term = np.sum(np.square(position))
    cos_term = np.sum(np.cos(2 * np.pi * np.array(position)))
    fitness_value = -20 * np.exp(-0.2 * np.sqrt(sum_term / dim)) - np.exp(cos_term / dim) + 20 + np.exp(1)
    return fitness_value

def sgo(fitness, max_iter, n, dim, minx, maxx, scale_param, shape_param):
    Fitness_Curve = np.zeros(max_iter)
    WBEST = np.zeros(max_iter)
    society = [Person(fitness, dim, minx, maxx, scale_param, shape_param) for _ in range(n)]
    Xbest = [0.0 for _ in range(dim)]
    c = 0.2
    Fbest = float('inf')

    for i in range(n):
        if society[i].fitness < Fbest:
            Fbest = society[i].fitness
            Xbest = np.copy(society[i].position)

    Iter = 0
    while Iter < max_iter:
        for i in range(n):
            Xnew = [c * society[i].position[j] + np.random.rand() * (Xbest[j] - society[i].position[j])
                    for j in range(dim)]
            Xnew = [max(min(x, maxx), minx) for x in Xnew]

            fnew = fitness(Xnew)
            if fnew < society[i].fitness:
                society[i].position = Xnew
                society[i].fitness = fnew
            if fnew < Fbest:
                Fbest = fnew
                Xbest = Xnew

        Fitness_Curve[Iter] = Fbest
        WBEST[Iter] = Fbest
        Iter += 1

    return Xbest, Fitness_Curve , Fbest  , WBEST

def plot_fitness_curve(algorithm, fitness_curve, title):
    plt.plot(range(len(fitness_curve)), fitness_curve, label=algorithm)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.show()

# User input
selected_function = int(input("Select a function: "))

max_iter =2000
n = 200
dim = 100
minx = -5.12
maxx = 5.12
scale_param = 1.0  # scale parameter
shape_param = 1.5

fitness_function = None
function_name = ""

if selected_function == 1:
    fitness_function = f1
    function_name = "Rastrigin"
if selected_function == 2:
    fitness_function = f2
    function_name = "Sphere"
if selected_function == 3:
    fitness_function = f3
    function_name = "Rosenbrock"
if selected_function == 4:
    fitness_function = f4
    function_name = "Griewank"
if selected_function == 5:
    fitness_function = f5
    function_name = "Ackley"

# Run SGO with Weibull distribution
Xbest_sgo, Fitness_Curve_sgo , Fbest_sgo , WBEST= sgo(fitness_function, max_iter, n, dim, minx, maxx, scale_param, shape_param)

# Plot fitness curve for SGO
plot_fitness_curve("SGO with Weibull", Fitness_Curve_sgo, f"Fitness Curve - SGO - {function_name} - Weibull Distribution")
print(f"Best Fitness Value: {Fbest_sgo}")

for values in  WBEST:
    print(values)