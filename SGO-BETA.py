import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

class Person:
    def __init__(self, fitness, dim, minx, maxx):
        self.position = [minx + np.random.rand() * (maxx - minx) for _ in range(dim)]
        self.fitness = fitness(self.position)

def f1(position):  # Rastrigin
    return np.sum([(xi * xi) - (10 * np.cos(2 * np.pi * xi)) + 10 for xi in position])

def f2(position):
    return np.sum(np.square(position))

def f3(position):
    return sum(100 * (xj - xi ** 2) ** 2 + (1 - xi) ** 2 for xi, xj in zip(position[:-1], position[1:]))

def f4(position):  # Griewank
    sum_term = np.sum(np.square(position))
    prod_term = np.prod(np.cos(position / np.sqrt(np.arange(1, len(position) + 1))))
    return sum_term / 4000 - prod_term + 1


def f5(position):  # Ackley
    dim = len(position)
    sum_term = np.sum(np.square(position))
    cos_term = np.sum(np.cos(2 * np.pi * np.array(position)))
    return -20 * np.exp(-0.2 * np.sqrt(sum_term / dim)) - np.exp(cos_term / dim) + 20 + np.exp(1)

def sgo(fitness, max_iter, n, dim, minx, maxx):
    Fitness_Curve = np.zeros(max_iter)
    BETABEST = np.zeros(max_iter)
    society = [Person(fitness, dim, minx, maxx) for _ in range(n)]
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
            Xnew = [c * society[i].position[j] + np.random.beta(2, 3) * (Xbest[j] - society[i].position[j])
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
        BETABEST[Iter] = Fbest
        Iter += 1

    return Xbest, Fitness_Curve, Fbest, BETABEST


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

fitness_function = None
function_name = ""

if selected_function == 1:
    fitness_function = f1
    function_name = "Rastrigin"
if selected_function == 2:
    fitness_function = f2
    function_name = "SOS"
if selected_function == 3:
    fitness_function = f3
    function_name = "Rosenbrock"
if selected_function == 4:
    fitness_function = f4
    function_name = "Griewank"
if selected_function == 5:
    fitness_function = f5
    function_name = "Ackley"

# Run SGO
Xbest_sgo, Fitness_Curve_sgo, Fbest_sgo, BETABEST = sgo(fitness_function, max_iter, n, dim, minx, maxx)

print(f"Best Fitness Value for each iteration:")
for values in BETABEST:
    print(values)
# Plot fitness curve for SGO
plot_fitness_curve("SGO", Fitness_Curve_sgo, f"Fitness Curve - SGO with Beta - {function_name}")
print(f"Best Fitness Value: {Fbest_sgo}")
