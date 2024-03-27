import numpy as np
import matplotlib.pyplot as plt

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

def sgo_normal(fitness, max_iter, n, dim, minx, maxx):
    Fitness_Curve = np.zeros(max_iter)
    NBEST = np.zeros(max_iter)
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
            Xnew = np.random.normal(loc=society[i].position, scale=(maxx - minx) / 4)
            Xnew = [max(min(x, maxx), minx) for x in Xnew]

            fnew = fitness(Xnew)
            if fnew < society[i].fitness:
                society[i].position = Xnew
                society[i].fitness = fnew
            if fnew < Fbest:
                Fbest = fnew
                Xbest = Xnew

        Fitness_Curve[Iter] = Fbest
        NBEST[Iter] = Fbest
        Iter += 1

    return Xbest, Fitness_Curve , Fbest , NBEST

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

# Run SGO with Normal distribution initialization
Xbest_sgo, Fitness_Curve_sgo , Fbest_sgo , NBEST = sgo_normal(fitness_function, max_iter, n, dim, minx, maxx)

# Plot fitness curve for SGO with Normal distribution initialization
plot_fitness_curve("SGO (Normal)", Fitness_Curve_sgo, f"Fitness Curve - SGO - {function_name} - Normal")
print(f"Best Fitness Value: {Fbest_sgo}")

for values in NBEST:
    print(values)