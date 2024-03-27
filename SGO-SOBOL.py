import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol

def next_power_of_2(n):
    """Calculate the next power of 2 greater than or equal to n."""
    return 2**np.ceil(np.log2(n))

def generate_sobol(dim, n, minx, maxx):
    """Generate n points in [minx, maxx] for each dimension using Sobol sequences."""
    n_adjusted = int(next_power_of_2(n))  # Adjust n to the next power of 2
    sobol_gen = Sobol(d=dim)
    samples = sobol_gen.random(n=n_adjusted)
    return minx + (samples * (maxx - minx))[:n]  # Ensure only n points are used

class Person:
    def __init__(self, fitness, dim, minx, maxx, position=None):
        if position is None:
            position = [minx + np.random.rand() * (maxx - minx) for _ in range(dim)]
        self.position = position
        self.fitness = fitness(self.position)

def f1(position):
    return np.sum([(xi ** 2) - (10 * np.cos(2 * np.pi * xi)) + 10 for xi in position])

def f2(position):
    return np.sum(np.square(position))


def f3(position):
    return sum(100 * (xj - xi ** 2) ** 2 + (1 - xi) ** 2 for xi, xj in zip(position[:-1], position[1:]))

def f4(position):  # Griewank
    sum_term = np.sum(np.square(position))
    prod_term = np.prod(np.cos(position / np.sqrt(np.arange(1, len(position) + 1))))
    return sum_term / 4000 - prod_term + 1

def f5(position):
    dim = len(position)
    sum_sq = np.sum(np.square(position))
    sum_cos = np.sum(np.cos(2 * np.pi * np.array(position)))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim)) - np.exp(sum_cos / dim) + 20 + np.exp(1)

def sgo(fitness, max_iter, n, dim, minx, maxx):
    Fitness_Curve = np.zeros(max_iter)
    SOBOLBEST = np.zeros(max_iter)  # This array will store the best fitness value found in each iteration
    initial_positions = generate_sobol(dim, n, minx, maxx)
    society = [Person(fitness, dim, minx, maxx, position=position) for position in initial_positions]
    Xbest = None
    Fbest = float('inf')

    for person in society:
        if person.fitness < Fbest:
            Fbest = person.fitness
            Xbest = np.copy(person.position)

    for Iter in range(max_iter):
        for i, person in enumerate(society):
            Xnew = [0.2 * x + np.random.rand() * (xb - x) for x, xb in zip(person.position, Xbest)]
            Xnew = [max(min(x, maxx), minx) for x in Xnew]
            fnew = fitness(Xnew)
            if fnew < person.fitness:
                person.position = Xnew
                person.fitness = fnew
            if fnew < Fbest:
                Fbest = fnew
                Xbest = Xnew
        Fitness_Curve[Iter] = Fbest
        SOBOLBEST[Iter] = Fbest  # Assuming you want to store the best fitness value of each iteration
        print(f" {Fbest}")
    return Xbest, Fitness_Curve, Fbest, SOBOLBEST

def plot_fitness_curve(algorithm, fitness_curve, title):
    plt.plot(range(len(fitness_curve)), fitness_curve, label=algorithm)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.show()

#PARAMETER
max_iter =200
n = 20
dim = 10
minx = -5.12
maxx = 5.12

fitness_function = f2
function_name = "SOS"

# Run SGO with Sobol initialization
Xbest_sgo, Fitness_Curve_sgo,SOBOLBEST,Fbest_sgo= sgo(fitness_function, max_iter, n, dim, minx, maxx)

# Plot fitness curve for SGO
plot_fitness_curve("SGO", Fitness_Curve_sgo, f"Fitness Curve- SOBOL - SGO - {function_name}")
print(f"Best Fitness Value: {Fbest_sgo}")