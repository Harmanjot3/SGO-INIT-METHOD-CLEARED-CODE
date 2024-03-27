import numpy as np
import matplotlib.pyplot as plt
import random


class Person:
    def __init__(self, fitness, dim, position):
        self.position = position[:dim]  # Ensure the dimensionality is correct
        self.fitness = fitness(self.position)

def f1(position):  # Rastrigin
    return np.sum([(xi * xi) - (10 * np.cos(2 * np.pi * xi)) + 10 for xi in position])

def f2(position): #SOS
    return np.sum(np.square(position))

def f3(position): # Rosenbrock
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
def torus_coordinates(D, r, theta, delta):
    a = (D + r * np.cos(theta)) * np.cos(delta)
    b = (D + r * np.cos(theta)) * np.sin(delta)
    c = r * np.sin(theta)
    return a, b, c

def torus_series(k, dim, D, r):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  # Extend as needed
    positions = []
    for i in range(dim // 3):  # Assuming each dimension handled by 3 coordinates (a, b, c)
        prime = primes[i % len(primes)]  # Loop through primes for dimensions > len(primes)
        f_theta = k * np.sqrt(prime) - np.floor(k * np.sqrt(prime))
        f_delta = k * np.sqrt(prime + 1) - np.floor(k * np.sqrt(prime + 1))
        theta = 2 * np.pi * f_theta
        delta = 2 * np.pi * f_delta
        a, b, c = torus_coordinates(D, r, theta, delta)
        positions.extend([a, b, c])
    return positions

def initialize_population_torus(fitness, n, dim, D, r):
    society = []
    for k in range(n):
        position = torus_series(k, dim, D, r)
        society.append(Person(fitness, dim, position))
    return society


def sgo(fitness, max_iter, n, dim, D, r):
    Fitness_Curve = np.zeros(max_iter)
    TOBEST = np.zeros(max_iter)
    society = initialize_population_torus(fitness, n, dim, D, r)

    Xbest = None
    Fbest = float('inf')

    for person in society:
        if person.fitness < Fbest:
            Fbest = person.fitness
            Xbest = np.copy(person.position)

    for Iter in range(max_iter):
        for person in society:
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
        TOBEST[Iter] = Fbest
    return Xbest, Fitness_Curve ,Fbest , TOBEST


def plot_fitness_curve(algorithm, fitness_curve, title):
    plt.plot(range(len(fitness_curve)), fitness_curve, label=algorithm)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.show()


# Parameters
max_iter =200
n = 20
dim = 12
# Make sure this is a multiple of 3 for the torus_series function
D, r = 5, 2  # Torus parameters
minx, maxx = -5.12, 5.12  # Bounds for SGO adjustments, if needed

# Fitness function selection
fitness_function = f2
function_name = "SOS"

# Run SGO with Torus distribution initialization
Xbest_sgo, Fitness_Curve_sgo , Fbest_sgo ,TOBEST= sgo(fitness_function, max_iter, n, dim, D, r)

# Plot fitness curve for SGO
plot_fitness_curve("SGO", Fitness_Curve_sgo, f"Fitness Curve Torous - SGO - {function_name}")
print(f"Best Fitness Value: {Fbest_sgo}")

for values in TOBEST:
    print(values)