import numpy as np
import matplotlib.pyplot as plt
import random


class Person:
    def __init__(self, fitness, dim, position):
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


def generate_knuth_sequence(x_min, x_max, dim):
    a = list(np.linspace(x_min, x_max, dim))
    random.shuffle(a)
    return a


def initialize_population(fitness, n, dim, minx, maxx):
    society = []
    for _ in range(n):
        position = generate_knuth_sequence(minx, maxx, dim)
        society.append(Person(fitness, dim, position))
    return society


def sgo(fitness, max_iter, n, dim, minx, maxx):
    Fitness_Curve = np.zeros(max_iter)
    KNBEST = np.zeros(max_iter)
    society = initialize_population(fitness, n, dim, minx, maxx)

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
        KNBEST[Iter] = Fbest

    return Xbest, Fitness_Curve ,Fbest, KNBEST


def plot_fitness_curve(algorithm, fitness_curve, title):
    plt.plot(range(len(fitness_curve)), fitness_curve, label=algorithm)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.show()


# Example usage
max_iter =2000
n = 200
dim = 100
# Dimensionality of the problem
minx = -5.12
maxx = 5.12

fitness_function = f2
function_name = "sos"

# Run SGO with Knuth sequence initialization
Xbest_sgo, Fitness_Curve_sgo , Fbest_sgo , KNBEST = sgo(fitness_function, max_iter, n, dim, minx, maxx)

# Plot fitness curve for SGO
plot_fitness_curve("SGO", Fitness_Curve_sgo, f"Fitness Curve knuth - SGO - {function_name}")
print(f"Best Fitness Value: {Fbest_sgo}")

for values in KNBEST:
    print(values)