import numpy as np
import matplotlib.pyplot as plt

SGOLN =np.zeros(99)
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

def sgo_lognormal_collect_best_fitness(fitness, max_iter, n, dim, minx, maxx):
    Fitness_Curve = np.zeros(max_iter)
    Best_Fitness_Values = []  # This will store the best fitness value of each iteration
    society = [Person(fitness, dim, minx, maxx) for _ in range(n)]
    Xbest = society[0].position
    Fbest = society[0].fitness

    for person in society:
        if person.fitness < Fbest:
            Fbest = person.fitness
            Xbest = person.position

    for Iter in range(max_iter):
        for i in range(n):
            Xnew = np.exp(np.random.normal(loc=np.log(np.maximum(society[i].position, 1e-10)), scale=(maxx - minx) / 4))
            Xnew = [max(min(x, maxx), minx) for x in Xnew]
            fnew = fitness(Xnew)
            if fnew < society[i].fitness:
                society[i].position = Xnew
                society[i].fitness = fnew
            if fnew < Fbest:
                Fbest = fnew
                Xbest = Xnew
        Best_Fitness_Values.append(Fbest)
        Fitness_Curve[Iter] = Fbest

    return Xbest, Fitness_Curve, Best_Fitness_Values

def plot_fitness_curve(algorithm, fitness_curve, title):
    plt.plot(range(len(fitness_curve)), fitness_curve, label=algorithm)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

#PARAMETERS
selected_function = 2 #SOS
max_iter =2000
n = 200
dim = 100
minx = -5.12
maxx = 5.12

fitness_function = [f1, f2, f3, f4, f5][selected_function - 1]
Xbest_sgo, Fitness_Curve_sgo, Best_Fitness_Values_sgo = sgo_lognormal_collect_best_fitness(
    fitness_function, max_iter, n, dim, minx, maxx)

plot_fitness_curve("SGO_LN", Best_Fitness_Values_sgo, f"Fitness Curve - SGO with LNNORMAL")

SGOLN = np.array(Best_Fitness_Values_sgo)


# Print the best fitness values from each iteration
print("Best Fitness Values Each Iteration:")
print(Best_Fitness_Values_sgo)

# Print the overall best fitness value
print("Overall Best Fitness Value:", np.min(SGOLN))
