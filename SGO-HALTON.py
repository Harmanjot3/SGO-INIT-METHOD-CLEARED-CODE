import numpy as np
import matplotlib.pyplot as plt

def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def get_primes(n):
    """Generate a list of the first n prime numbers."""
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

def halton_sequence_value(index, base):
    result = 0.0
    f = 1.0
    while index > 0:
        f = f / base
        result = result + f * (index % base)
        index = index // base
    return result

def generate_halton(dim, n):
    primes = get_primes(dim)
    points = np.zeros((n, dim))
    for d in range(dim):
        base = primes[d]
        for i in range(1, n + 1):
            points[i-1, d] = halton_sequence_value(i, base)
    return points

def scale_halton_points(points, minx, maxx):
    return minx + points * (maxx - minx)

class Person:
    def __init__(self, fitness, dim, minx, maxx, position=None):
        if position is None:
            position = np.random.rand(dim) * (maxx - minx) + minx
        self.position = position
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


def f5(position):
    dim = len(position)
    sum_sq = np.sum(np.square(position))
    sum_cos = np.sum(np.cos(2 * np.pi * np.array(position)))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim)) - np.exp(sum_cos / dim) + 20 + np.exp(1)


def sgo(fitness, max_iter, n, dim, minx, maxx):
    Fitness_Curve = np.zeros(max_iter)
    HBEST = np.zeros(max_iter)
    halton_points = generate_halton(dim, n)
    scaled_halton_points = scale_halton_points(halton_points, minx, maxx)
    society = [Person(fitness, dim, minx, maxx, position=scaled_halton_points[i]) for i in range(n)]
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
        HBEST[Iter] = Fbest

    return Xbest, Fitness_Curve , Fbest , HBEST

def plot_fitness_curve(algorithm, fitness_curve, title):
    plt.plot(range(len(fitness_curve)), fitness_curve, label=algorithm)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.show()


#PARAMETERS
max_iter =2000
n = 200
dim = 100
minx = -5.12
maxx = 5.12

fitness_function = f2
function_name = "SOS"

# Run SGO with Halton sequence initialization
Xbest_sgo, Fitness_Curve_sgo , Fbest_sgo ,  HBEST= sgo(fitness_function, max_iter, n, dim, minx, maxx)

# Plot fitness curve for SGO
plot_fitness_curve("SGO", Fitness_Curve_sgo, f"Fitness Curve - Halton - SGO - {function_name}")
print(f"Best Fitness Value: {Fbest_sgo}")

for values in HBEST:
    print(values)