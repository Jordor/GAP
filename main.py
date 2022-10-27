import pandas as pd
import numpy as np

MAP = pd.read_csv(r'../data/tour50.csv')
#nr of cities
n_city = 49
TIME_LIMIT = 100000


class Path:
    def __init__(self):
        self.cycle
        self.fintess

def read_csv():
    MAP = pd.read_csv(r'../data/tour50.csv')
    #nr of cities
    pass

def randomPath():
    path = np.random.permutation(n_city)
    return path

def calculate_fitness(path: Path):
    # update fitness value
    pass

def mutate_path(path: Path):
    #
    pass

def cross_parents():
    pass

def initialize_population():
    pass

def variation(population, mutationRate):
    # population list of paths
    for p in population:
        population[p] = mutate_path(p)

    for i in range(0, len(population)/2):
        pass


if __name__ == '__main__':
    read_csv()
    population = initialize_population()
    while (TIME_LIMIT > 0):
        TIME_LIMIT--
        pass
