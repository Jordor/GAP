import pandas as pd
import numpy as np

MAP = pd.read_csv(r'C:\Users\zosia\OneDrive\Pulpit\MSC Bioinf Leuven\EA\EA - project\tour50.csv')
print(MAP)
#nr of cities
R = 49

class Path:
    def __init__(self):
        self.cycle
        self.fintess

def randomPath():
    path = np.random.permutation(R)
    return path

def calculate_fitness(path: Path):
    # update fitness value
    pass

def mutate_path(path: Path):
    #
    pass

def cross_parents():
    pass

def algorith():
    # population list of paths

    for p in population:
        population[p] = mutate_path(p)

    for i in range(0, len(population)\2):



if __name__ == '__main__':
