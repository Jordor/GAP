Timport pandas as pd
import numpy as np

MAP = pd.read_csv(r'C:\Users\zosia\OneDrive\Pulpit\MSC Bioinf Leuven\EA\EA - project\tour50.csv')
print(MAP)
#nr of cities
R = 49

class Individual:
    def __init__(self):
        self.path
        self.fintess

def randomPath():
    path = np.random.permutation(R)

def fitness(indiv: Individual):
    pass

if __name__ == '__main__':