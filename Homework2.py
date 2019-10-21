# Homework 2 Artificial Intelligence kvpatel
import numpy as np
from bga import BGA

# Greedy Search Algorithm
def greedySearchAlgorithm(type, cost, reach, budget):
    typeFlag = [0 for i in type]
    currentCost = 0
    currentReach = 0
    Flag = True
    while Flag:
        max = 0
        index = None
        for count, i in enumerate(cost):
            if i > max and (currentCost + i) <= budget and not typeFlag[count]:
                max = i
                index = str(count)
        if index == None:
            Flag = False
        else:
            typeFlag[int(index)] = 1
            currentCost = currentCost + cost[int(index)]
            currentReach = currentReach + reach[int(index)]
    return (''.join(str(e) for e in typeFlag), currentCost, currentReach)

# Genetic Algorithm
def geneticAlgorithm(type, cost, reach, budget):
    num_pop = 100
    problem_dimentions = len(type)

    test = BGA(pop_shape=(num_pop, problem_dimentions), method=values, GAcost=cost, GAreach=reach, GAbudget=budget, p_c=0.8, p_m=0.2, max_round = 100, early_stop_rounds=None, verbose = None, maximum=True)
    best_solution, best_fitness = test.run()
    totalCost = 0
    for count, i in enumerate(best_solution):
        if i == 1:
            totalCost = totalCost + cost[count]
    return (''.join(str(e) for e in best_solution), totalCost, best_fitness)

# Evaluation function for Genetic algorithm
def values(arr, costInThousands, expectedReach, totalBudget):
    """costInThousands = np.array([20, 30, 60, 70, 50, 90, 40])
    expectedReach = np.array([6, 5, 8, 9, 6, 7, 3])
    totalBudget = 100"""
    currentCost = 0
    currentReach = 0
    for count, i in enumerate(arr):
        if i == 1:
            currentCost = currentCost + costInThousands[count]
            currentReach = currentReach + expectedReach[count]
    if currentCost <= totalBudget:
        return currentReach
    else:
        return 0

# Driver program
if __name__ == "__main__":
    adPlacement = ['Magazine 1', 'Magazine 2', 'Magazine 3', 'Magazine 4', 'Magazine 5', 'Magazine 6', 'Magazine 7']
    costInThousands = [20, 30, 60, 70, 50, 90, 40]
    expectedReach = [6, 5, 8, 9, 6, 7, 3]
    totalBudget = 100
    greedyResult, totalCost, totalReach = greedySearchAlgorithm(adPlacement, costInThousands, expectedReach, totalBudget)
    print()
    print("Greedy Algorithm Output of first scenario")
    print("\""+greedyResult +"\" \nCost(In 1000’s): "+ str(totalCost) + " \nReach(In Million (M)): "+ str(totalReach))
    print()
    geneticResult, totalCost, totalReach = geneticAlgorithm(adPlacement, costInThousands, expectedReach, totalBudget)
    print("Genetic Algorithm Output of First Scenario")
    print("\""+geneticResult +"\" \nCost(In 1000’s): "+ str(totalCost) + " \nReach(In Million (M)): "+ str(totalReach))
    print()
    adPlacement = ['TV Sports', 'Music Radio', 'TV News', 'Newspapers', 'Web', 'Flyers']
    costInThousands = [3000, 800, 500, 2000, 600, 50]
    expectedReach = [80, 20, 22, 75, 10, 4]
    totalBudget = 5000
    greedyResult, totalCost, totalReach = greedySearchAlgorithm(adPlacement, costInThousands, expectedReach, totalBudget)
    print("Greedy Algorithm Output of Second Scenario")
    print("\""+greedyResult +"\" \nCost(In 1000’s): "+ str(totalCost) + " \nReach(In Million (M)): "+ str(totalReach))
    print()
    geneticResult, totalCost, totalReach = geneticAlgorithm(adPlacement, costInThousands, expectedReach, totalBudget)
    print("Genetic Algorithm Output of First Scenario")
    print("\""+geneticResult +"\" \nCost(In 1000’s): "+ str(totalCost) + " \nReach(In Million (M)): "+ str(totalReach))
    print()















    
