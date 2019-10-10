# Homework 1 Part 2 Artificial Intelligence kvpatel
from collections import deque
import queue as queue
from timeit import default_timer as timer
import math

N = 0
maze = []

# Internal Representation of Matrix
matrix = {}
def internalRepresentation():
    count = 1
    x = 0
    y = 0
    for i in range(N):
        for j in range(N):
            matrix[count] = [i,j,maze[i][j]]
            count+=1
    printMaze(maze)

# A utility function to print internal representation as a square maze in console and write to output file as well.
def printMaze(data):
    print("-----------------")
    f.write('Input Maze Problem : \n')
    for i in range(len(data)):
        for j in range(len(data[i])):
            print(' ' +data[i][j], end=' ')
            f.write(' ' +data[i][j]+ ' ')
        print()
        f.write('\n')
    print("-----------------")

# Visited nodes incremented by parameter value
visitedNodes = 0
lengthOfSolution = 1
def nodesVisited(v):
    global visitedNodes
    visitedNodes += v
    return visitedNodes

# It counts total cost from initial state to final state
def pathCost(cost):
    global lengthOfSolution
    lengthOfSolution += cost
    return lengthOfSolution

# Position of element returns
def getKey(data):
    keys = []
    for i in data:
        for index, value in matrix.items():
            if value == i[:3]:
                keys.append(index)
                break
    return keys

# Return Path numbers as a position in maze
# Top to bottom and left to right starting from 1.
def getNumbericPath(data):
    length = len(data) - 2
    while length != -1:
        current = data[length]
        next = data[length+1]
        if len(current) == 3:
            currentValue = current[2]
        else:
            currentValue = current[3]
        if len(next) == 3:
            nextValue = next[2]
        else:
            nextValue = next[3]
        if currentValue == 'a':
            nextNode = 'b'
        if currentValue == 'b':
            nextNode = 'c'            
        if currentValue == 'c':
            nextNode = 'a'
        if nextValue != nextNode:
            del data[length]
            length = length - 1
            continue
        if ((current[0] - next[0]) == 0 and (current[1] - next[1]) in [1, -1]) or ((current[1] - next[1]) == 0 and (current[0] - next[0]) in [1, -1]):
            addingCost = 2 if len(current) == 4 else 1
            pathCost(addingCost)
        else:
            del data[length]
            length = length - 1
            continue
        length = length - 1
    return getKey(data)

# This function returns set of states(successor) that can be reached from current state.
# It allows 1 step and 2 step elements(Total 12 States).
succOfA = ['b','*']
succOfB = ['c','*']
succOfC = ['a','*']
succOfstar = ['a','b','c','*']
def succ(s):
    x = s[0]
    y = s[1]
    currentNode = s[2]
    succSet = []
    if currentNode == '*':
        currentNode = s[3] if len(s) == 4 else '*'
        nextNode = succOfstar
    if currentNode == 'a':
        nextNode = succOfA
        secondNextNode = succOfB
    if currentNode == 'b':
        nextNode = succOfB
        secondNextNode = succOfC
    if currentNode == 'c':
        nextNode = succOfC
        secondNextNode = succOfA
    # Going Up(1-Step)
    if x-1 >= 0 and y >= 0 and x-1 < N and y < N and maze[x-1][y] in nextNode:
        if maze[x-1][y] == '*' :
            succSet.append([x-1,y,maze[x-1][y], nextNode[0]])
        else:
            succSet.append([x-1,y,maze[x-1][y]])
    # Going Down(1-Step)
    if x+1 >= 0 and y >= 0 and x+1 < N and y < N and maze[x+1][y] in nextNode:
        if maze[x+1][y] == '*' :
            succSet.append([x+1,y,maze[x+1][y], nextNode[0]])
        else:
            succSet.append([x+1,y,maze[x+1][y]])
    # Going Left(1-Step)
    if x >= 0 and y-1 >= 0 and x < N and y-1 < N and maze[x][y-1] in nextNode:
        if maze[x][y-1] == '*' :
            succSet.append([x,y-1,maze[x][y-1], nextNode[0]])
        else:
            succSet.append([x,y-1,maze[x][y-1]])
    # Going Right(1-Step)
    if x >= 0 and y+1 >= 0 and x < N and y+1 < N and maze[x][y+1] in nextNode:
        if maze[x][y+1] == '*' :
            succSet.append([x,y+1,maze[x][y+1], nextNode[0]])
        else:
            succSet.append([x,y+1,maze[x][y+1]])
    # Going Two Up(2-Step)
    if x-1 >= 0 and y >= 0 and x-1 < N and y < N and maze[x-1][y] in nextNode:
        if x-2 >= 0 and y >= 0 and x-2 < N and y < N and maze[x-2][y] in secondNextNode:
            doubleSuccSet = []
            if maze[x-1][y] == '*' :
                doubleSuccSet.append([x-1,y,maze[x-1][y], nextNode[0]])
            else:
                doubleSuccSet.append([x-1,y,maze[x-1][y]])
            if maze[x-2][y] == '*' :
                doubleSuccSet.append([x-2,y,maze[x-2][y], secondNextNode[0]])
            else:
                doubleSuccSet.append([x-2,y,maze[x-2][y]])
            succSet.append(doubleSuccSet)
    # Going Two Down(2-Step)
    if x+1 >= 0 and y >= 0 and x+1 < N and y < N and maze[x+1][y] in nextNode:
        if x+2 >= 0 and y >= 0 and x+2 < N and y < N and maze[x+2][y] in secondNextNode:
            doubleSuccSet = []
            if maze[x+1][y] == '*' :
                doubleSuccSet.append([x+1,y,maze[x+1][y], nextNode[0]])
            else:
                doubleSuccSet.append([x+1,y,maze[x+1][y]])
            if maze[x+2][y] == '*' :
                doubleSuccSet.append([x+2,y,maze[x+2][y], secondNextNode[0]])
            else:
                doubleSuccSet.append([x+2,y,maze[x+2][y]])
            succSet.append(doubleSuccSet)
    # Going Two Right(2-Step)
    if x >= 0 and y+1 >= 0 and x < N and y+1 < N and maze[x][y+1] in nextNode:
        if x >= 0 and y+2 >= 0 and x < N and y+2 < N and maze[x][y+2] in secondNextNode:
            doubleSuccSet = []
            if maze[x][y+1] == '*' :
                doubleSuccSet.append([x,y+1,maze[x][y+1], nextNode[0]])
            else:
                doubleSuccSet.append([x,y+1,maze[x][y+1]])
            if maze[x][y+2] == '*' :
                doubleSuccSet.append([x,y+2,maze[x][y+2], secondNextNode[0]])
            else:
                doubleSuccSet.append([x,y+2,maze[x][y+2]])
            succSet.append(doubleSuccSet)
    # Going Two Left(2-Step)
    if x >= 0 and y-1 >= 0 and x < N and y-1 < N and maze[x][y-1] in nextNode:
        if x >= 0 and y-2 >= 0 and x < N and y-2 < N and maze[x][y-2] in secondNextNode:
            doubleSuccSet = []
            if maze[x][y-1] == '*' :
                doubleSuccSet.append([x,y-1,maze[x][y-1], nextNode[0]])
            else:
                doubleSuccSet.append([x,y-1,maze[x][y-1]])
            if maze[x][y-2] == '*' :
                doubleSuccSet.append([x,y-2,maze[x][y-2], secondNextNode[0]])
            else:
                doubleSuccSet.append([x,y-2,maze[x][y-2]])
            succSet.append(doubleSuccSet)
    # Going Down and Left(2-Step)
    if x+1 >= 0 and y >= 0 and x+1 < N and y < N and maze[x+1][y] in nextNode:
        if x+1 >= 0 and y-1 >= 0 and x+1 < N and y-1 < N and maze[x+1][y-1] in secondNextNode:
            doubleSuccSet = []
            if maze[x+1][y] == '*' :
                doubleSuccSet.append([x+1,y,maze[x+1][y], nextNode[0]])
            else:
                doubleSuccSet.append([x+1,y,maze[x+1][y]])
            if maze[x+1][y-1] == '*' :
                doubleSuccSet.append([x+1,y-1,maze[x+1][y-1], secondNextNode[0]])
            else:
                doubleSuccSet.append([x+1,y-1,maze[x+1][y-1]])
            succSet.append(doubleSuccSet)
    # Going Down and Right(2-Step)
    if x+1 >= 0 and y >= 0 and x+1 < N and y < N and maze[x+1][y] in nextNode:
        if x+1 >= 0 and y+1 >= 0 and x+1 < N and y+1 < N and maze[x+1][y+1] in secondNextNode:
            doubleSuccSet = []
            if maze[x+1][y] == '*' :
                doubleSuccSet.append([x+1,y,maze[x+1][y], nextNode[0]])
            else:
                doubleSuccSet.append([x+1,y,maze[x+1][y]])        
            if maze[x+1][y+1] == '*' :
                doubleSuccSet.append([x+1,y+1,maze[x+1][y+1], secondNextNode[0]])
            else:
                doubleSuccSet.append([x+1,y+1,maze[x+1][y+1]])
            succSet.append(doubleSuccSet)
    # Going Up and Left(2-Step)
    if x-1 >= 0 and y >= 0 and x-1 < N and y < N and maze[x-1][y] in nextNode:
        if x-1 >= 0 and y-1 >= 0 and x-1 < N and y-1 < N and maze[x-1][y-1] in secondNextNode:
            doubleSuccSet = []
            if maze[x-1][y] == '*' :
                doubleSuccSet.append([x-1,y,maze[x-1][y], nextNode[0]])
            else:
                doubleSuccSet.append([x-1,y,maze[x-1][y]])
            if maze[x-1][y-1] == '*' :
                doubleSuccSet.append([x-1,y-1,maze[x-1][y-1], secondNextNode[0]])
            else:
                doubleSuccSet.append([x-1,y-1,maze[x-1][y-1]])
            succSet.append(doubleSuccSet)
    # Going Up and Right(2-Step)
    if x-1 >= 0 and y >= 0 and x-1 < N and y < N and maze[x-1][y] in nextNode:
        if x-1 >= 0 and y+1 >= 0 and x-1 < N and y+1 < N and maze[x-1][y+1] in secondNextNode:
            doubleSuccSet = []
            if maze[x-1][y] == '*' :
                doubleSuccSet.append([x-1,y,maze[x-1][y], nextNode[0]])
            else:
                doubleSuccSet.append([x-1,y,maze[x-1][y]])
            if maze[x-1][y+1] == '*' :
                doubleSuccSet.append([x-1,y+1,maze[x-1][y+1], secondNextNode[0]])
            else:
                doubleSuccSet.append([x-1,y+1,maze[x-1][y+1]])
            succSet.append(doubleSuccSet)
    return succSet

# Returns lowest heursitic step for greedy algorithm
def heuristicGreedy(type, successors, goal):
    lowestValue = []
    heuristicValue = float('inf') # Max value of float
    for i in successors:
        if (len(i) == 2):
            nextValue = i[1]
        else:
            nextValue = i
        newHeuristicValue = getHeuristicValue(type, nextValue, goal)
        # It returns best successor among all successors by comparing heuristic value from next to goal state.
        if newHeuristicValue < heuristicValue:
            heuristicValue = newHeuristicValue
            lowestValue = i
    return lowestValue

# Returns actual path cost from current node to next node
def getActualCostValue(currentNode, nextNode):
    actualCost = 0
    if len(nextNode) != 2:
        actualCost = 1 if len(nextNode) == 3 else 2
    else:
        for j in nextNode:
            actualCost = actualCost + (1 if len(j) == 3 else 2)
    return actualCost

# Returns lowest heursitic step for A* algorithm
def heuristicAstar(type, successors, current, goal):
    lowestValue = []
    heuristicValue = float('inf') # Max value of float
    for i in successors:
        if (len(i) == 2):
            nextValue = i[1]
        else:
            nextValue = i
        newHeuristicValueToEnd = getHeuristicValue(type, nextValue, goal)
        newValueFromCurrent = getActualCostValue(current, i)
        Total = newHeuristicValueToEnd + newValueFromCurrent
        # It returns best successor among all successors by comparing heuristic value sum of current to next and next to goal state value.
        if Total < heuristicValue:
            heuristicValue = Total
            lowestValue = i
    return lowestValue

# Returns lowest heursitic step for gradient descent(Hill Climbing) algorithm
def heuristicGradient(type, current, successors, goal):
    lowestValue = []
    heuristicValueFromCurrent = getHeuristicValue(type, current, goal)
    for i in successors:
        if (len(i) == 2):
            nextValue = i[1]
        else:
            nextValue = i
        heuristicValueFromSuccessor = getHeuristicValue(type, nextValue, goal)
        # It returns the successor if it is better then current node.
        if heuristicValueFromSuccessor < heuristicValueFromCurrent:
            heuristicValueFromCurrent = heuristicValueFromSuccessor
            lowestValue = i
    # If it stuck in local optima then using stochastic algorithm it tries to get best successor
    if len(lowestValue) == 0:
        heuristicValue = float('inf') # Max value of float
        for i in successors:
            if (len(i) == 2):
                nextValue = i[1]
            else:
                nextValue = i
            newHeuristicValue = getHeuristicValue(type, nextValue, goal)
            # It returns best successor among all successors by comparing heuristic value.
            if newHeuristicValue < heuristicValue:
                heuristicValue = newHeuristicValue
                lowestValue = i
    return lowestValue

# General Heuristic Function
def getHeuristicValue(type, start, end):
    # M for Manhattan and e for Euclidean
    if type == "m":
        return manhattan(start, end)
    elif type == "e":
        return euclidean(start, end)
    else:
        print("Distance type did not matched")
        exit()

# Manhattan Distance Finder
def manhattan(start, end):
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]

    return float(abs(x1-x2) + abs(y1 - y2))

# Euclidean Distance Finder
def euclidean(start, end):
    x1 = start[0]
    y1 = start[1]
    x2 = end[0]
    y2 = end[1]

    return float(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))

# Main function that calls search algorithm and shows output
def mazeSolver(matrix):
    startNode = [0,0,'a']
    lastNode = [N-1,N-1,'c']
    start = timer()
    # Calls Greedy search algorithm Function
    path = mazeSolverGreedySearch(matrix, startNode, lastNode)
    print()
    f.write("\n")
    if distanceTypeUserInput == "m":
        print("Solution using Manhattan distance \n")
        f.write("Solution using Manhattan distance \n")
    else:
        print("Solution using Euclidean distance \n")
        f.write("Solution using Euclidean distance \n")
    print("Solution of Greedy Search: %s" % (path,))
    f.write("Solution of Greedy Search: %s \n" % (path,))
    f.write("Time %s seconds \n \n"%(timer() - start))
    print("Time %s seconds"%(timer() - start))
    nodesVisited(-visitedNodes)
    pathCost((-lengthOfSolution+1))
    start = timer()
    # Calls A star algorithm Function
    path = mazeSolverAstarSearch(matrix, startNode, lastNode)
    print()
    f.write("\n")
    print("Solution of A* Search: %s" % (path,))
    f.write("Solution of A* Search: %s \n" % (path,))
    f.write("Time %s seconds \n \n"%(timer() - start))
    print("Time %s seconds"%(timer() - start))
    nodesVisited(-visitedNodes)
    pathCost((-lengthOfSolution+1))
    start = timer()
    # Calls Gradient Descent search algorithm Function
    path = mazeSolverGradientDescentSearch(matrix, startNode, lastNode)
    print()
    f.write("\n")
    print("Solution of Gradient Descent Search: %s" % (path,))
    f.write("Solution of Gradient Descent Search: %s \n" % (path,))
    f.write("Time %s seconds \n \n"%(timer() - start))
    print("Time %s seconds"%(timer() - start))
    nodesVisited(-visitedNodes)
    pathCost((-lengthOfSolution+1))
    return True

# Greedy Search Algorithm
def mazeSolverGreedySearch(matrix ,start, goal):
    stack = [start]
    visited = [start]
    path = []
    path.append(start)
    nodesVisited(1)
    # Runs until stack is not empty
    while stack:
        current = stack.pop()
        stack = []
        # If goal is matched then returns output.
        if current == goal:
            path.append(current)
            numPath = getNumbericPath(path)
            length = pathCost(0)
            visitedStates = nodesVisited(0)
            return (length, visitedStates, numPath)
        neighbors = succ(current)
        # For loop removes visited nodes from neighbors
        for i in visited:
            for j in neighbors:
                if len(j) == 2:
                    for k in j:
                        if k == i:
                            neighbors.remove(j)
                else:
                    if i == j:
                        neighbors.remove(j)
        # Get best successor node using heurisitic function
        nextNode = heuristicGreedy(distanceTypeUserInput, neighbors, goal)
        # Number of nodes visited counts
        nodesVisited(2) if len(nextNode) == 2 else nodesVisited(0) if len(nextNode) == 0 else nodesVisited(1)
        # If nextNode is empty then it remove current and backtracks
        if len(nextNode) == 0:
            path.remove(current)
            if len(path) > 0:
                stack.append(path[-1])
        # If nextNode is not empty then it adds to the path, stack and visited array
        elif len(nextNode) == 2:
            path = path + nextNode
            visited = visited + nextNode
            stack = stack + nextNode
        else:
            path.append(nextNode)
            visited.append(nextNode)
            stack.append(nextNode)
    # If path not found returns 
    return (None, nodesVisited(0), None)

# A* Search Algorithm
def mazeSolverAstarSearch(matrix ,start, goal):
    stack = [start]
    visited = [start]
    path = []
    path.append(start)
    nodesVisited(1)
    # Runs until stack is not empty
    while stack:
        current = stack.pop()
        stack = []
        # If goal is matched then returns output.
        if current == goal:
            path.append(current)
            numPath = getNumbericPath(path)
            length = pathCost(0)
            visitedStates = nodesVisited(0)
            return (length, visitedStates, numPath)
        neighbors = succ(current)
        # For loop removes visited nodes from neighbors
        for i in visited:
            for j in neighbors:
                if len(j) == 2:
                    for k in j:
                        if k == i:
                            neighbors.remove(j)
                else:
                    if i == j:
                        neighbors.remove(j)
        # Get best successor node using heurisitic function
        nextNode = heuristicAstar(distanceTypeUserInput, neighbors, current, goal)
        # Number of nodes visited counts
        nodesVisited(2) if len(nextNode) == 2 else nodesVisited(0) if len(nextNode) == 0 else nodesVisited(1)
        # If nextNode is empty then it remove current and backtracks
        if len(nextNode) == 0:
            path.remove(current)
            if len(path) > 0:
                stack.append(path[-1])
        # If nextNode is not empty then it adds to the path, stack and visited array
        elif len(nextNode) == 2:
            path = path + nextNode
            visited = visited + nextNode
            stack = stack + nextNode
        else:
            path.append(nextNode)
            visited.append(nextNode)
            stack.append(nextNode)
    # If path not found returns 
    return (None, nodesVisited(0), None)

# Gradient Descent(Hill Climbing) Search Algorithm
def mazeSolverGradientDescentSearch(matrix ,start, goal):
    stack = [start]
    visited = [start]
    path = []
    path.append(start)
    nodesVisited(1)
    # Runs until stack is not empty
    while stack:
        current = stack.pop()
        stack = []
        # If goal is matched then returns output.
        if current == goal:
            path.append(current)
            numPath = getNumbericPath(path)
            length = pathCost(0)
            visitedStates = nodesVisited(0)
            return (length, visitedStates, numPath)
        neighbors = succ(current)
        # For loop removes visited nodes from neighbors
        for i in visited:
            for j in neighbors:
                if len(j) == 2:
                    for k in j:
                        if k == i:
                            neighbors.remove(j)
                else:
                    if i == j:
                        neighbors.remove(j)
        # Get best successor node using heurisitic function
        nextNode = heuristicGradient(distanceTypeUserInput, current, neighbors, goal)
        # Number of nodes visited counts
        nodesVisited(2) if len(nextNode) == 2 else nodesVisited(0) if len(nextNode) == 0 else nodesVisited(1)
        # If nextNode is empty then it returns path not found
        if len(nextNode) == 0:
            return (None, nodesVisited(0), None)
        # If nextNode is not empty then it adds to the path, stack and visited array
        elif len(nextNode) == 2:
            path = path + nextNode
            visited = visited + nextNode
            stack = stack + nextNode
        else:
            path.append(nextNode)
            visited.append(nextNode)
            stack.append(nextNode)
    # If path not found returns 
    return (None, nodesVisited(0), None)

# Driver program to test Search algorithm
distanceTypeUserInput = "K"
if __name__ == "__main__": 
    # Initialising the maze
    while(not N > 1):
        print("\nEnter value of dimension(N)")
        N = int(input())
    while(not len(maze) == N):
        print("\nEnter list(%s) of characters from (a,b,c,*) \neg. a b a c" %(N*N))
        string = str(input())
        userInput = string.split(' ')
        if len(userInput) != N*N:
            print("\nPlease enter %s characters only" %(N*N))
            continue
        wrongInput = False
        for i in userInput:
            if i not in ['a', 'b', 'c', '*']:
                wrongInput = True
        if wrongInput:
            continue
        count = 0
        for i in range(N):
            temp = []
            for j in range(N):
                temp.append(userInput[count])
                count = count + 1
            maze.append(temp)
        while distanceTypeUserInput != "m" and distanceTypeUserInput != "e":
            print("\nEnter Distance Type \n    Type 'm' For Manhattan \n    Type 'e' For Euclidean")
            distanceTypeUserInput = str(input())[0]
    f= open("hw1Part2Output.txt","a+")
    internalRepresentation()
    mazeSolver(matrix)
    f.write("--------------------------------------------------\n\n")
    f.close()
    


















