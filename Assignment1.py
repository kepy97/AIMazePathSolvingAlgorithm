# Homework 1 Artificial Intelligence kvpatel
from collections import deque
import queue as queue
from timeit import default_timer as timer

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
    matrixToGraph()

# A utility function to print internal representation as a square maze
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

# Cost of path incremented
lengthOfSolution = 0
def cost(v):
    global lengthOfSolution
    lengthOfSolution += v

# Total cost from initial state to final state
def pathCost():
    global lengthOfSolution
    return lengthOfSolution + 1

# Position of element returns
def getKey(data):
    keys = []
    for i in data:
        for index, value in matrix.items():
            if value == i[:3]:
                keys.append(index)
                break
    return keys

# It returns graph from matrix
graph = {}
def matrixToGraph():
    for i in range(1,N*N+1):
        temp = succ(matrix[i])# if matrix[i][2] != '*' else []
        graph[i] = getKey(temp)
    #print(matrix)

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
            length = len(data) - 2
            continue
        if ((current[0] - next[0]) == 0 and (current[1] - next[1]) in [1, -1]) or ((current[1] - next[1]) == 0 and (current[0] - next[0]) in [1, -1]):
            pass
        else:
            del data[length]
            length = len(data) - 2
            continue
        length = length - 1
    return getKey(data)

# This function returns set of states that can be reached from current state.
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
    if currentNode == 'b':
        nextNode = succOfB
    if currentNode == 'c':
        nextNode = succOfC
    # Going Up
    if x-1 >= 0 and y >= 0 and x-1 < N and y < N and maze[x-1][y] in nextNode:
        if maze[x-1][y] == '*' :
            succSet.append([x-1,y,maze[x-1][y], nextNode[0]])
        else:
            succSet.append([x-1,y,maze[x-1][y]])
    # Going Down
    if x+1 >= 0 and y >= 0 and x+1 < N and y < N and maze[x+1][y] in nextNode:
        if maze[x+1][y] == '*' :
            succSet.append([x+1,y,maze[x+1][y], nextNode[0]])
        else:
            succSet.append([x+1,y,maze[x+1][y]])
    # Going Left
    if x >= 0 and y-1 >= 0 and x < N and y-1 < N and maze[x][y-1] in nextNode:
        if maze[x][y-1] == '*' :
            succSet.append([x,y-1,maze[x][y-1], nextNode[0]])
        else:
            succSet.append([x,y-1,maze[x][y-1]])
    # Going Right
    if x >= 0 and y+1 >= 0 and x < N and y+1 < N and maze[x][y+1] in nextNode:
        if maze[x][y+1] == '*' :
            succSet.append([x,y+1,maze[x][y+1], nextNode[0]])
        else:
            succSet.append([x,y+1,maze[x][y+1]])
    return succSet

# Main function that calls search algorithm and shows output
def mazeSolver(graph):
    startNode = [0,0,'a']
    lastNode = [N-1,N-1,'c']
    start = timer()
    path = mazeSolverBFS(matrix, startNode, lastNode)
    print()
    f.write("\n")
    print("Solution of Breath-First Search: %s" % (path,))
    f.write("Solution of Breath-First Search: %s \n" % (path,))
    f.write("Time %s seconds \n \n"%(timer() - start))
    print("Time %s seconds"%(timer() - start))
    cost(-lengthOfSolution)
    start = timer()
    path = mazeSolverDFS(matrix, startNode, lastNode)
    print()
    print("Solution of Depth-First Search: %s" % (path,))
    print("Time %s seconds"%(timer() - start))
    f.write("Solution of Depth-First Search: %s \n" % (path,))
    f.write("Time %s seconds \n \n"%(timer() - start))
    cost(-lengthOfSolution)
    start = timer()
    path = mazeSolverUCS(graph, 1, (N*N))
    print()
    print("Solution of Uniform Cost Search: %s" % (path,))
    print("Time %s seconds"%(timer() - start))
    f.write("Solution of Uniform Cost Search: %s \n" % (path,))
    f.write("Time %s seconds \n \n"%(timer() - start))
    cost(-lengthOfSolution)
    start = timer()
    path = mazeSolverIterativeDeepening(matrix, startNode, lastNode)
    print()
    print("Solution of Iterative Deepening: %s" % (path,))
    print("Time %s seconds"%(timer() - start))
    print("Output is also stored in hw1Output.txt file")
    f.write("Solution of Iterative Deepening: %s \n" % (path,))
    f.write("Time %s seconds \n \n"%(timer() - start))
    return True

# BFS Algorithm
def mazeSolverBFS(matrix ,start, goal):
    queue = deque([start])
    visited = [start]
    path = []
    path.append(start)
    while queue:
        current = queue.popleft()
        if current == goal:
            return path.append(goal)
        for neighbor in succ(current):
            cost(1)
            if neighbor == goal:
                path.append(neighbor)
                numPath = getNumbericPath(path)
                length = len(numPath)
                visitedStates = pathCost()
                return (length, visitedStates, numPath)
            if neighbor in visited:
                continue
            visited.append(neighbor)
            queue.append(neighbor)
            path.append(neighbor)
    return (None, pathCost(), None)

# DFS Algorithm
def mazeSolverDFS(matrix, start, goal):
    stack = [start]
    visited = [start]
    path = []
    path.append(start)
    while stack:
        current = stack.pop()
        if current == goal:
            return path.append(goal)
        for neighbor in succ(current)[::-1]:
            if neighbor == goal:
                path.append(neighbor)
                numPath = getNumbericPath(path)
                length = len(numPath)
                visitedStates = pathCost()
                return (length, visitedStates, numPath)
            cost(1)
            if neighbor in visited:
                continue
            visited.append(neighbor)
            stack.append(neighbor)
            path.append(neighbor)
    return (None, pathCost(), None)

# UCS Algorithm
def mazeSolverUCS(graph, start, goal):
    visited = set()                  
    q = queue.PriorityQueue()        
    q.put((1, start, [start]))
    while not q.empty():  
        f, current_node, path = q.get()
        visited.add(current_node)
        if current_node == goal:
            length = len(path)
            visitedStates = pathCost()
            return (length, visitedStates, path)
        else:
            cost(1)
            for edge in graph[current_node]:
                if edge not in visited:
                    q.put((1, edge, path + [edge]))
    return (None, pathCost(), None)

# Iterative Deepning Algorithm
boolID = True
depthID = 0
def mazeSolverIterativeDeepening(matrix, start, goal):
    depth = 1
    global boolID
    global depthID
    while boolID:
        result = mazeSolverDepthLimited(matrix, start, goal, depth)
        depth = depth + 1
        if depth == 200:
            boolID = False
            result = (None, pathCost(), None)
    depthID = depth
    return result
def mazeSolverDepthLimited(matrix, start, goal, depth):
    global boolID
    leaves = queue.LifoQueue()
    d = 0
    leaves.put(start)
    path = []
    while True:
        if leaves.empty():
            return (None, pathCost(), None)
        actual = leaves.get()
        path.append(actual)
        if actual == goal:
            boolID = False
            length = len(path)
            return (length, d, getNumbericPath(path))
        elif d is not depth:
            d += 1
            child = queue.LifoQueue()
            temp = succ(actual)
            for i in temp:
                child.put(i)
            while not child.empty():
                leaves.put(child.get())

# Driver program to test Search algorithm
if __name__ == "__main__": 
    # Initialising the maze
    while(not N > 1):
        print("Enter value of dimension(N)")
        N = int(input())
    while(not len(maze) == N):
        print("Enter list(%s) of characters from (a,b,c,*) \neg. a b a c" %(N*N))
        string = str(input())
        userInput = string.split(' ')
        if len(userInput) != N*N:
            print("Please enter %s characters only" %(N*N))
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
    f= open("hw1Output.txt","a+")
    internalRepresentation()
    mazeSolver(graph)
    f.write("--------------------------------------------------\n\n")
    f.close()
    


















