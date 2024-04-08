import collections
import copy
import time
import sys

sys.setrecursionlimit(2000000)
start = time.time()
def bfs(puzzle, directions, queue = None, visited = None):
    if queue is None:
        queue = collections.deque()
    if visited is None:
        visited = set()
    queue.append(puzzle)
    current = queue.popleft()
    if current.check():
        end = time.time()
        return len(current.path), current.path, len(visited), current.deep, round((end - start), 3)
    
    for i in range(4): # 4 bo tyle jest porzadkow przeszukiwania sasiedztwa: L,R,U,D
        move_way(current, directions[i], queue, visited)
    return bfs(puzzle, directions, queue, visited)

def MoveAndCheck(array, direction, queue, visited): # operujemy głównie na kolejce
    temp = copy.deepcopy(array) # tworzymy kopię tablicy
    temp.move(direction) 
    if temp.hashState() not in visited:
        visited.add(temp.hashState())
        temp.path += direction
        temp.deep += 1
        queue.append(temp)

def move_way(array, direction, queue, visited):
    if direction == "U":
        if array.findIndexOfNum(0)[0] != 0 and array.lastmove != "D":
            MoveAndCheck(array, "U", queue, visited)
    elif direction == "D":
        if array.findIndexOfNum(0)[0] != array.rows - 1 and array.lastmove != "U":
            MoveAndCheck(array, "D", queue, visited)
    elif direction == "R":
        if array.findIndexOfNum(0)[1] != array.columns - 1 and array.lastmove != "L":
            MoveAndCheck(array, "R", queue, visited)
    elif direction == "L":
        if array.findIndexOfNum(0)[1] != 0 and array.lastmove != "R":
            MoveAndCheck(array, "L", queue, visited)
    return
