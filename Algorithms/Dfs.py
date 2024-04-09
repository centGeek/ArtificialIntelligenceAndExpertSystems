import copy
import collections
import time

def dfs(puzzle, the_way):
    MaxDepth = 20  # definiujemy maksymalną głębokość rekursji

    stack = collections.deque() # inicjalizacja stosu
    start = copy.deepcopy(puzzle) # operujemy na kopii
    iteration = 1
    stack.append(start)
    start = time.time() 
    visited = {
        puzzle.hashState(): puzzle.deep
    }

    while stack: # warunek zawsze spelniony, stos nigdy nie jest pusty
        iteration += 1 #liczba
        current = stack.pop()
        if current.deep == MaxDepth:
            continue

        if current.check():
            end = time.time()
            return len(current.path), current.path, len(visited), iteration, current.deep, round((end - start) * 1000)

        for i in range(4):
            move_way(current, the_way[i], visited, stack)

def MoveAndCheck(array, direction, visited, stack):
        temp = copy.deepcopy(array)
        temp.move(direction)
        if temp.hashState() in visited and visited[temp.hashState()] > temp.deep or temp.hashState() not in visited:
            visited[temp.hashState()] = array.deep
            temp.lastmove = direction
            temp.path += direction
            temp.deep += 1
            stack.append(temp)

def move_way(array, direction, visited, stack):
        if direction == "U":
            if array.findIndexOfNum(0)[0] != 0 and array.lastmove != "D":
                MoveAndCheck(array, "U",visited, stack)
        elif direction == "D":
            if array.findIndexOfNum(0)[0] != array.rows - 1 and array.lastmove != "U":
                MoveAndCheck(array, "D",visited, stack)
        elif direction == "R":
            if array.findIndexOfNum(0)[1] != array.columns - 1 and array.lastmove != "L":
                MoveAndCheck(array, "R",visited, stack)
        elif direction == "L":
            if array.findIndexOfNum(0)[1] != 0 and array.lastmove != "R":
                MoveAndCheck(array, "L",visited, stack)
        return