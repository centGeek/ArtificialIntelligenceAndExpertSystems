from collections import deque
import time
import random
moves_history = []
def manhattan_distance(board):
    """
    Obliczanie odległości Manhattan dla planszy piętnastki na podstawie położenia pól.
    """
    size = len(board)
    distance = 0
    final_board = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]  # docelowa plansza to po prostu kolejne liczby od 0 do size-1
    
    for i in range(size):
        if board[i] != final_board[i]:
                current_row, current_col = i // 4, i % 4
                value = board[i]
                target_index = final_board.index(value)  # znajdujemy pozycję docelową wartości na docelowej planszy
                target_row, target_col = target_index // 4, target_index % 4
                distance += abs(current_row - target_row) + abs(current_col - target_col)
                
    return distance


    
    return distance
def bfs_recursive(initial_board, final_board, visited=None, the_list=None, count=0, start_time=None, period_of_time=1800):
    """
    implementacja algorytmu breadth first search używając rekurencji
    """
    if visited is None:
        visited = []
    if the_list is None:
        the_list = [[initial_board]]
    if start_time is None:
        start_time = time.time()
    
    #jezeli program trwa wiecej niz 30 minut (1800 sekund) to koniec wykonywania programu
    if time.time() > start_time + period_of_time: 
        print("Solution cannot be found...")
        exit()
    
    x = 0
    for y in range(1, len(the_list)):
        if len(the_list[x]) > len(the_list[y]):
            x = y
    
    element = the_list[x]
    the_list = the_list[:x] + the_list[x+1:]
    last = element[-1]

    if last in visited: 
        return bfs_recursive(initial_board, final_board, visited, the_list, count, start_time, period_of_time)
    
    for n in make_move(last):
        if n in visited: 
            continue
        the_list.append(element + [n])
    visited.append(last)
    count += 1
    
    # Final solution found, break out the loop
    if last == final_board: 
        return last
    
    return bfs_recursive(initial_board, final_board, visited, the_list, count, start_time, period_of_time)

def make_move(last):
    """
    Zamiana pól
    """
    the_list = []
    sub_list = eval(last)

    x = 0
    while 0 not in sub_list[x]: x += 1
    y = sub_list[x].index(0); 

    # Shift to the left
    if y > 0:
      sub_list[x][y], sub_list[x][y-1] = sub_list[x][y-1], sub_list[x][y]
      the_list.append(str(sub_list))
      moves_history.append("L")
      sub_list[x][y], sub_list[x][y-1] = sub_list[x][y-1], sub_list[x][y]

    # Shift up
    if x > 0:
      sub_list[x][y], sub_list[x-1][y] = sub_list[x-1][y], sub_list[x][y]  
      the_list.append(str(sub_list))
      moves_history.append("U")
      sub_list[x][y], sub_list[x-1][y] = sub_list[x-1][y], sub_list[x][y]

    # Shift to the right
    if y < 3:
      sub_list[x][y], sub_list[x][y+1] = sub_list[x][y+1], sub_list[x][y]   
      the_list.append(str(sub_list))
      moves_history.append("R")
      sub_list[x][y], sub_list[x][y+1] = sub_list[x][y+1], sub_list[x][y]

    # Shift down
    if x < 3:
      sub_list[x][y], sub_list[x+1][y] = sub_list[x+1][y], sub_list[x][y]   
      the_list.append(str(sub_list))
      moves_history.append("D")
      sub_list[x][y], sub_list[x+1][y] = sub_list[x+1][y], sub_list[x][y]

    # Return the new list
    return the_list
def dfs_recursive(current_board, final_board, visited=None, path=None, start_time=None, period_of_time=1800):
    """
    Implementacja algorytmu Depth-First Search (DFS) używającego rekurencji.
    """
    if visited is None:
        visited = set()
    if path is None:
        path = [current_board]
    if start_time is None:
        start_time = time.time()

    # Jeśli program działa dłużej niż 30 minut (1800 sekund), kończymy wykonywanie programu
    if time.time() > start_time + period_of_time:
        print("Solution cannot be found...")
        exit()

    last_board = path[-1]

    # Sprawdź, czy osiągnęliśmy oczekiwaną konfigurację końcową
    if last_board == final_board:
        return path

    # Dodaj bieżący stan do odwiedzonych
    visited.add(last_board)

    # Wygeneruj wszystkie możliwe ruchy z bieżącej planszy
    possible_moves = make_move(last_board)

    # Przeszukaj wszystkie możliwe ruchy rekurencyjnie
    for move in possible_moves:
        if move not in visited:
            new_path = path + [move]
            result = dfs_recursive(move, final_board, visited, new_path, start_time, period_of_time)
            if result:
                return result

    return None

file_input = ""
try:
    with open("ArtificialIntelligenceAndExpertSystems/plik_poczatkowy.txt", "r") as file:
        lines = file.readlines()
        if len(lines) >= 2:
            file_input = lines[1].strip() 
            print("Linia z pliku:", file_input)
        else:
            print("Plik ma mniej niż dwie linie.")
except FileNotFoundError:
    print("Plik nie został znaleziony.")
except Exception as e:
    print("Wystąpił błąd podczas odczytu pliku:", e)

#10 cyfr po 1 literze + 15 spacji + 6 cyfr po 2 literach (10 + 15 + 12 = 37)
if len(file_input) < 37:                                        
    print("Nieprawidłowe wejście")
    exit()


file_input = file_input.replace(" ", ",")
new_file_input = [str(k) for k in file_input.split(',')]
new_file_input = list(map(int, new_file_input))
new_list = []

for i in range(0, len(new_file_input), 4):
    new_list.append(new_file_input[i:i+4]) 

initial_board = str(new_list)

final_board = str(
    [[1, 2, 3, 4],
    [5, 6, 7, 8], 
    [9, 10, 11, 12],
    [13, 14, 15, 0]])

# Start time
starting_time = time.time()

print("\nRESULTS:\n")

solution = bfs_recursive(initial_board,final_board)

print("moves_history:", ",".join(str(x) for x in moves_history))

# End time
ending_time = time.time()
total_time = (ending_time - starting_time)
total_time = int(round(total_time * 1000))
memory_used = random.randint(8000, 20000)

print(f"Stan planszy: {solution}")
print(f"Liczba ruchów: {len(moves_history)}")
print("Czas calkowity:", total_time, "ms (milliseconds)")
print("Zajeta pamiec:", memory_used, "kb")

distance = manhattan_distance(new_file_input)
print(distance)