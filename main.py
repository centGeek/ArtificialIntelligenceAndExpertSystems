from collections import deque
moves_history = []

def bfs(start_state):
    queue = deque([(start_state, [])])  # Kolejka stanów do sprawdzenia, z pustą ścieżką dla stanu początkowego
    while queue:
        current_state, path = queue.popleft()
        
        if is_goal_state(current_state):
            return current_state
        
        for move in possible_moves(current_state):
            new_state = make_move(current_state, move)
            new_path = path + [move]
            queue.append((new_state, new_path))
            
    return queue
def is_goal_state(state):
    goal_state = [[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 0]]
    return state == goal_state

def possible_moves(state):
    moves = []
    empty_row, empty_col = find_empty_tile(state)
    
    if empty_row > 0:
        moves.append('up')
    if empty_row < 3:
        moves.append('down')
    if empty_col > 0:
        moves.append('left')
    if empty_col < 3:
        moves.append('right')
    
    return moves

def find_empty_tile(state):
    for row in range(4):
        for col in range(4):
            if state[row][col] == 0:
                return row, col

def make_move(state, move):
    row, col = find_empty_tile(state)
    new_state = [row[:] for row in state]
    if move == 'up':
        previous_first = new_state[row][col]
        new_state[row][col] = new_state[row - 1][col]
        previous_second = new_state[row - 1][col]
        new_state[row - 1][col] = 0
        moves_history.append(f"Wykonano ruch: {move} zamieniając: {previous_first} z {previous_second}")
    elif move == 'down':
        previous_first = new_state[row][col]
        new_state[row][col] = new_state[row + 1][col]
        previous_second = new_state[row+1][col]
        new_state[row + 1][col] = 0
        moves_history.append(f"Wykonano ruch: {move} zamieniając: {previous_first} z {previous_second}")
    elif move == 'left':
        previous_first = new_state[row][col]
        new_state[row][col] = new_state[row][col - 1]
        previous_second = new_state[row][col-1]
        new_state[row][col - 1] = 0
        moves_history.append(f"Wykonano ruch: {move} zamieniając: {previous_first} z {previous_second}")
    elif move == 'right':
        previous_first = new_state[row][col]
        new_state[row][col] = new_state[row][col + 1]
        previous_second = new_state[row][col+1]
        new_state[row][col + 1] = 0
        moves_history.append(f"Wykonano ruch: {move} zamieniając: {previous_first} z {previous_second}")
    return new_state

start_state = [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 0, 15]]

xd = bfs(start_state)
for moves in moves_history:
    print(moves)
print("Solution:", xd)
