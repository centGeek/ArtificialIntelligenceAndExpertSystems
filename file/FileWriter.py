from Algorithms.Bfs import bfs
from Algorithms.Dfs import dfs
from Algorithms.AStar import aStarPuzzle
class FileWriter():
    
    def save_to_file(strategy, puzzle, direction, outputFile, stats):
        if (strategy == "bfs"):
            sol = bfs(puzzle, direction)

            with open(outputFile, 'w') as f:
                print(sol[0], file=f)
                print(sol[1], file=f)

            with open(stats, 'w') as f:
                print(sol[0], file=f)
                print(sol[2], file=f)
                print(sol[3], file=f)
                print(sol[4], file=f)   
                # print(sol[5], file=f)  

        elif (strategy == "dfs"):
            sol = dfs(puzzle, direction)

            with open(outputFile, 'w') as f:
                print(sol[0], file=f)
                print(sol[1], file=f)

            with open(stats, 'w') as f:
                print(sol[0], file=f)
                print(sol[2], file=f)
                print(sol[3], file=f)
                print(sol[4], file=f)   
                # print(sol[5], file=f)  

        elif (strategy == "astr"):
            sol = aStarPuzzle(puzzle,direction)

            with open(outputFile, 'w') as f:
                print(sol[0], file=f)
                print(sol[1], file=f)

            with open(stats, 'w') as f:
                print(sol[0], file=f)
                print(sol[2], file=f)
                print(sol[3], file=f)
                print(sol[4], file=f)  
                print(sol[5], file=f)
