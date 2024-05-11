import sys
sys.path.append('..')
from zadanie1.BoardState import BoardState
from file.FileWriter import FileWriter

if len(sys.argv) > 4: 
    strategy = sys.argv[1]
    direction = sys.argv[2]
    inputFile = sys.argv[3]
    outputFile = sys.argv[4]
    stats = sys.argv[5]

def loadPuzzle():
    puzzle = []
    file = open(inputFile, 'r')
    temp = file.readline()
    cols = temp.split(" ")[0]
    rows = temp.split(" ")[1].split()[0]

    for line in file.readlines():
        numbers = line.strip().split()
        numbers = [int(number) for number in numbers]
        puzzle.append(numbers)

    return rows, cols, puzzle

puzzle = BoardState(*loadPuzzle())
FileWriter.save_to_file(strategy, puzzle, direction, outputFile, stats)

