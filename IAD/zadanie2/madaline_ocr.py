"""Rozpoznawanie znakow siecia MADALINE.

Uzycie zgodne ze specyfikacja zadania projektowego:

    madaline_ocr.py train_directory test_directory

Oba katalogi musza zawierac pliki png ze znakami oraz plik description.txt
wygenerowany przez font_generator.py. Ze znakow z train_directory budowana
jest siec, a znaki z test_directory sa rozpoznawane.
"""

import sys

from madaline import MadalineError, MadalineNetwork, image_to_vector, read_description

USAGE = "Uzycie: madaline_ocr.py train_directory test_directory"


def recognize_directory(train_directory, test_directory):
    """Buduje siec ze wzorcow i zwraca liste wynikow rozpoznawania."""
    network = MadalineNetwork.from_directory(train_directory)
    results = []
    for path, label in read_description(test_directory):
        vector, _ = image_to_vector(path)
        recognized, confidence = network.recognize(vector)
        results.append((label, recognized, confidence))
    return results


def main(argv):
    if len(argv) != 2:
        print(USAGE)
        return 1
    try:
        results = recognize_directory(argv[0], argv[1])
    except MadalineError as error:
        print(f"Blad: {error}")
        return 1
    for label, recognized, confidence in results:
        print(f"{label} -> {recognized}, confidence: {confidence:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
