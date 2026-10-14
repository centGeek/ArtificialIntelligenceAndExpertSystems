import os

# Katalog projektu wyznaczony z polozenia tego pliku, a nie z katalogu roboczego.
# Wczesniej sciezki byly wzgledne ("./zadanie2/..."), wiec program dzialal tylko
# uruchomiony z katalogu nadrzednego, a odpalony z wnetrza zadanie2/ tworzyl
# zagniezdzony katalog zadanie2/zadanie2/files zamiast znalezc wlasciwy.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FILES_DIR = os.path.join(PROJECT_DIR, "files")

PATH_TRAIN_DATA = os.path.join(DATA_DIR, "train_data.csv")
PATH_TEST_DATA = os.path.join(DATA_DIR, "test_data.csv")
PATH_ASOCJACJA = os.path.join(DATA_DIR, "autoasocjacja.csv")

PATH_TEST_LOGGING = os.path.join(FILES_DIR, "test_logging.log")
PATH_STATS = os.path.join(FILES_DIR, "statistics.log")
PATH_SAVED = os.path.join(FILES_DIR, "siec.pkl")
PATH_TRAIN_EFFECTS = os.path.join(FILES_DIR, "training_effectiveness_info.txt")
