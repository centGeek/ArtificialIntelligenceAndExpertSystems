"""Generator obrazow pojedynczych liter o zadanym poziomie zaszumienia.

Uzycie zgodne ze specyfikacja zadania projektowego:

    font_generator.py w h x y font_file letter noise_level output_directory

gdzie:
    w, h              - szerokosc i wysokosc generowanego obrazu
    x, y              - pozycja litery na obrazie
    font_file         - plik ttf z krojem czcionki
    letter            - litera do narysowania
    noise_level       - poziom zaszumienia w procentach (0 = obraz czysty)
    output_directory  - katalog docelowy

Opcjonalny dziewiaty argument podaje rozmiar czcionki w punktach. Specyfikacja
go nie wymienia, wiec domyslnie jest wyliczany z wysokosci obrazu.
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from madaline import BINARY_THRESHOLD, MadalineError, append_description

USAGE = ("Uzycie: font_generator.py w h x y font_file letter noise_level "
         "output_directory [font_size]")


def find_font(font_file):
    """Szuka pliku czcionki w katalogu roboczym, a potem w katalogach systemowych."""
    candidates = [font_file,
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), font_file),
                  os.path.join("/System/Library/Fonts/Supplemental", font_file),
                  os.path.join("/System/Library/Fonts", font_file),
                  os.path.join("/Library/Fonts", font_file),
                  os.path.join(os.path.expanduser("~/Library/Fonts"), font_file),
                  os.path.join("C:\\Windows\\Fonts", font_file)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise MadalineError(
        f"Nie znaleziono pliku czcionki {font_file}. Szukano w katalogu roboczym "
        f"i w katalogach systemowych z czcionkami.")


def generate(width, height, x, y, font_file, letter, noise_level, output_directory,
             font_size=None, seed=None):
    """Tworzy binarny obraz litery, zaszumia go i zapisuje razem z linia opisu."""
    if width <= 0 or height <= 0:
        raise MadalineError("Szerokosc i wysokosc obrazu musza byc dodatnie.")
    if not 0 <= noise_level <= 100:
        raise MadalineError("Poziom zaszumienia musi byc z zakresu 0..100 procent.")
    if len(letter) != 1:
        raise MadalineError(f"Parametr letter ma byc pojedynczym znakiem, a jest '{letter}'.")

    font_path = find_font(font_file)
    if font_size is None:
        font_size = max(8, int(height * 0.75))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError as error:
        raise MadalineError(f"Nie udalo sie wczytac czcionki {font_path}: {error}")

    # biale tlo, czarny tusz - zgodnie z konwencja obrazow skanowanych
    image = Image.new("L", (width, height), 255)
    ImageDraw.Draw(image).text((x, y), letter, font=font, fill=0)

    pixels = np.asarray(image, dtype=np.int16)
    binary = np.where(pixels < BINARY_THRESHOLD, 0, 255).astype(np.uint8)

    if noise_level > 0:
        generator = np.random.default_rng(seed)
        flip = generator.random(binary.shape) <= noise_level / 100.0
        binary = np.where(flip, 255 - binary, binary).astype(np.uint8)

    os.makedirs(output_directory, exist_ok=True)
    file_name = f"{letter}.png"
    Image.fromarray(binary, mode="L").save(os.path.join(output_directory, file_name))
    append_description(output_directory, file_name,
                       f"letter {letter}, noise level {noise_level}%")
    return os.path.join(output_directory, file_name)


def main(argv):
    if not 8 <= len(argv) <= 9:
        print(USAGE)
        return 1
    try:
        width, height, x, y = (int(argv[0]), int(argv[1]), int(argv[2]), int(argv[3]))
        noise_level = int(argv[6])
        font_size = int(argv[8]) if len(argv) == 9 else None
    except ValueError:
        print("Parametry w, h, x, y, noise_level i font_size musza byc liczbami calkowitymi.")
        print(USAGE)
        return 1

    try:
        path = generate(width, height, x, y, argv[4], argv[5], noise_level, argv[7],
                        font_size)
    except MadalineError as error:
        print(f"Blad: {error}")
        return 1
    print(f"Zapisano {path} (litera '{argv[5]}', zaszumienie {noise_level}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
