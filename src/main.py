from pathlib import Path
import re
from Programme1 import autoValidPresences
from functions.miscellaneous import ensure_clean_dir, ensure_dir

#parameters
DATA_ROOT = Path("data")
STUDENT_CLASS_SIGNATURES = DATA_ROOT / "STUDENT_CLASS_SIGNATURES"
PROGRAMME1_RESULTS = Path("outputs") / "programme1"


def form_sort_key(path):
    match = re.search(r"\d+", path.name)
    if match:
        return int(match.group())
    return 10**9


def list_form_presence_dirs(data_root):
    #cherche les dossier de presences
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dossier data introuvable : {data_root}")

    form_dirs = [
        p for p in data_root.iterdir()
        if p.is_dir() and re.fullmatch(r"FORM\d+", p.name)
    ]

    form_dirs.sort(key=form_sort_key)

    for form_dir in form_dirs:
        presence_dir = form_dir / f"EXAM_{form_dir.name}_PRESENCES"

        if presence_dir.is_dir():
            yield presence_dir
        else:
            print(f"dossier de présence introuvable : {presence_dir}")


def main():
    ensure_dir(PROGRAMME1_RESULTS)
    ensure_clean_dir(PROGRAMME1_RESULTS)

    presence_dirs = list(list_form_presence_dirs(DATA_ROOT))

    if len(presence_dirs) == 0:
        print("[ERROR] Aucun dossier EXAM_FORMXX_PRESENCES trouvé.")
        return

    for presence_dir in presence_dirs:
        autoValidPresences(
            EXAM_FORMXX_PRESENCES=str(presence_dir),
            STUDENT_CLASS_SIGNATURES=str(STUDENT_CLASS_SIGNATURES),
            EXAM_FORMXX_RESULTS=str(PROGRAMME1_RESULTS),
        )

    print("\nTous les fichiers Programme 1 ont été générés dans :")
    print(PROGRAMME1_RESULTS)


if __name__ == "__main__":
    main()