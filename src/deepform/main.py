"""MAIN - orchestrates Programme 1 and Programme 2 (project section 3.6).
 
Define the exam name and the signatures directory; everything else is derived.
Creates the results directory and runs autoValidPresences then autoReadForm.
"""
 
import os
from os.path import join
 
from .program1 import autoValidPresences
from .program2 import autoReadForm
 
 
def run(exam_name, signatures_dir, data_root="data_exemple"):
    """exam_name e.g. 'FORM3'.  Folder layout follows the project statement.
 
    data_root : racine contenant les dossiers d'examen.
                Par défaut 'data_exemple' (structure du dépôt Git).
                Pour le challenge, passer le chemin du vrai répertoire.
    """
    form_root = join(data_root, exam_name)
    pdf_dir = join(form_root, f"EXAM_{exam_name}_PDF")
    presences_dir = join(form_root, f"EXAM_{exam_name}_PRESENCES")
    results_dir = join(form_root, f"EXAM_{exam_name}_RESULTS")
    os.makedirs(results_dir, exist_ok=True)
 
    print("=" * 60)
    print(f"Exam        : {exam_name}")
    print(f"Data root   : {data_root}")
    print(f"PDF dir     : {pdf_dir}")
    print(f"Presences   : {presences_dir}")
    print(f"Signatures  : {signatures_dir}")
    print(f"Results dir : {results_dir}")
    print("=" * 60)
 
    print("\n--- PROGRAMME 1 : autoValidPresences ---")
    autoValidPresences(presences_dir, signatures_dir, results_dir)
 
    print("\n--- PROGRAMME 2 : autoReadForm ---")
    autoReadForm(pdf_dir, signatures_dir, results_dir)
 
    print("\nDone. Results in:", results_dir)
    return results_dir
 
 
if __name__ == "__main__":
    # Lancement depuis src/ :
    #   python -m deepform.main
    #
    # Pour le challenge, adapter les deux paramètres ci-dessous :
    run(
        exam_name="FORM3",
        signatures_dir="data_exemple/STUDENT_CLASS_SIGNATURES",
        data_root="data_exemple",
    )