from pathlib import Path
from os import path
from predict_similarity import predict_signature_from_box, load_similarity_model
from functions.debugging_programme1 import draw_results
from functions.id_grid_pipeline import read_student_id_grid_from_path
from functions.labels import find_regions
from functions.outils_generaux_images import load_binary_image, load_gray_image
from functions.signature_box_pipeline import find_signature_box, detect_if_the_signature_is_there
from functions.outils_format_XLSX import append_presence_row, create_presence_workbook

#parameters
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", "JPG", ".png", ".tif", ".tiff"}
SIGNATURE_MIN_INK_RATIO = 0.006
SIGNATURE_SVM_THRESHOLD = 0.0
SAVE_DEBUG_IMAGES = False
MODEL_DIR = "signature_model_similarity"
MODEL_PATH = path.join(MODEL_DIR, "signature_similarity.joblib")

def autoValidID(filename_jpg, STUDENT_CLASS_SIGNATURES, EXAM_FORMXX_PRESENCES_xlsx, EXAM_FORMXX_RESULTS, model):
    ###############################################
    #definition des paths
    
    img_path = Path(filename_jpg)
    xlsx_path = Path(EXAM_FORMXX_PRESENCES_xlsx)
    results_dir = Path(EXAM_FORMXX_RESULTS)
    image_name = img_path.name
    student_id_grid = None
    student_id_signature = None
    signature_score = 0.0
    signature_box = None
    
    ##############################################
    #chargement de l'image
    #dans le cas d'une erreur lors du chargement de l'image on return juste l'image dans le xlsx
    
    try:
        img_gray = load_gray_image(str(img_path))
        img_binary = load_binary_image(str(img_path))
    except:
        append_presence_row(xlsx_path=xlsx_path, image_name=image_name, student_id_grid=None, student_id_signature=None)
        return {
            "imageName": image_name,
            "studentID_grid": None,
            "studentID_signature": None,
            "signature_score": 0.0,
            "status": "image_error",
        }

    ##############################################
    #lecture de l'id de l'étudiant via la grille
    
    id_result = {
        "student_id": None,
        "grid": None,
        "ratios": None,
        "status": "not_run",
    }

    try:
        id_result = read_student_id_grid_from_path(str(img_path))
        student_id_grid = id_result.get("student_id")
    except:
        student_id_grid = None
        id_result["status"] = "grid_error"

    ##############################################
    #detection de la presence de la boite de signature et de la signature
    
    has_signature = False
    ink_ratio = 0.0

    try:
        items_regions = find_regions(img_binary)
        signature_box = find_signature_box(items_regions, img_binary.shape)
        has_signature, ink_ratio = detect_if_the_signature_is_there(img_binary, signature_box, SIGNATURE_MIN_INK_RATIO)
    except:
        signature_box = None
        has_signature = False
        ink_ratio = 0.0

    ##############################################
    #reconnaissance de la signature
    
    if has_signature and model is not None and student_id_grid not in [None, ""]:
        student_id_signature, signature_score = predict_signature_from_box(
            img_gray=img_gray,
            signature_box=signature_box,
            model=model,
            student_id_grid=student_id_grid
        )
    else:
        student_id_signature = None
        signature_score = 0.0

    ##############################################
    #debug (save les images qui posent problèmes avec la grid créé et la boite de signature détecté et recrée)
    
    if SAVE_DEBUG_IMAGES:
        try:
            debug_dir = results_dir / "debug_programme1" / img_path.parent.name
            debug_dir.mkdir(parents=True, exist_ok=True)
            grid_info = {
                "grid": id_result.get("grid"),
                "ratios": id_result.get("ratios"),
                "status": id_result.get("status"),
            }
            out_path = debug_dir / image_name
            draw_results(img_gray, signature_box, grid_info, str(out_path))
        except Exception as e:
            print(f"[WARNING] Impossible de sauvegarder le debug pour {image_name} :", e)
            
    ##############################################
    #ecriture dans le fichier xlsx

    append_presence_row(xlsx_path=xlsx_path, image_name=image_name, student_id_grid=student_id_grid, student_id_signature=student_id_signature)
    
    ##############################################
    #return (pour les statistiques)
    
    return {
        "imageName": image_name,
        "studentID_grid": student_id_grid,
        "studentID_signature": student_id_signature,
        "signature_score": signature_score,
        "has_signature": has_signature,
        "ink_ratio": ink_ratio,
        "grid_status": id_result.get("status"),
    }



def autoValidPresences(EXAM_FORMXX_PRESENCES, STUDENT_CLASS_SIGNATURES, EXAM_FORMXX_RESULTS):
    ##############################################
    #path et directory management
    presences_dir = Path(EXAM_FORMXX_PRESENCES)
    results_dir = Path(EXAM_FORMXX_RESULTS)

    if not presences_dir.is_dir():
        raise FileNotFoundError(f"Répertoire de présences introuvable : {presences_dir}")
    
    xlsx_path = results_dir / f"{presences_dir.name}.xlsx"
    
    ###############################################
    #lancement du programme, création du xlsx et loading du model(afin d'éviter de le load à chaque images)
    print("\n########################################################")
    print("PROGRAMME 1 : validation des présences")
    print("output du xlsx :", xlsx_path)
    
    create_presence_workbook(xlsx_path)

    try:
        model = load_similarity_model(MODEL_PATH)
        print("Modèle de similarité chargé :", MODEL_PATH)
        print("Clés du modèle :", model.keys())

    except Exception as e:
        model = None
        print(f"Pas de modèle de similarité trouvé au path : {MODEL_PATH}")
        print("Erreur :", e)
    ###############################################
    #recuperation des images et initialisation des statistiques
    image_paths = [
        p for p in sorted(presences_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    stats = {
        "total": 0,
        "grid_read": 0,
        "signature_read": 0,
        "grid_signature_match": 0,
        "grid_signature_mismatch": 0,
    }
    
    ###############################################
    #main loop for each photos in the choosen form
    for img_path in image_paths:
        result = autoValidID(
            filename_jpg=str(img_path),
            STUDENT_CLASS_SIGNATURES=STUDENT_CLASS_SIGNATURES,
            EXAM_FORMXX_PRESENCES_xlsx=str(xlsx_path),
            EXAM_FORMXX_RESULTS=str(results_dir),
            model=model,
        )

        #ajout personnel des statistiques 
        stats["total"] += 1
        grid_id = result.get("studentID_grid")
        sig_id = result.get("studentID_signature")

        if grid_id not in [None, ""]:
            stats["grid_read"] += 1

        if sig_id not in [None, ""]:
            stats["signature_read"] += 1

        if grid_id not in [None, ""] and sig_id not in [None, ""]:
            if str(grid_id) == str(sig_id):
                stats["grid_signature_match"] += 1
            else:
                stats["grid_signature_mismatch"] += 1

    #print des statistiques
    print("\nrésumé :", presences_dir.name)
    print(f"images traitées                 : {stats['total']}")
    print(f"studentID grid lus             : {stats['grid_read']}/{stats['total']}")
    print(f"studentID signature prédits      : {stats['signature_read']}/{stats['total']}")
    print(f"grille/signature identiques      : {stats['grid_signature_match']}")
    print(f"grille/signature différentes     : {stats['grid_signature_mismatch']}")
    print("fichier sauvegardé               :", xlsx_path)

    return str(xlsx_path)