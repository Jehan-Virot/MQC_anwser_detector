from functions.id_number_detection_functions import read_student_id
from functions.load_image_functions import load_binary_image
from functions.region_labeling import find_regions, find_signature_box, find_square_boxes
from functions.detect_inside_boxes import detect_if_the_signature_is_there
from functions.draw_result import draw_results
from os import listdir, mkdir
from os.path import exists
from shutil import rmtree
import re

MIN_CHECKED_EXTRA_RATIO = 1.20
SIGNATURE_MIN_INK_RATIO = 0.006
SOURCE_PATH_DATA = "data/"

source_dir = listdir(SOURCE_PATH_DATA)
source_dir = [form for form in source_dir if re.match(r"^FORM\d$", form)]

rmtree("signature_fails")
rmtree("id_fails")
if not exists("signature_fails"):
    mkdir("signature_fails")
if not exists("id_fails"):
    mkdir("id_fails")

list_etudiant_valide = []
list_etudiant_non_valide = []
list_etudiant_signature = []
list_etudiant_non_signature = []
for form in source_dir:
    temp_dir_path = f"{SOURCE_PATH_DATA}{form}/EXAM_{form}_PRESENCES/"
    presence_pages = listdir(temp_dir_path)
    for current_presence_page in presence_pages:
        current_presence_page_path = f"{temp_dir_path}{current_presence_page}"
        print(current_presence_page_path)
        try:
            img_binaire = load_binary_image(current_presence_page_path)
        except:
            print("format photo bug")
            break
        items_regions = find_regions(img_binaire)
        signature_box = find_signature_box(items_regions, img_binaire.shape)
        has_signature, ink_ratio = detect_if_the_signature_is_there(img_binaire, signature_box, SIGNATURE_MIN_INK_RATIO)
        the_small_boxes_for_the_id_number = find_square_boxes(items_regions)
        student_id, grid_info = read_student_id(the_small_boxes_for_the_id_number, img_binaire.shape, MIN_CHECKED_EXTRA_RATIO)
        
        if student_id == None:
            list_etudiant_non_valide.append((student_id, current_presence_page))
            draw_results(img_binaire, signature_box, grid_info, f"id_fails/{current_presence_page}")
        elif current_presence_page.find(student_id) != -1:
            list_etudiant_valide.append(student_id)
        else:
            list_etudiant_non_valide.append((student_id,current_presence_page))
            draw_results(img_binaire, signature_box, grid_info, f"id_fails/{current_presence_page}")
        if has_signature:
            list_etudiant_signature.append(student_id)
        else:
            list_etudiant_non_signature.append((student_id, current_presence_page))
            draw_results(img_binaire, signature_box, grid_info, f"signature_fails/{current_presence_page}")

            
print(f"nb étudiant avec ID détécté : {len(list_etudiant_valide)}/{len(list_etudiant_valide)+len(list_etudiant_non_valide)}")
print(f"nb étudiant avec ID non détécté : {len(list_etudiant_non_valide)}/{len(list_etudiant_valide)+len(list_etudiant_non_valide)}")

print(f"nb étudiant avec signature détécté : {len(list_etudiant_signature)}/{len(list_etudiant_signature)+len(list_etudiant_non_signature)}")
print(f"nb étudiant avec signature non détécté : {len(list_etudiant_non_signature)}/{len(list_etudiant_signature)+len(list_etudiant_non_signature)}")

print("liste étudiant id detect : ", list_etudiant_valide)
print("liste étudiant non id detect : ", list_etudiant_non_valide)
print("liste étudiant signature detect : ", list_etudiant_signature)
print("liste étudiant non signature detect : ", list_etudiant_non_signature)



