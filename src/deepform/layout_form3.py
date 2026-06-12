"""Layout specification for the FORM-3 family of exam forms.

All regions of interest are given in normalised fiducial-box coordinates
(see registration.PageFrame).  Keeping the layout in one place makes the
pipeline adaptable to other forms: only this file changes.

PAGE-01 field order matches the generated .xlsx exactly (column A labels).
"""

# ---- PAGE-01 row labels, in the order they appear in the output sheet -------
PAGE01_LABELS = [
    "Module",                  # 1  printed   (OCR)
    "Professor",               # 2  printed   (OCR)
    "Date",                    # 3  printed   (OCR)
    "Code",                    # 4  printed   (OCR)
    "Notes de cours",          # 5  graphic   (condition checkbox)
    "Notes manuscrites",       # 6  graphic
    "Ordinateur portable",     # 7  graphic
    "Calculatrice ",           # 8  graphic   (note: trailing space, as template)
    "Feuilles brouillon",      # 9  graphic
    "Note maximale",           # 10 printed   (OCR)
    "Note pour valider",       # 11 printed   (OCR)
    None,                      # 12 blank row (matches reference template)
    "Prénom",                  # 13 handwriting
    "Nom",                     # 14 handwriting
    "Validation signature",    # 15 signature auth
    "Group",                   # 16 graphic   (OMR grid)
    "STUDENT ID",              # 17 graphic   (OMR grid)
    "Validation cryptogramme", # 18 graphic   (cross-page check)
]

# ---- Region of interest, normalised (u0, v0, u1, v1) ------------------------
ROI = {
    "header_band":   (0.02, 0.045, 0.98, 0.085),
    "student_id":    (0.66, 0.130, 1.00, 0.460),   # 5 cols x 10 rows
    "group":         (0.43, 0.165, 0.70, 0.460),   # 2 digit cols + letter col
    "prenom":        (0.00, 0.110, 0.42, 0.150),   # handwritten boxes
    "nom":           (0.00, 0.175, 0.42, 0.215),
    "signature":     (0.00, 0.230, 0.42, 0.380),
    "note_max":      (0.55, 0.710, 0.69, 0.765),
    "note_valid":    (0.55, 0.770, 0.69, 0.825),
    # exam-condition YES/NO blocks (5 conditions across the page)
    "conditions":    (0.00, 0.470, 1.00, 0.560),
}

# exam grading sheet
EXAM_CHOICE_COLS = ["CHOIX A", "CHOIX B", "CHOIX C", "CHOIX D",
                    "CHOIX E", "CHOIX F", "CHOIX G", "CHOIX H"]
EXAM_HEADER = ["QUESTION"] + EXAM_CHOICE_COLS + ["MANTISSE", "EXPOSANT", "UNITE"]

# first exam page (1-indexed) per the project statement (p5 -> end)
FIRST_EXAM_PAGE = 5
