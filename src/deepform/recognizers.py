"""Interfaces for the parts of the system that still need data or models.

These are deliberately separated so the pipeline runs end-to-end today and the
missing pieces can be dropped in without touching the orchestration:

* handwriting (Prenom / Nom / mantisse / exposant) needs a trained recogniser
  and an annotated train/val/test split (project section 4.2);
* signature authentication needs the STUDENT_CLASS_SIGNATURES database, which
  is not shipped with the example data;
* the exam multiple-choice and condition readers need ROI calibration on a
  real exam-page sample (only one ground-truth form is currently available).

Each function returns a neutral value plus a status string, never raising, so
the generated workbook always has the correct structure.
"""


# ----------------------------------------------------------------------------
# Handwriting (printed numbers are handled by OCR; these need a digit model)
# ----------------------------------------------------------------------------
def recognise_name(gray_roi):
    """Return (text, status).  TODO: connect a handwriting OCR / CNN model."""
    return None, "handwriting_model_not_loaded"


def recognise_scientific_number(gray_roi):
    """Return (mantisse, exposant, status) for a handwritten 'm,mm' field.

    TODO: segment digits + comma + minus, classify with a digit CNN trained on
    a dedicated split, then assemble the mantissa and optional exponent.
    """
    return None, None, "handwriting_model_not_loaded"


# ----------------------------------------------------------------------------
# Signature authentication
# ----------------------------------------------------------------------------
def authenticate_signature(gray_roi, signature_db_path):
    """Return (student_id, valid_flag, status).

    valid_flag is 1/0/None and feeds the 'Validation signature' cell.
    TODO: extract the signature, match against STUDENT_CLASS_SIGNATURES with
    the trained verification model.
    """
    return None, None, "signature_db_not_available"
