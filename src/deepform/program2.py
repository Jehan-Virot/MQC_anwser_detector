"""PROGRAMME 2 - automatic reading of the scanned PDF exam forms.

autoReadForm   : iterate over every EXAM_FORMXX_*.pdf in a folder.
autoReadFormID : read one PDF -> one EXAM_FORMXX_abcd.xlsx (PAGE-01 + EXAM).

Graphic elements (grids, cryptogram) are read with low-level operations only;
printed text uses OCR; handwriting / signature go through recognizers.py.
"""

import os
from os.path import join, basename, splitext

from . import io_utils, ocr_printed, recognizers
from .registration import build_frame
from .omr_grid import read_digit_grid, read_group
from .cryptogram import extract_cryptogram, check_consistency
from .xlsx_writer import write_workbook
from .layout_form3 import ROI, FIRST_EXAM_PAGE, PAGE01_LABELS

DPI = 150


def _read_page01(frame, signature_db):
    """Return a dict label -> value for the PAGE-01 sheet."""
    values = {lbl: None for lbl in PAGE01_LABELS}

    # printed header (OCR)
    hdr = ocr_printed.parse_header_band(frame)
    values["Module"] = hdr["Module"] or None
    values["Professor"] = hdr["Professor"] or None
    values["Date"] = hdr["Date"] or None
    values["Code"] = hdr["Code"] or None

    # printed grade boxes (OCR)
    values["Note maximale"] = ocr_printed.read_int_box(frame, ROI["note_max"])
    values["Note pour valider"] = ocr_printed.read_int_box(frame, ROI["note_valid"])

    # graphic OMR grids (low level)
    sid, _ = frame.roi(*ROI["student_id"])
    sid_res = read_digit_grid(sid, n_cols=5, n_rows=10)
    values["STUDENT ID"] = int(sid_res["value"]) if sid_res["value"] else None

    grp, _ = frame.roi(*ROI["group"])
    grp_res = read_group(grp)
    values["Group"] = ("G" + grp_res["value"]) if grp_res["value"] else None

    # handwriting (interface; needs model)
    pren, _ = frame.roi(*ROI["prenom"])
    values["Prénom"], _ = recognizers.recognise_name(pren)
    nom, _ = frame.roi(*ROI["nom"])
    values["Nom"], _ = recognizers.recognise_name(nom)

    # signature authentication (interface; needs signature DB)
    sig, _ = frame.roi(*ROI["signature"])
    _, sig_valid, _ = recognizers.authenticate_signature(sig, signature_db)
    values["Validation signature"] = sig_valid

    # exam conditions (graphic, low-level checkbox reading)
    from .omr_conditions import read_conditions
    for lbl, val in read_conditions(frame).items():
        values[lbl] = val

    return values, sid_res, grp_res


def _read_exam_pages(frames):
    """Return (exam_answers dict, n_questions). Currently structural only."""
    # TODO: detect the QUESTION blocks and their choice checkboxes / numeric
    # answer boxes on each exam page (frames[FIRST_EXAM_PAGE-1:]).  Until the
    # exam-page ROIs are calibrated we emit an empty (but correctly shaped)
    # answer table; n_questions defaults to the template's 11.
    return {}, 11


def autoReadFormID(pdf_path, signature_db_path, results_dir):
    """Read one form PDF and write the matching .xlsx into results_dir."""
    name = splitext(basename(pdf_path))[0]          # EXAM_FORM3_0001
    out_path = join(results_dir, name + ".xlsx")

    pages = io_utils.pdf_to_page_images(pdf_path, dpi=DPI)
    frames = [build_frame(p)[0] for p in pages]

    if frames[0] is None:
        # registration failed on page 1 -> emit empty structured workbook
        write_workbook(out_path, {lbl: None for lbl in PAGE01_LABELS}, {}, 11)
        return out_path, {"status": "page1_registration_failed"}

    page01, sid_res, grp_res = _read_page01(frames[0], signature_db_path)

    # cryptogram consistency across all pages
    bitmaps = [extract_cryptogram(f) if f is not None else None for f in frames]
    crypto_ok, crypto_score = check_consistency(bitmaps)
    page01["Validation cryptogramme"] = 1 if crypto_ok else 0

    exam_answers, n_questions = _read_exam_pages(frames)

    write_workbook(out_path, page01, exam_answers, n_questions)

    report = {
        "status": "ok",
        "student_id": page01["STUDENT ID"],
        "group": page01["Group"],
        "crypto_ok": crypto_ok,
        "crypto_score": round(crypto_score, 3),
        "module": page01["Module"],
        "code": page01["Code"],
        "note_max": page01["Note maximale"],
        "note_valid": page01["Note pour valider"],
    }
    return out_path, report


def autoReadForm(pdf_dir, signature_db_path, results_dir):
    """Read every EXAM_FORMXX_*.pdf in pdf_dir into results_dir."""
    os.makedirs(results_dir, exist_ok=True)
    pdfs = sorted(f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    reports = []
    for pdf in pdfs:
        out, rep = autoReadFormID(join(pdf_dir, pdf), signature_db_path,
                                  results_dir)
        rep["file"] = pdf
        reports.append(rep)
        print(f"[autoReadForm] {pdf} -> {basename(out)} : {rep}")
    return reports
