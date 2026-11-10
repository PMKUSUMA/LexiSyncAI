import fitz
from pathlib import Path
from .chunking import chunk_pages

def ingest_pdf(path: str, filename: str):
    doc = fitz.open(path)
    pages, digital, scanned, sections, table_count, warnings = [], 0, 0, [], 0, []
    for i, page in enumerate(doc):
        # PyMuPDF block extraction retains block coordinates and is generally more robust
        # than flattening page text; sort by vertical then horizontal reading position.
        blocks = page.get_text("blocks", sort=True)
        text_blocks = [b for b in blocks if len(b) > 4 and str(b[4]).strip()]
        text = "\n".join(str(b[4]).strip() for b in text_blocks)
        is_scanned = len(text.strip()) < 35
        if is_scanned:
            scanned += 1
            try:
                import pytesseract
                from PIL import Image
                import io
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                words = [w for w in ocr["text"] if w.strip()]
                text = " ".join(words)
                confs = [float(c) for c in ocr["conf"] if str(c).replace("-","").isdigit() and float(c) >= 0]
                if text.strip(): warnings.append(f"Page {i+1}: OCR fallback used; mean word confidence {sum(confs)/len(confs):.1f}" if confs else f"Page {i+1}: OCR fallback used")
                else: warnings.append(f"Page {i+1}: scanned page; OCR returned no text")
            except Exception:
                warnings.append(f"Page {i+1}: scanned page flagged; install Tesseract and pytesseract for OCR")
        else:
            digital += 1
        if not text.strip(): continue
        # Basic table signal from PyMuPDF table finder when available.
        try:
            table_count += len(page.find_tables().tables)
        except Exception:
            pass
        page_dict = {"filename": filename, "page": i+1, "text": text,
                     "blocks": [{"text":str(b[4]).strip(),"bbox":[round(float(x),1) for x in b[:4]]} for b in text_blocks]}
        pages.append(page_dict)
        for line in text.splitlines():
            s=line.strip()
            if len(s) < 120 and (s.isupper() or (s[:1].isdigit() and ('.' in s[:8] or ')' in s[:8]))):
                if s not in sections: sections.append(s)
    chunks = chunk_pages(pages)
    return {"chunks":chunks, "summary":{"filename":filename,"pages":len(doc),
            "digital_pages":digital,"scanned_pages":scanned,"sections":sections[:40],
            "tables":table_count,"warnings":warnings}}
