# LexiSyncAI — Evidence-Grounded Legal Document Analyst

A Streamlit web application for ingesting public legal PDFs, building a hybrid BM25 + dense retrieval index, reranking evidence with a cross-encoder, and generating grounded answers through the Groq API.

## 1. Requirements

- Python 3.10 or 3.11 recommended.
- Internet access on first run to download SentenceTransformers models and call Groq.
- A Groq API key: https://console.groq.com/keys
- Optional OCR: install the Tesseract OCR system binary. `pytesseract` alone is not sufficient.

> **API/local-model clarification:** Groq is an API-hosted inference service, not a local LLM. The assignment explicitly allows an API if a local model cannot be used. Do not describe this implementation as fully local.

## 2. Run

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
streamlit run app.py
```

### macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
streamlit run app.py
```

The API key can alternatively be pasted into the sidebar at runtime. Never commit `.env` to GitHub.

## 3. Demo flow

1. Open the app and create a case name.
2. Upload 2–5 public legal PDFs (no confidential/client material).
3. Click **Ingest documents**. Review page counts, scanned-page flags, detected headings and tables.
4. Click **Build hybrid search index**. First run downloads the embedding and cross-encoder models.
5. Ask a factual clause question and inspect the cited source pages.
6. Ask a question whose answer is absent from the PDFs; demonstrate the insufficient-evidence response.
7. Run the evaluation tab and explain which values are measured and which require manual gold labels.

## 4. Architecture

PDF → PyMuPDF page/block extraction → scanned-page detection / optional Tesseract OCR → section-aware chunks → BM25 + transformer embeddings → reciprocal-rank fusion → cross-encoder reranking → evidence sufficiency gate → Groq LLM → cited answer and confidence.

## 5. Requirement traceability

| Brief requirement | Implementation |
|---|---|
| Digital PDF parsing / reading order | `ingestion.py`, PyMuPDF block extraction with sorted reading order |
| Scanned page flag + OCR fallback | `ingestion.py`, optional Tesseract |
| Clause/section-aware chunks | `chunking.py` |
| BM25 exact clause retrieval | `indexing.py` |
| Dense transformer embeddings | `all-MiniLM-L6-v2` |
| Hybrid merge | Reciprocal Rank Fusion |
| Cross-encoder reranker | `ms-marco-MiniLM-L-6-v2` |
| Recall@k | Evaluation scaffold; requires manually labelled gold chunk IDs |
| Grounded generation | `answering.py`, retrieved passages only in prompt |
| Page/section citations | Citation evidence panels in UI |
| Abstention | Evidence gate and explicit abstention state |
| ≥15 held-out questions | 16-question starter scaffold; customize to the selected PDFs |
| No-retrieval vs retrieval / dense vs hybrid | Search modes are exposed in the indexing class; extend evaluation with labelled runs |
| Latency | Search and end-to-end timing |
| Multi-document conflict | Prompt instructions + conflict field from LLM |
| Table extraction | PyMuPDF table detection count; full merged-cell reconstruction is not implemented |
| Bounding-box citations | Source block bounding boxes retained as metadata; chunk-level region mapping is not yet implemented |
| Public deployment | Streamlit Community Cloud / Hugging Face Spaces compatible with a suitable requirements file |

## 6. Evaluation protocol — important

The starter set contains 16 question templates, including three unanswerable questions. They are **not** a valid final benchmark until adapted to the PDFs used in the experiment.

For a defensible report:
1. Write at least 15 held-out questions from the chosen legal document bundle.
2. Include factual questions, at least two-page multi-hop questions, and at least three unanswerable questions.
3. For every answerable question, manually label the supporting `chunk_id` values (or page/section evidence) as `gold_chunk_ids`.
4. Record the expected answer and manually grade correctness.
5. Calculate Recall@5, citation precision (claims with correct supporting citations / all cited claims), abstention accuracy on unanswerable questions, and latency.
6. Compare at least one mandatory pair: dense-only vs hybrid, or retrieval vs no-retrieval. Use the same question set and report the exact setup.
7. Report failures and limitations honestly. Do not claim metrics that have not been measured.

## 7. Technical limitations to disclose

- Reading order is based on PyMuPDF's sorted text blocks. Complex multi-column layouts, footnotes and headers may need manual validation.
- OCR requires the Tesseract executable to be installed separately; OCR quality must be sampled and measured.
- Table detection currently reports table counts; complete merged-cell/spanning-header reconstruction and table-grounded QA are future work.
- Bounding boxes are retained at block level, but precise claim-to-region citations need additional mapping.
- The current abstention gate is a conservative heuristic, not a calibrated classifier. Calibrate it on a validation set.
- Cross-document conflict detection depends on evidence retrieval and LLM interpretation; verify conflicts manually.
- The evaluation starter does not invent gold answers. Fill the gold labels from the uploaded test documents before reporting recall or citation precision.
- Groq model availability and rate limits may change. Select a model available to your account.

## 8. Suggested public legal sources

Use public, non-confidential PDFs from EUR-Lex, SEC EDGAR, legislation.gov.uk, or CourtListener. Keep the chosen bundle and question set fixed for reproducibility.

## 9. GitHub submission

```bash
git init
git add .
git commit -m "Initial LexiSyncAI implementation"
git add .
git commit -m "Add evaluation and documentation"
```

Push the repository and put its URL in the documentation submitted to eCampus. Do not upload `.env` or expose the Groq key.
