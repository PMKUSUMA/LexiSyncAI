import streamlit as st
from pathlib import Path
import tempfile, time, os
from dotenv import load_dotenv
from lexisync.ingestion import ingest_pdf
from lexisync.indexing import HybridIndex
from lexisync.answering import answer_question
from lexisync.evaluation import run_evaluation, run_ablation, default_questions

load_dotenv()
st.set_page_config(page_title="LexiSyncAI", page_icon="⚖️", layout="wide")
st.title("LexiSyncAI")
st.caption("Evidence-Grounded Legal Document Analyst | Hybrid Retrieval · Reranking · Cited Answers · Abstention")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password",
                            help="Create a key at https://console.groq.com/keys")
    model = st.selectbox("Groq model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                                        "openai/gpt-oss-20b", "openai/gpt-oss-120b"])
    st.caption("The key is used only for this session unless you add it to .env.")
    st.divider()
    st.markdown("**Suggested demo:** upload 2–5 public legal PDFs, index, ask a clause question, then ask something absent.")

if "documents" not in st.session_state: st.session_state.documents = []
if "index" not in st.session_state: st.session_state.index = None
if "chunks" not in st.session_state: st.session_state.chunks = []
if "last_result" not in st.session_state: st.session_state.last_result = None

tabs = st.tabs(["1 · Case & documents", "2 · Ask LexiSyncAI", "3 · Evaluation", "4 · Architecture & methods"])

with tabs[0]:
    st.subheader("Create a case and upload public legal documents")
    case_name = st.text_input("Case name", value="Legal document review")
    uploads = st.file_uploader("Upload 2–5 public legal PDF files", type=["pdf"], accept_multiple_files=True)
    if uploads:
        st.info(f"{len(uploads)} PDF(s) selected. The assignment specifies a bundle of 2–5 public PDFs.")
        if len(uploads) > 5:
            st.error("Please upload no more than five PDFs.")
        if st.button("Ingest documents", type="primary", disabled=len(uploads) > 5):
            all_chunks, summaries = [], []
            progress = st.progress(0)
            for i, up in enumerate(uploads):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(up.getvalue()); path = tmp.name
                try:
                    result = ingest_pdf(path, up.name)
                    all_chunks.extend(result["chunks"])
                    summaries.append(result["summary"])
                finally:
                    os.unlink(path)
                progress.progress((i+1)/len(uploads))
            st.session_state.documents = summaries
            st.session_state.chunks = all_chunks
            st.session_state.index = None
            st.success(f"Ingestion complete: {len(summaries)} document(s), {len(all_chunks)} evidence chunks.")
    if st.session_state.documents:
        st.markdown("#### Detected document structure")
        for d in st.session_state.documents:
            with st.expander(f"{d['filename']} · {d['pages']} pages"):
                st.write(f"Digital text pages: {d['digital_pages']} · Scanned/image-only pages: {d['scanned_pages']}")
                st.write(f"Detected sections/headings: {d['sections']}")
                st.write(f"Tables detected: {d['tables']}")
                if d.get("warnings"): st.warning(" · ".join(d["warnings"]))
        if st.button("Build hybrid search index", type="primary"):
            with st.spinner("Building BM25 and dense embedding indexes; loading cross-encoder..."):
                st.session_state.index = HybridIndex(st.session_state.chunks)
            st.success("Index ready. BM25 + dense vectors + cross-encoder reranker initialized.")
    st.caption("Scanned pages are detected. OCR fallback uses Tesseract when installed; otherwise they are clearly flagged.")

with tabs[1]:
    st.subheader("Ask a question grounded in the uploaded documents")
    if not st.session_state.index:
        st.warning("Upload and index documents first.")
    else:
        q = st.text_area("Legal question", placeholder='e.g. Does clause 7.2 limit liability for the indemnities in Section 9?', height=90)
        c1,c2 = st.columns([1,1])
        with c1: top_k = st.slider("Evidence passages", 3, 12, 6)
        with c2: threshold = st.slider("Abstention threshold", 0.10, 0.95, 0.42, 0.01)
        if st.button("Ask LexiSyncAI", type="primary", disabled=not q.strip()):
            start = time.perf_counter()
            with st.spinner("Retrieving, reranking and verifying evidence..."):
                result = answer_question(q, st.session_state.index, api_key, model, top_k=top_k, threshold=threshold)
            result["latency_seconds"] = round(time.perf_counter()-start, 2)
            st.session_state.last_result = result
        r = st.session_state.last_result
        if r:
            status = r["confidence"]
            if status == "INSUFFICIENT EVIDENCE": st.error("INSUFFICIENT EVIDENCE · Abstained")
            elif status == "HIGH": st.success("HIGH confidence")
            elif status == "MEDIUM": st.info("MEDIUM confidence")
            else: st.warning("LOW confidence")
            st.markdown("### Answer")
            st.write(r["answer"])
            st.caption(f"Latency: {r.get('latency_seconds','—')} seconds · Retrieval score: {r.get('retrieval_score',0):.3f}")
            if r.get("explanation"): st.markdown("**Grounding explanation:** " + r["explanation"])
            st.markdown("### Citations and evidence")
            if not r.get("citations"):
                st.write("No sufficiently reliable supporting passage was found.")
            for cit in r.get("citations", []):
                with st.expander(f"{cit['filename']} · Page {cit['page']} · {cit.get('section') or 'Section not detected'}"):
                    st.markdown(cit["text"])
                    st.caption(f"Chunk: {cit['chunk_id']} · Source region: {cit.get('bbox','not available')}")
            if r.get("conflicts"):
                st.warning("Potential cross-document conflict detected")
                for conflict in r["conflicts"]: st.write("• " + conflict)
            with st.expander("Retrieval trace (technical demo)"):
                st.json(r.get("trace", {}))

with tabs[2]:
    st.subheader("Evaluation suite")
    st.write("The assignment requires ≥15 held-out questions, including factual, multi-hop and ≥3 unanswerable questions.")
    st.warning("Add document-specific gold answers and supporting page numbers before reporting final academic results. The included starter set is a scaffold, not a claim of measured accuracy.")
    if st.session_state.index:
        questions = default_questions()
        if st.button("Run retrieval and abstention evaluation"):
            with st.spinner("Running evaluation set..."):
                report = run_evaluation(questions, st.session_state.index)
            st.session_state.eval_report = report
        if "eval_report" in st.session_state:
            rep = st.session_state.eval_report
            a,b,c,d = st.columns(4)
            a.metric("Questions", rep["total"])
            b.metric("Recall@5", f"{rep['recall_at_5']:.1%}" if rep["recall_at_5"] is not None else "N/A")
            c.metric("Unanswerable abstention", f"{rep['abstention_accuracy']:.1%}" if rep["abstention_accuracy"] is not None else "N/A")
            d.metric("Mean retrieval latency", f"{rep['mean_latency_ms']:.0f} ms")
            st.dataframe(rep["rows"], use_container_width=True)
            st.download_button("Download evaluation report (JSON)", json.dumps(rep, indent=2), "evaluation_report.json")
            if st.button("Run mandatory dense-only vs hybrid comparison"):
                st.session_state.ablation = run_ablation(questions, st.session_state.index)
            if "ablation" in st.session_state:
                st.markdown("#### Retrieval ablation: dense-only vs hybrid")
                ab=st.session_state.ablation
                st.dataframe([{"Configuration":m,"Recall@5":v["recall_at_k"],"Mean latency (ms)":round(v["mean_latency_ms"],1)} for m,v in ab.items()], use_container_width=True)
                st.caption("Recall is shown only when gold_chunk_ids have been manually configured. Otherwise report N/A, not zero.")
    else:
        st.info("Index documents to enable evaluation.")

with tabs[3]:
    st.subheader("System architecture")
    st.code("""Public legal PDFs (2–5)
        │
        ▼
PyMuPDF extraction ── reading order / page metadata / scanned-page detection
        │                         └── Tesseract OCR fallback
        ▼
Clause- and section-aware chunking (+ table extraction)
        │
        ├── BM25 lexical index (exact clause numbers such as 7.2)
        └── SentenceTransformer dense embeddings
                    │
                    ▼
          Reciprocal Rank Fusion (hybrid merge)
                    │
                    ▼
          Cross-encoder reranking
                    │
                    ▼
        Evidence sufficiency / abstention gate
                    │
                    ▼
           Groq-hosted LLM (context only)
                    │
                    ▼
    Claim-level page + section citations / conflict warning
                    │
                    ▼
      UI + evaluation (recall@k, citation precision,
             abstention, latency, ablations)""", language="text")
    st.markdown("**Core implementation:** `lexisync/ingestion.py`, `chunking.py`, `indexing.py`, `answering.py`, `evaluation.py`.")
    st.markdown("**Important:** Groq is an API-hosted LLM, not a local LLM. The assignment explicitly permits an API if a local model cannot be used.")
