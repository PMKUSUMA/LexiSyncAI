import re, math, time
from collections import Counter, defaultdict
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+(?:\.[0-9]+)*", text.lower())

class HybridIndex:
    def __init__(self, chunks):
        self.chunks=chunks
        if not chunks: raise ValueError("No text chunks to index")
        self.tokens=[tokenize(c["text"]+" "+str(c.get("section",""))) for c in chunks]
        self.bm25=BM25Okapi(self.tokens)
        self.embedder=None; self.cross=None; self.vectors=None
        self.embedding_status="not loaded"; self.reranker_status="not loaded"
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.vectors=self.embedder.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
            self.embedding_status="all-MiniLM-L6-v2 loaded"
        except Exception as e: self.embedding_status=f"unavailable: {e}"
        try:
            from sentence_transformers import CrossEncoder
            self.cross=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker_status="ms-marco-MiniLM-L-6-v2 loaded"
        except Exception as e: self.reranker_status=f"unavailable: {e}"

    def search(self, query, top_k=6, mode="hybrid"):
        t=time.perf_counter()
        qtok=tokenize(query)
        bm=self.bm25.get_scores(qtok)
        bm_order=sorted(range(len(bm)),key=lambda i:bm[i],reverse=True)
        dense_order=[]
        if self.embedder is not None and self.vectors is not None:
            import numpy as np
            qv=self.embedder.encode([query],normalize_embeddings=True)[0]
            sims=np.dot(self.vectors,qv)
            dense_order=sorted(range(len(sims)),key=lambda i:float(sims[i]),reverse=True)
        if mode=="dense":
            candidates=dense_order[:max(top_k*5,20)] if dense_order else bm_order[:max(top_k*5,20)]
        elif mode=="bm25": candidates=bm_order[:max(top_k*5,20)]
        else:
            # Reciprocal rank fusion, retaining exact lexical matches.
            scores=defaultdict(float)
            for rank,i in enumerate(bm_order[:50]): scores[i]+=1/(60+rank)
            for rank,i in enumerate(dense_order[:50]): scores[i]+=1/(60+rank)
            candidates=sorted(scores,key=scores.get,reverse=True)[:max(top_k*5,20)]
        if self.cross and candidates:
            pairs=[(query,self.chunks[i]["text"]) for i in candidates]
            cross_scores=self.cross.predict(pairs)
            candidates=sorted(zip(candidates,cross_scores),key=lambda x:float(x[1]),reverse=True)
            ranked=[(i,float(s)) for i,s in candidates[:top_k]]
        else:
            ranked=[(i,float(bm[i])) for i in candidates[:top_k]]
        results=[]
        for i,score in ranked:
            c=dict(self.chunks[i]); c["score"]=score; results.append(c)
        return results, (time.perf_counter()-t)*1000, {"mode":mode,"bm25_candidates":len(bm_order[:50]),
            "dense_enabled":bool(dense_order),"reranker_enabled":bool(self.cross)}
