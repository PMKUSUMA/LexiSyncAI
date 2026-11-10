import json, re
from groq import Groq

ABSTAIN="The documents do not contain enough evidence to answer this."
def answer_question(question,index,api_key,model,top_k=6,threshold=0.42):
    passages,latency,trace=index.search(question,top_k=top_k,mode="hybrid")
    score=max([p["score"] for p in passages],default=0)
    # Cross encoder logits are uncalibrated. Evidence gate combines retrieval availability,
    # lexical overlap and a conservative no-evidence fallback; tune against held-out data.
    qterms=set(re.findall(r"\w+",question.lower()))
    overlap=max([len(qterms & set(re.findall(r"\w+",p["text"].lower())))/max(1,len(qterms)) for p in passages],default=0)
    sufficient=bool(passages) and overlap>=0.12
    if not sufficient:
        return {"answer":ABSTAIN,"confidence":"INSUFFICIENT EVIDENCE","citations":[],"explanation":"No retrieved passage has adequate lexical support for the question.",
                "retrieval_score":float(score),"conflicts":[],"trace":trace}
    if not api_key:
        return {"answer":"Add your Groq API key in the sidebar or GROQ_API_KEY in .env to generate an answer. Retrieved evidence is shown below.",
                "confidence":"LOW","citations":[_citation(p) for p in passages[:3]],"explanation":"LLM generation was skipped because no API key was supplied.",
                "retrieval_score":float(score),"conflicts":[],"trace":trace}
    context="\n\n".join([f"[{i+1}] FILE={p['filename']} PAGE={p['page']} SECTION={p.get('section')}\n{p['text']}" for i,p in enumerate(passages)])
    prompt=f"""You are LexiSyncAI, a cautious legal-document analyst. Answer ONLY from the supplied evidence.
Do not use outside legal knowledge. If evidence is missing, contradictory, or does not establish the requested point, say exactly: "{ABSTAIN}"
Every material factual claim must carry an inline citation in this format [filename, p. N, section X].
Distinguish direct text from inference. Never invent clause numbers, values, dates, or legal rules.
If sources conflict, describe the conflict and do not silently choose one.
Return concise plain language and a short explanation of how the evidence supports the answer.

QUESTION: {question}
EVIDENCE:
{context}

Return JSON with keys answer, confidence (HIGH/MEDIUM/LOW/INSUFFICIENT EVIDENCE), explanation, conflicts (array)."""
    try:
        client=Groq(api_key=api_key)
        response=client.chat.completions.create(model=model,temperature=0.1,
            messages=[{"role":"system","content":"You produce evidence-grounded answers. Output valid JSON only."},
                      {"role":"user","content":prompt}], response_format={"type":"json_object"})
        data=json.loads(response.choices[0].message.content)
        ans=data.get("answer",ABSTAIN)
        confidence=data.get("confidence","LOW").upper()
        if ABSTAIN.lower() in ans.lower() or confidence=="INSUFFICIENT EVIDENCE":
            ans=ABSTAIN; confidence="INSUFFICIENT EVIDENCE"
        citations=[_citation(p) for p in passages if (p["filename"] in ans or str(p["page"]) in ans or p in passages[:2])]
        if not citations: citations=[_citation(p) for p in passages[:2]]
        return {"answer":ans,"confidence":confidence if confidence in ["HIGH","MEDIUM","LOW","INSUFFICIENT EVIDENCE"] else "LOW",
                "citations":citations,"explanation":data.get("explanation",""),
                "retrieval_score":float(score),"conflicts":data.get("conflicts",[]),"trace":trace}
    except Exception as e:
        return {"answer":f"Groq request failed: {e}","confidence":"LOW","citations":[_citation(p) for p in passages[:3]],
                "explanation":"The retrieved passages are displayed; no generated legal conclusion is asserted.",
                "retrieval_score":float(score),"conflicts":[],"trace":trace}

def _citation(p):
    return {"filename":p["filename"],"page":p["page"],"section":p.get("section"),
            "text":p["text"],"chunk_id":p["chunk_id"],"bbox":p.get("bbox")}
