import time

def default_questions():
    # Starter evaluation scaffold. Replace with document-specific questions and gold chunk IDs.
    return [
      {"id":"Q01","question":"What is the purpose of the agreement?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q02","question":"Who are the parties to the document?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q03","question":"What is the effective date?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q04","question":"What does clause 1.1 define?","answerable":True,"gold_chunk_ids":[],"type":"clause"},
      {"id":"Q05","question":"What is the payment deadline?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q06","question":"What are the termination rights?","answerable":True,"gold_chunk_ids":[],"type":"clause"},
      {"id":"Q07","question":"What limitation of liability is specified?","answerable":True,"gold_chunk_ids":[],"type":"clause"},
      {"id":"Q08","question":"Which obligations survive termination?","answerable":True,"gold_chunk_ids":[],"type":"multi-hop"},
      {"id":"Q09","question":"How does the indemnity clause interact with the liability cap?","answerable":True,"gold_chunk_ids":[],"type":"multi-hop"},
      {"id":"Q10","question":"What notice period is required?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q11","question":"Which law governs this agreement?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q12","question":"What dispute resolution procedure applies?","answerable":True,"gold_chunk_ids":[],"type":"clause"},
      {"id":"Q13","question":"What is the contract's confidential information definition?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q14","question":"What is the CEO's home address?","answerable":False,"gold_chunk_ids":[],"type":"unanswerable"},
      {"id":"Q15","question":"What is the private bank password of the supplier?","answerable":False,"gold_chunk_ids":[],"type":"unanswerable"},
      {"id":"Q16","question":"What was the weather on the signing date?","answerable":False,"gold_chunk_ids":[],"type":"unanswerable"},
    ]

def run_evaluation(questions,index):
    rows=[]; hits=[]; abstain_correct=[]; times=[]
    for q in questions:
        t=time.perf_counter()
        passages,_,_=index.search(q["question"],top_k=5,mode="hybrid")
        ms=(time.perf_counter()-t)*1000; times.append(ms)
        ids=[p["chunk_id"] for p in passages]
        gold=q.get("gold_chunk_ids",[])
        if gold: hits.append(bool(set(ids)&set(gold)))
        # Retrieval-only proxy: no evidence found is not the same as calibrated abstention.
        predicted_abstain=(len(passages)==0)
        if not q["answerable"]: abstain_correct.append(predicted_abstain)
        rows.append({"id":q["id"],"type":q["type"],"answerable":q["answerable"],
                     "retrieved_passages":len(passages),"gold_evidence_configured":bool(gold),
                     "manual_answer_correctness": "NOT GRADED", "citation_precision": "NOT GRADED",
                     "retrieval_only_abstention":predicted_abstain,"latency_ms":round(ms,1)})
    return {"total":len(rows),"recall_at_5":sum(hits)/len(hits) if hits else None,
            "abstention_accuracy":sum(abstain_correct)/len(abstain_correct) if abstain_correct else None,
            "mean_latency_ms":sum(times)/len(times) if times else 0,"rows":rows,
            "note":"Recall is calculated only for rows with manually configured gold_chunk_ids. Abstention metric is a retrieval-only proxy; validate generated answers manually."}


def run_ablation(questions,index,top_k=5):
    """Compare dense-only and hybrid retrieval on the same held-out questions."""
    import time
    results={}
    for mode in ("dense","hybrid"):
        recalls=[]; latencies=[]; rows=[]
        for q in questions:
            start=time.perf_counter()
            passages,_,_=index.search(q["question"],top_k=top_k,mode=mode)
            latencies.append((time.perf_counter()-start)*1000)
            gold=q.get("gold_chunk_ids",[])
            if gold: recalls.append(bool(set(p["chunk_id"] for p in passages)&set(gold)))
            rows.append({"id":q["id"],"mode":mode,"retrieved":len(passages),"gold_labelled":bool(gold)})
        results[mode]={"recall_at_k":sum(recalls)/len(recalls) if recalls else None,
                        "mean_latency_ms":sum(latencies)/len(latencies) if latencies else 0, "rows":rows}
    return results
