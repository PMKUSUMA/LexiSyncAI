import re, hashlib

CLAUSE = re.compile(r"(?m)^\s*((?:\d+\.){1,5}\d+\.?|(?:Article|Section|Clause)\s+\d+[A-Za-z0-9().-]*)\s+")
def chunk_pages(pages, max_chars=1500, overlap=180):
    out=[]
    for p in pages:
        text=p["text"]
        # Split at likely legal headings/clause labels; fallback to paragraphs.
        parts=re.split(r"(?m)(?=^\s*(?:(?:\d+\.){1,5}\d+\.?|ARTICLE\s+\w+|Article\s+\w+|Section\s+\d+|SECTION\s+\d+|Clause\s+\d+)\s+)", text)
        current, section = "", ""
        for part in parts:
            part=part.strip()
            if not part: continue
            m=CLAUSE.match(part)
            if m: section=m.group(1).strip()
            if len(part)>max_chars:
                for start in range(0,len(part),max_chars-overlap):
                    frag=part[start:start+max_chars]
                    out.append(_make(p,frag,section))
            else:
                current = (current+"\n\n"+part).strip()
                if len(current)>=max_chars:
                    out.append(_make(p,current,section)); current=""
        if current.strip(): out.append(_make(p,current,section))
    return out

def _make(p,text,section):
    raw=f"{p['filename']}|{p['page']}|{section}|{text}"
    cid=hashlib.sha1(raw.encode("utf-8","ignore")).hexdigest()[:12]
    return {"chunk_id":cid,"filename":p["filename"],"page":p["page"],
            "section":section,"text":text,"bbox":None}
