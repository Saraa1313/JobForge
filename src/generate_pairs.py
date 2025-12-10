import pandas as pd

def skill_overlap(r, j):
    r = set(r.split(","))
    j = set(j.split(","))
    return len(r & j) / len(j)

def compute_label(d1, d2, overlap):
    return 0.7 * (1 if d1 == d2 else 0) + 0.3 * overlap

res = pd.read_csv("data/resumes.csv")
jobs = pd.read_csv("data/jobs.csv")

rows = []
for _, r in res.iterrows():
    for _, j in jobs.iterrows():
        ov = skill_overlap(r["skills"], j["skills"])
        score = compute_label(r["domain"], j["domain"], ov)
        rows.append([r["resume_id"], j["job_id"], score, r["text"], j["text"]])

df = pd.DataFrame(rows, columns=["resume_id","job_id","match_score","resume_text","job_text"])
df.to_csv("data/pairs.csv", index=False)
print("Generated data/pairs.csv")
