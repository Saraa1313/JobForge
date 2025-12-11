import pandas as pd

RESUME_PATH = "data/resumes.csv"
JOB_PATH = "data/jobs.csv"
OUTPUT_PATH = "data/pairs_to_label.csv"

def main():
    resumes = pd.read_csv(RESUME_PATH)
    jobs = pd.read_csv(JOB_PATH)

    rows = []
    for _, r in resumes.iterrows():
        for _, j in jobs.iterrows():
            rows.append({
                "resume_id": r["resume_id"],
                "resume_role": r["role"],
                "resume_text": r["text"],
                "job_id": j["job_id"],
                "job_role": j["role"],
                "job_text": j["text"],
                "label": ""   # <-- YOU will fill 0/1/2 here
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} pairs into {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
