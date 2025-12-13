import csv
from collections import Counter

with open("data/pairs_labeled.csv", "r") as f:
    pairs = list(csv.DictReader(f))

labels = Counter(p["label"] for p in pairs)

print("Label Distribution:")
for label in ["0", "1", "2"]:
    count = labels[label]
    pct = count / len(pairs) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

print("\nStrong Match Examples:")
strong = [p for p in pairs if p["label"] == "2"][:5]

if not strong:
    print("⚠️ No strong matches found")
else:
    for p in strong:
        print(f"{p['resume_role']} -> {p['job_role']}")
        print(f"Resume tech: {p['resume_text'][:80]}")
        print(f"Job tech: {p['job_text'][:80]}")
        print()