import pandas as pd
import random

TOTAL_RESUMES = 50

DISTRIBUTION = {
    "SWE": 12, "ML": 10, "DS": 10, 
    "FE": 8, "DA": 6, "PM": 4
}


DEGREES = [
    "B.S. in Computer Science", "M.S. in Computer Science", "M.S. in Data Science",
    "B.S. in Electrical Engineering", "M.S. in Artificial Intelligence", "M.B.A."
]

DOMAINS = {
    "SWE": ["Distributed Systems", "Fintech", "Cloud Architecture", "High-Frequency Trading"],
    "ML": ["NLP", "Computer Vision", "Recommendation Systems", "Generative AI"],
    "DS": ["Predictive Analytics", "Fraud Detection", "Market Research", "Consumer Behavior"],
    "FE": ["Responsive Design", "SPA Development", "Accessibility", "Design Systems"],
    "DA": ["Business Intelligence", "Supply Chain", "Financial Reporting", "Marketing Analytics"],
    "PM": ["B2B SaaS", "Mobile Apps", "Enterprise Software", "Consumer Tech"]
}

ACHIEVEMENTS = {
    "SWE": ["Optimized backend API latency by 40% handling 10k requests/sec", "Built payment system processing $1M+ daily volume", "Led migration to AWS reducing infra costs by 20%"],
    "ML": ["Fine-tuned LLaMA model reducing support tickets by 60%", "Improved defect detection accuracy by 25% via Computer Vision", "Deployed recommender system boosting CTR by 15%"],
    "DS": ["Designed churn model improving retention by 12%", "Identified $50k in annual savings via supply chain analysis", "Built pricing algorithm increasing margin by 5%"],
    "FE": ["Reduced initial page load time by 3 seconds", "Built design system adopted by 4 product teams", "Refactored codebase reducing bug reports by 30%"],
    "DA": ["Automated reporting saving 15 hours/week", "Identified key trend leading to 10% revenue lift", "Consolidated 5 data sources into single source of truth"],
    "PM": ["Launched mobile app acquiring 50k users in month one", "Increased user retention by 20% through feature optimization", "Led cross-functional team to on-time Q4 launch"]
}

SKILL_POOLS = {
    "SWE": ["Python", "Java", "C++", "Go", "AWS", "Docker", "Kubernetes", "Redis", "Kafka", "PostgreSQL", "Microservices", "CI/CD"],
    "ML": ["Python", "PyTorch", "TensorFlow", "Hugging Face", "Scikit-learn", "MLOps", "CUDA", "OpenCV", "Generative AI", "LLMs"],
    "DS": ["Python", "R", "SQL", "Tableau", "Pandas", "Spark", "BigQuery", "Snowflake", "Statistical Modeling", "A/B Testing"],
    "FE": ["React", "Vue.js", "TypeScript", "Redux", "HTML", "CSS", "Next.js", "Tailwind", "Jest", "Figma"],
    "DA": ["SQL", "Excel", "PowerBI", "Tableau", "Google Analytics", "Looker", "dbt", "Python"],
    "PM": ["Jira", "Agile", "Scrum", "User Research", "Roadmapping", "A/B Testing", "Figma", "SQL"]
}

ROLE_MAP = {
    "SWE": "Software Engineer", "ML": "Machine Learning Engineer",
    "DS": "Data Scientist", "FE": "Frontend Developer",
    "DA": "Data Analyst", "PM": "Product Manager"
}


data = []
counter = 1

for category, count in DISTRIBUTION.items():
    for _ in range(count):
        role_title = ROLE_MAP[category]
        years = random.randint(2, 9)
        degree = random.choice(DEGREES)
        domain = random.choice(DOMAINS[category])
        achievement = random.choice(ACHIEVEMENTS[category])
        
        pool = SKILL_POOLS[category]
        top_skills = random.sample(pool, 3)
        all_skills_list = random.sample(pool, k=random.randint(6, min(len(pool), 10)))
        
        summary = f"Senior {role_title} with {years} years experience, holding a {degree}. Expert in {', '.join(top_skills)}. {achievement}."
        
        full_text = f"{summary}\nSKILLS: {', '.join(all_skills_list)}"
        
        data.append({
            "resume_id": f"Syn_R{counter}",
            "resume_role": category,
            "resume_text": full_text
        })
        counter += 1


df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True) 

filename = "synthetic_resumes_50.csv"
df.to_csv(filename, index=False)

print(f"Generated")