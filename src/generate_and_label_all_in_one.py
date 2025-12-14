"""

This is a custom script for our use case and below is the flow:
1.⁠ ⁠Loads data using csv module (not pandas initially to avoid numpy issues)
2.⁠ ⁠Generates pairs
3.⁠ ⁠Auto-labels them
4.⁠ ⁠Saves ready-to-train dataset
"""

import csv
import os
import re
from collections import defaultdict, Counter



NUM_JOBS = 75  
USE_REAL_RESUMES = True
USE_SYNTHETIC_RESUMES = True

STRONG_MATCH_THRESHOLD = 2  
MODERATE_MATCH_THRESHOLD = 1 
HIGH_OVERLAP_STRONG = 2  



TECH_KEYWORDS = {
    'java', 'python', 'javascript', 'typescript', 'go', 'golang', 'rust', 'c++', 'cpp',
    'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'sql', 'c#', 'csharp',
    'react', 'vue', 'angular', 'django', 'flask', 'spring', 'springboot', 'node.js',
    'express', 'fastapi', 'next.js', 'nuxt', 'redux', 'graphql', 'rest', 'asp.net',
    'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'sklearn', 'hugging face',
    'transformers', 'llm', 'nlp', 'computer vision', 'opencv', 'cuda', 'pandas',
    'numpy', 'scipy', 'spark', 'hadoop', 'airflow', 'kafka', 'dbt', 'tableau',
    'powerbi', 'snowflake', 'bigquery', 'redshift', 'databricks',
    'aws', 'azure', 'gcp', 'kubernetes', 'k8s', 'docker', 'terraform', 'jenkins',
    'ci/cd', 'prometheus', 'grafana', 'helm', 'ansible', 'circleci',
    'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
    'dynamodb', 'neo4j', 'microservices', 'api', 'grpc', 'websocket', 'git',
    'linux', 'bash', 'shell', 'unix', 'windows', 'macos', 'algorithm', 'data structure'
}

ROLE_KEYWORDS = {
    'SWE': {'software', 'engineer', 'developer', 'programming', 'coding', 'backend', 'frontend', 'fullstack', 'application'},
    'ML': {'machine learning', 'deep learning', 'neural', 'model', 'ai', 'artificial intelligence', 'data science'},
    'DS': {'data scientist', 'data science', 'statistics', 'statistical', 'analysis', 'analytics', 'modeling'},
    'DA': {'data analyst', 'analyst', 'business intelligence', 'bi', 'reporting', 'visualization'},
    'FE': {'frontend', 'front-end', 'ui', 'ux', 'design', 'interface', 'web'},
    'DEVOPS': {'devops', 'sre', 'infrastructure', 'cloud', 'deployment', 'operations', 'reliability'},
    'PM': {'product manager', 'product management', 'roadmap', 'strategy', 'agile'}
}



def load_csv(filepath):
    """Load CSV file using pure Python"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    except FileNotFoundError:
        print(f"Error")
        return None
    except Exception as e:
        print(f"Error reading")
        return None

def save_csv(filepath, data, fieldnames):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def map_job_category(category):
    cat_lower = str(category).lower()
    
    if any(term in cat_lower for term in ['software engineer', 'fullstack', 'backend', 
                                            'mobile', 'ios', 'android']):
        return 'SWE'
    elif any(term in cat_lower for term in ['frontend', 'front-end', 'ui']):
        return 'FE'
    elif any(term in cat_lower for term in ['machine learning', 'ml engineer', 'ai']):
        return 'ML'
    elif any(term in cat_lower for term in ['data scientist', 'data science']):
        return 'DS'
    elif any(term in cat_lower for term in ['data engineer', 'data analyst']):
        return 'DA'
    elif any(term in cat_lower for term in ['devops', 'sre', 'cloud', 'infrastructure']):
        return 'DEVOPS'
    elif any(term in cat_lower for term in ['product manager', 'pm']):
        return 'PM'
    elif any(term in cat_lower for term in ['hardware', 'embedded']):
        return 'SWE' 
    else:
        return 'OTHER' 

def calculate_tech_overlap(text1, text2):
    words1 = set(re.sub(r'[^a-z0-9\s]', ' ', str(text1).lower()).split())
    words2 = set(re.sub(r'[^a-z0-9\s]', ' ', str(text2).lower()).split())
    
    tech1 = words1 & TECH_KEYWORDS
    tech2 = words2 & TECH_KEYWORDS
    overlap = len(tech1 & tech2)
    
    return overlap, tech1, tech2

def has_role_indicators(text, role):
    if role not in ROLE_KEYWORDS:
        return True 
    
    text_lower = str(text).lower()
    role_words = ROLE_KEYWORDS[role]
    
   
    for keyword in role_words:
        if keyword in text_lower:
            return True
    return False

def label_pair(resume_role, job_role_mapped, resume_text, job_text):

    
    tech_overlap, resume_tech, job_tech = calculate_tech_overlap(resume_text, job_text)
    
    
    if len(job_tech) == 0:
        return 0  # Poor match if the job is not in tech
    
    # If job category is completely unknown/non-tech then also Poor Match
    if job_role_mapped == 'OTHER':
        return 0  
    
    job_has_role_match = has_role_indicators(job_text, resume_role)
    
    if resume_role == job_role_mapped:
        if tech_overlap >= 2:  
            return 2  # Strong match
        elif tech_overlap >= 1:  
            return 1  # Moderate match
        elif len(resume_tech) > 0 and len(job_tech) > 0:
            
            return 1  # Moderate match
        else:
            return 0  # Poor
    
    related_pairs = [
        ('SWE', 'ML'), ('ML', 'SWE'),
        ('ML', 'DS'), ('DS', 'ML'),
        ('DS', 'DA'), ('DA', 'DS'),
        ('SWE', 'FE'), ('FE', 'SWE'),
        ('SWE', 'DEVOPS'), ('DEVOPS', 'SWE'),
        ('DA', 'ML'), ('ML', 'DA'),
        ('FE', 'DEVOPS'), ('DEVOPS', 'FE'),
    ]
    
    if (resume_role, job_role_mapped) in related_pairs:
        if tech_overlap >= 3:  
            return 2 
        elif tech_overlap >= 2:  
            return 1 
        elif tech_overlap >= 1: 
            return 1  
        else:
            return 0  
    
    if tech_overlap >= 3:
        return 1  
    elif tech_overlap >= 1:
        return 0  
    else:
        return 0  


def main():
   
    
    jobs = load_csv('hn_job_dataset.csv')
    if jobs is None:
        
        jobs = load_csv('data/hn_job_dataset.csv')
    if jobs is None:
        print("not found")
        return 1
    
    real_resumes = load_csv('real_resumes_extracted.csv')
    if real_resumes is None:
        real_resumes = load_csv('data/real_resumes_extracted.csv')
    if real_resumes is None:
        print("not found")
        return 1
    
    if real_resumes and len(real_resumes) > 0:
        required_cols = ['resume_id', 'resume_role', 'resume_text']
        actual_cols = list(real_resumes[0].keys())
        missing = [col for col in required_cols if col not in actual_cols]
        if missing:
            print(f"ERROR")
            return 1
    
    
    synthetic_resumes = load_csv('synthetic_resumes_50.csv')
    if synthetic_resumes is None:
        synthetic_resumes = load_csv('data/synthetic_resumes_50.csv')
    if synthetic_resumes is None:
        print("Not found")
        return 1
    
    if synthetic_resumes and len(synthetic_resumes) > 0:
        required_cols = ['resume_id', 'resume_role', 'resume_text']
        actual_cols = list(synthetic_resumes[0].keys())
        missing = [col for col in required_cols if col not in actual_cols]
        if missing:
            print(f"ERROR")
            return 1
    
    
    # Count categories
    job_categories = Counter(job['job_role'] for job in jobs)
    for cat, count in job_categories.most_common(5):
        print(f"  {cat}: {count}")
    
    jobs_by_category = defaultdict(list)
    for job in jobs:
        jobs_by_category[job['job_role']].append(job)
    
    top_categories = [cat for cat, _ in job_categories.most_common(5)]
    samples_per_cat = NUM_JOBS // len(top_categories)
    
    sampled_jobs = []
    for cat in top_categories:
        cat_jobs = jobs_by_category[cat]
        n_samples = min(samples_per_cat, len(cat_jobs))
        
        sorted_jobs = sorted(cat_jobs, key=lambda x: hash(x['job_id']))
        sampled_jobs.extend(sorted_jobs[:n_samples])
    
   
    

    resumes = []
    if USE_REAL_RESUMES:
        resumes.extend(real_resumes)
        print(f"Added {len(real_resumes)} real resumes")
    
    if USE_SYNTHETIC_RESUMES:
        resumes.extend(synthetic_resumes)
        print(f"Added {len(synthetic_resumes)} synthetic resumes")
    
    print(f"\n Total resumes: {len(resumes)}")
    

   
    
   
    
    pairs = []
    for resume in resumes:
        for job in sampled_jobs:
            # Handle both job_desc and job_text column names
            job_text = job.get('job_text') or job.get('job_desc', '')
            
            pairs.append({
                'resume_id': resume['resume_id'],
                'resume_role': resume['resume_role'],
                'resume_text': resume['resume_text'],
                'job_id': job['job_id'],
                'job_role': job['job_role'],
                'job_text': job_text
            })
    
    print(f" Generated {len(pairs)} pairs")
    

    

    
    for pair in pairs:
        job_role_mapped = map_job_category(pair['job_role'])
        label = label_pair(
            pair['resume_role'],
            job_role_mapped,
            pair['resume_text'],
            pair['job_text']
        )
        pair['label'] = label
    
    # Count labels
    label_counts = Counter(pair['label'] for pair in pairs)
    
   
    
    label_names = {0: 'Poor Match', 1: 'Moderate Match', 2: 'Strong Match'}
    for label in [0, 1, 2]:
        count = label_counts[label]
        pct = (count / len(pairs)) * 100
        print(f"  {label} ({label_names[label]}): {count} ({pct:.1f}%)")
    
    if len(label_counts) == 3:
        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        ratio = min_count / max_count
        print(f"\nBalance Ratio: {ratio:.2f}")
        if ratio > 0.3:
            print("Well balanced!")
        elif ratio > 0.2:
            print(" Acceptable balance")
        else:
            print(" Some imbalance (still usable)")
    

    # Save pairs
    fieldnames = ['resume_id', 'resume_role', 'resume_text', 'job_id', 'job_role', 'job_text', 'label']
    save_csv('data/pairs_labeled.csv', pairs, fieldnames)
    print(f"Saved")
    
    # Save resumes
    resume_fieldnames = ['resume_id', 'resume_role', 'resume_text']
    save_csv('data/resumes.csv', resumes, resume_fieldnames)
    print(f"Saved")
    
    # Save jobs
    job_fieldnames = ['job_id', 'job_role', 'job_text']
    jobs_to_save = [{'job_id': j['job_id'], 'job_role': j['job_role'], 
                     'job_text': j.get('job_text') or j.get('job_desc', '')} 
                    for j in sampled_jobs]
    save_csv('data/jobs.csv', jobs_to_save, job_fieldnames)
    print(f"Saved")
    

if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)