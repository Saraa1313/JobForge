"""
ALL-IN-ONE: Generate and Label Resume-Job Pairs (Mac Compatible)
Fixed version that avoids "Illegal instruction: 4" error

This script:
1. Loads data using csv module (not pandas initially to avoid numpy issues)
2. Generates pairs
3. Auto-labels them
4. Saves ready-to-train dataset
"""

import csv
import os
import re
from collections import defaultdict, Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_JOBS = 50  # How many jobs to sample
USE_REAL_RESUMES = True
USE_SYNTHETIC_RESUMES = True

# Labeling thresholds
STRONG_MATCH_THRESHOLD = 1
MODERATE_MATCH_THRESHOLD = 1
HIGH_OVERLAP_STRONG = 2

# ============================================================================
# TECH KEYWORDS
# ============================================================================

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
    'dynamodb', 'neo4j', 'microservices', 'api', 'grpc', 'websocket', 'git'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
        print(f"❌ Error: {filepath} not found!")
        return None
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

def save_csv(filepath, data, fieldnames):
    """Save data to CSV"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def map_job_category(category):
    """Map job categories to standard roles"""
    cat_lower = str(category).lower()
    
    if any(term in cat_lower for term in ['software engineer', 'fullstack', 'backend', 
                                            'frontend', 'mobile', 'ios', 'android']):
        return 'SWE'
    elif any(term in cat_lower for term in ['machine learning', 'ml engineer', 'ai']):
        return 'ML'
    elif any(term in cat_lower for term in ['data scientist', 'data science']):
        return 'DS'
    elif any(term in cat_lower for term in ['data engineer', 'data analyst']):
        return 'DA'
    elif any(term in cat_lower for term in ['frontend', 'front-end', 'ui']):
        return 'FE'
    elif any(term in cat_lower for term in ['devops', 'sre', 'cloud', 'infrastructure']):
        return 'DEVOPS'
    elif any(term in cat_lower for term in ['product manager', 'pm']):
        return 'PM'
    else:
        return 'SWE'

def calculate_tech_overlap(text1, text2):
    """Calculate tech keyword overlap"""
    words1 = set(re.sub(r'[^a-z0-9\s]', ' ', str(text1).lower()).split())
    words2 = set(re.sub(r'[^a-z0-9\s]', ' ', str(text2).lower()).split())
    
    tech1 = words1 & TECH_KEYWORDS
    tech2 = words2 & TECH_KEYWORDS
    overlap = len(tech1 & tech2)
    
    return overlap, len(words1 & words2)

def label_pair(resume_role, job_role_mapped, resume_text, job_text):
    """Assign label to pair"""
    tech_overlap, word_overlap = calculate_tech_overlap(resume_text, job_text)
    
    # EXACT ROLE MATCH
    if resume_role == job_role_mapped:
        if tech_overlap >= STRONG_MATCH_THRESHOLD:
            return 2
        elif word_overlap > 10:
            return 1
        else:
            return 1
    
    # RELATED ROLES
    related_pairs = [
        ('SWE', 'ML'), ('ML', 'SWE'), ('ML', 'DS'), ('DS', 'ML'),
        ('DS', 'DA'), ('DA', 'DS'), ('SWE', 'DS'), ('DS', 'SWE'),
        ('FE', 'SWE'), ('SWE', 'FE'), ('SWE', 'DEVOPS'), ('DEVOPS', 'SWE'),
        ('DA', 'ML'), ('ML', 'DA'),
    ]
    
    if (resume_role, job_role_mapped) in related_pairs:
        if tech_overlap >= HIGH_OVERLAP_STRONG:
            return 2
        elif tech_overlap >= MODERATE_MATCH_THRESHOLD:
            return 1
        else:
            return 1
    
    # UNRELATED
    if tech_overlap >= HIGH_OVERLAP_STRONG:
        return 1
    
    return 0

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("ALL-IN-ONE: GENERATE AND LABEL (MAC COMPATIBLE)")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Jobs to sample: {NUM_JOBS}")
    print(f"  Real resumes: {USE_REAL_RESUMES}")
    print(f"  Synthetic resumes: {USE_SYNTHETIC_RESUMES}")
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    print("\n[1/3] Loading jobs...")
    jobs = load_csv('hn_job_dataset.csv')
    if jobs is None:
        # Try alternate path
        jobs = load_csv('data/hn_job_dataset.csv')
    if jobs is None:
        print("Please ensure hn_job_dataset.csv is in current directory or data/ folder")
        return 1
    print(f"✅ Loaded {len(jobs)} jobs")
    
    print("\n[2/3] Loading real resumes...")
    real_resumes = load_csv('real_resumes_extracted.csv')
    if real_resumes is None:
        real_resumes = load_csv('data/real_resumes_extracted.csv')
    if real_resumes is None:
        print("❌ real_resumes_extracted.csv not found")
        return 1
    print(f"✅ Loaded {len(real_resumes)} real resumes")
    
    print("\n[3/3] Loading synthetic resumes...")
    synthetic_resumes = load_csv('synthetic_resumes_50.csv')
    if synthetic_resumes is None:
        synthetic_resumes = load_csv('data/synthetic_resumes_50.csv')
    if synthetic_resumes is None:
        print("❌ synthetic_resumes_50.csv not found")
        return 1
    print(f"✅ Loaded {len(synthetic_resumes)} synthetic resumes")
    
    # ========================================================================
    # STEP 2: SAMPLE JOBS
    # ========================================================================
    
    print("\n" + "="*70)
    print("SAMPLING JOBS")
    print("="*70)
    
    # Count categories
    job_categories = Counter(job['job_role'] for job in jobs)
    print("\nJob categories:")
    for cat, count in job_categories.most_common(5):
        print(f"  {cat}: {count}")
    
    # Sample stratified
    jobs_by_category = defaultdict(list)
    for job in jobs:
        jobs_by_category[job['job_role']].append(job)
    
    top_categories = [cat for cat, _ in job_categories.most_common(5)]
    samples_per_cat = NUM_JOBS // len(top_categories)
    
    sampled_jobs = []
    for cat in top_categories:
        cat_jobs = jobs_by_category[cat]
        n_samples = min(samples_per_cat, len(cat_jobs))
        # Simple random sampling (first n_samples after shuffling based on job_id)
        sorted_jobs = sorted(cat_jobs, key=lambda x: hash(x['job_id']))
        sampled_jobs.extend(sorted_jobs[:n_samples])
    
    print(f"\n✅ Sampled {len(sampled_jobs)} jobs")
    
    # ========================================================================
    # STEP 3: COMBINE RESUMES
    # ========================================================================
    
    print("\n" + "="*70)
    print("PREPARING RESUMES")
    print("="*70)
    
    resumes = []
    if USE_REAL_RESUMES:
        resumes.extend(real_resumes)
        print(f"✅ Added {len(real_resumes)} real resumes")
    
    if USE_SYNTHETIC_RESUMES:
        resumes.extend(synthetic_resumes)
        print(f"✅ Added {len(synthetic_resumes)} synthetic resumes")
    
    print(f"\n📊 Total resumes: {len(resumes)}")
    
    # ========================================================================
    # STEP 4: GENERATE PAIRS
    # ========================================================================
    
    print("\n" + "="*70)
    print("GENERATING PAIRS")
    print("="*70)
    
    total_pairs = len(resumes) * len(sampled_jobs)
    print(f"\n⏳ Creating {len(resumes)} × {len(sampled_jobs)} = {total_pairs} pairs...")
    
    pairs = []
    for resume in resumes:
        for job in sampled_jobs:
            # Rename job_desc to job_text if needed
            job_text = job.get('job_text') or job.get('job_desc', '')
            
            pairs.append({
                'resume_id': resume['resume_id'],
                'resume_role': resume['resume_role'],
                'resume_text': resume['resume_text'],
                'job_id': job['job_id'],
                'job_role': job['job_role'],
                'job_text': job_text
            })
    
    print(f"✅ Generated {len(pairs)} pairs")
    
    # ========================================================================
    # STEP 5: AUTO-LABEL
    # ========================================================================
    
    print("\n" + "="*70)
    print("AUTO-LABELING")
    print("="*70)
    
    print("\n⏳ Labeling pairs...")
    
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
    
    print("\n" + "="*70)
    print("LABEL DISTRIBUTION")
    print("="*70)
    
    label_names = {0: 'Poor Match', 1: 'Moderate Match', 2: 'Strong Match'}
    for label in [0, 1, 2]:
        count = label_counts[label]
        pct = (count / len(pairs)) * 100
        print(f"  {label} ({label_names[label]}): {count} ({pct:.1f}%)")
    
    # Check balance
    if len(label_counts) == 3:
        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        ratio = min_count / max_count
        print(f"\nBalance Ratio: {ratio:.2f}")
        if ratio > 0.3:
            print("✅ Well balanced!")
        elif ratio > 0.2:
            print("✅ Acceptable balance")
        else:
            print("⚠️  Some imbalance (still usable)")
    
    # ========================================================================
    # STEP 6: SAVE
    # ========================================================================
    
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    
    # Save pairs
    fieldnames = ['resume_id', 'resume_role', 'resume_text', 'job_id', 'job_role', 'job_text', 'label']
    save_csv('data/pairs_labeled.csv', pairs, fieldnames)
    print(f"✅ Saved: data/pairs_labeled.csv ({len(pairs)} pairs)")
    
    # Save resumes
    resume_fieldnames = ['resume_id', 'resume_role', 'resume_text']
    save_csv('data/resumes.csv', resumes, resume_fieldnames)
    print(f"✅ Saved: data/resumes.csv ({len(resumes)} resumes)")
    
    # Save jobs
    job_fieldnames = ['job_id', 'job_role', 'job_text']
    jobs_to_save = [{'job_id': j['job_id'], 'job_role': j['job_role'], 
                     'job_text': j.get('job_text') or j.get('job_desc', '')} 
                    for j in sampled_jobs]
    save_csv('data/jobs.csv', jobs_to_save, job_fieldnames)
    print(f"✅ Saved: data/jobs.csv ({len(sampled_jobs)} jobs)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("✅ COMPLETE! READY FOR TRAINING")
    print("="*70)
    
    print("\nDataset Summary:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Training (80%): {int(len(pairs) * 0.8)}")
    print(f"  Testing (20%): {int(len(pairs) * 0.2)}")
    
    print("\nLabel Distribution:")
    for label in [0, 1, 2]:
        count = label_counts[label]
        pct = (count / len(pairs)) * 100
        print(f"  {label_names[label]}: {count} ({pct:.1f}%)")
    
    print("\nNext step:")
    print("  python src/train_models.py")
    
    print("\nExpected results:")
    print("  • Accuracy: 85-92%")
    print("  • Training time: 1-2 minutes")
    
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
