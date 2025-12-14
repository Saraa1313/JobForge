"""
Auto-Label Resume-Job Pairs Based on Role Matching

This script automatically labels pairs based on:
- Role matching (SWE, ML, DS, DA, FE, PM)
- Skill overlap analysis
- Tech stack similarity

Usage: python auto_label.py
"""

import pandas as pd

def auto_label_pairs(input_file='data/pairs_to_label.csv', output_file='data/pairs_labeled.csv'):
    
  
    # Loading our data 
    df = pd.read_csv(input_file)
    print(f"\n✅ Loaded {len(df)} pairs")
    
    # defining important keywords for overlap detection
    tech_keywords = {
        'java', 'python', 'javascript', 'react', 'django', 'spring', 'springboot',
        'go', 'redis', 'grpc', 'nlp', 'pytorch', 'sklearn', 'tensorflow', 
        'transformers', 'cnn', 'computer', 'vision', 'statistical', 'regression',
        'clustering', 'sql', 'tableau', 'excel', 'airflow', 'etl', 'spark',
        'typescript', 'redux', 'nextjs', 'next.js', 'css', 'api', 'rest',
        'microservices', 'mysql', 'postgresql', 'docker', 'aws', 'cloud'
    }
    
    def assign_label(row):
        resume_role = row['resume_role']
        job_role = row['job_role']
        resume_text = row['resume_text'].lower()
        job_text = row['job_text'].lower()
        
        resume_words = set(resume_text.replace(',', ' ').replace('.', ' ').split())
        job_words = set(job_text.replace(',', ' ').replace('.', ' ').split())
        
        
        resume_tech = resume_words & tech_keywords
        job_tech = job_words & tech_keywords
        tech_overlap = len(resume_tech & job_tech)
        
        total_overlap = len(resume_words & job_words)
        
       
        if resume_role == job_role:
            # Strong match would be same role + high tech overlap
            if tech_overlap >= 3:
                return 2  # Strong match
            # Moderate match would same role + some tech overlap
            elif tech_overlap >= 1 or total_overlap > 15:
                return 1  # Moderate match
            else:
                return 1 
        
        related_pairs = [
            ('SWE', 'ML'), 
            ('ML', 'SWE'), 
            ('ML', 'DS'),   
            ('DS', 'ML'),   
            ('DS', 'DA'),  
            ('DA', 'DS'),   
            ('SWE', 'DS'), 
            ('DS', 'SWE'), 
            ('FE', 'SWE'),  
            ('SWE', 'FE'),  
            ('DA', 'ML'),  
            ('ML', 'DA'),   
        ]
        
        if (resume_role, job_role) in related_pairs:
            # Checking if there is significant overlap
            if tech_overlap >= 2 or total_overlap > 15:
                return 1  
            else:
                return 1  
        
        # UNRELATED ROLES
        # May be few exceptions like PM with technical background
        if resume_role == 'PM' or job_role == 'PM':
            if tech_overlap >= 2:
                return 1  # Technical PM might work
            else:
                return 0  # Poor match
        
        # Frontend vs Data/ML roles are very different
        if (resume_role == 'FE' and job_role in ['ML', 'DS', 'DA']) or \
           (job_role == 'FE' and resume_role in ['ML', 'DS', 'DA']):
            return 0  # Poor match
        
       
        return 0 
    
    print("\n⏳ Labeling pairs...")
    df['label'] = df.apply(assign_label, axis=1)
    
    # Printing distribution for our debug
    print("\n" + "="*70)
    print("LABEL DISTRIBUTION")
    print("="*70)
    label_counts = df['label'].value_counts().sort_index()
    
    if len(label_counts) == 0:
        print("Error: No labels assigned!")
        return
    
    for label in [0, 1, 2]:
        if label in label_counts.index:
            count = label_counts[label]
            label_name = ['Poor Match', 'Moderate Match', 'Strong Match'][label]
            percentage = (count / len(df)) * 100
            print(f"  {label} ({label_name}): {count} ({percentage:.1f}%)")
        else:
            label_name = ['Poor Match', 'Moderate Match', 'Strong Match'][label]
            print(f"  {label} ({label_name}): 0 (0.0%)")
    
    if len(label_counts) < 3:
        print(f"\n⚠️  WARNING: Only {len(label_counts)} classes found!")
        print("   Machine learning works best with all 3 classes.")
        if 2 not in label_counts.index:
            print("   Missing: Strong Match (label 2)")
            print("   Suggestion: Lower tech_overlap threshold or manually add some label 2s")
    
    df.to_csv(output_file, index=False)
    print(f"\n Saved labeled data")
    


def main():
    
    import os
    
    possible_paths = [
        'data/pairs_to_label.csv',
        'pairs_to_label.csv',
        '../data/pairs_to_label.csv'
    ]
    
    input_file = None
    for path in possible_paths:
        if os.path.exists(path):
            input_file = path
            break
    
    if input_file is None:
        print("Error: Could not find pairs_to_label.csv")
        print("Searched in:")
        for path in possible_paths:
            print(f"   - {path}")
        return 1
    
    # Determine output path
    if 'data/' in input_file:
        output_file = input_file.replace('pairs_to_label.csv', 'pairs_labeled.csv')
    else:
        output_file = 'pairs_labeled.csv'
    
    auto_label_pairs(input_file, output_file)
    return 0

if __name__ == "__main__":
    exit(main())