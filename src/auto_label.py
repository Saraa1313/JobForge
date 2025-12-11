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
    """Auto-label pairs based on role and skill matching"""
    
    print("="*70)
    print("AUTO-LABELING RESUME-JOB PAIRS")
    print("="*70)
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"\n✅ Loaded {len(df)} pairs")
    
    # Define important tech keywords for overlap detection
    tech_keywords = {
        'java', 'python', 'javascript', 'react', 'django', 'spring', 'springboot',
        'go', 'redis', 'grpc', 'nlp', 'pytorch', 'sklearn', 'tensorflow', 
        'transformers', 'cnn', 'computer', 'vision', 'statistical', 'regression',
        'clustering', 'sql', 'tableau', 'excel', 'airflow', 'etl', 'spark',
        'typescript', 'redux', 'nextjs', 'next.js', 'css', 'api', 'rest',
        'microservices', 'mysql', 'postgresql', 'docker', 'aws', 'cloud'
    }
    
    # Define labeling logic
    def assign_label(row):
        resume_role = row['resume_role']
        job_role = row['job_role']
        resume_text = row['resume_text'].lower()
        job_text = row['job_text'].lower()
        
        # Extract words from text
        resume_words = set(resume_text.replace(',', ' ').replace('.', ' ').split())
        job_words = set(job_text.replace(',', ' ').replace('.', ' ').split())
        
        # Calculate tech keyword overlap
        resume_tech = resume_words & tech_keywords
        job_tech = job_words & tech_keywords
        tech_overlap = len(resume_tech & job_tech)
        
        # Calculate overall word overlap
        total_overlap = len(resume_words & job_words)
        
        # EXACT ROLE MATCH
        if resume_role == job_role:
            # Strong match: same role + high tech overlap
            if tech_overlap >= 3:
                return 2  # Strong match
            # Moderate match: same role + some tech overlap
            elif tech_overlap >= 1 or total_overlap > 15:
                return 1  # Moderate match
            else:
                return 1  # Still moderate (same role but different stack)
        
        # RELATED ROLES (can transition between these)
        related_pairs = [
            ('SWE', 'ML'),  # Software eng → ML (both use Python)
            ('ML', 'SWE'),  # ML → Software eng
            ('ML', 'DS'),   # ML and DS are very related
            ('DS', 'ML'),   # DS → ML
            ('DS', 'DA'),   # DS and Data Analyst related
            ('DA', 'DS'),   # DA → DS
            ('SWE', 'DS'),  # SWE and DS share Python/SQL
            ('DS', 'SWE'),  # DS → SWE
            ('FE', 'SWE'),  # Frontend can do full-stack
            ('SWE', 'FE'),  # Backend can do full-stack
            ('DA', 'ML'),   # Data Analyst → ML (with more experience)
            ('ML', 'DA'),   # ML can do analytics
        ]
        
        if (resume_role, job_role) in related_pairs:
            # Check if there's significant overlap
            if tech_overlap >= 2 or total_overlap > 15:
                return 1  # Moderate match (related + some overlap)
            else:
                return 1  # Still moderate (related roles)
        
        # UNRELATED ROLES
        # Few exceptions: PM with technical background
        if resume_role == 'PM' or job_role == 'PM':
            if tech_overlap >= 2:
                return 1  # Technical PM might work
            else:
                return 0  # Poor match
        
        # Frontend vs Data/ML roles (very different)
        if (resume_role == 'FE' and job_role in ['ML', 'DS', 'DA']) or \
           (job_role == 'FE' and resume_role in ['ML', 'DS', 'DA']):
            return 0  # Poor match
        
        # Default: unrelated
        return 0  # Poor match
    
    # Apply labeling
    print("\n⏳ Labeling pairs...")
    df['label'] = df.apply(assign_label, axis=1)
    
    # Show distribution
    print("\n" + "="*70)
    print("LABEL DISTRIBUTION")
    print("="*70)
    label_counts = df['label'].value_counts().sort_index()
    
    if len(label_counts) == 0:
        print("❌ Error: No labels assigned!")
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
    
    # Check if we have all 3 classes
    if len(label_counts) < 3:
        print(f"\n⚠️  WARNING: Only {len(label_counts)} classes found!")
        print("   Machine learning works best with all 3 classes.")
        if 2 not in label_counts.index:
            print("   Missing: Strong Match (label 2)")
            print("   Suggestion: Lower tech_overlap threshold or manually add some label 2s")
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved labeled data to: {output_file}")
    
    # Show examples from each class
    print("\n" + "="*70)
    print("EXAMPLE LABELS")
    print("="*70)
    
    for label in [2, 1, 0]:
        if label in df['label'].values:
            label_name = ['Poor Match', 'Moderate Match', 'Strong Match'][label]
            examples = df[df['label'] == label]
            if len(examples) > 0:
                example = examples.iloc[0]
                print(f"\n[{label_name.upper()}]")
                print(f"Resume: {example['resume_id']} ({example['resume_role']})")
                print(f"  {example['resume_text'][:80]}...")
                print(f"Job: {example['job_id']} ({example['job_role']})")
                print(f"  {example['job_text'][:80]}...")
                print(f"Label: {label}")
    
    print("\n" + "="*70)
    print("✅ AUTO-LABELING COMPLETE!")
    print("="*70)
    print("\nYou can now run: python src/train_models.py")
    print("\nNote: These are auto-generated labels based on simple rules.")
    print("For your actual project, review and adjust labels as needed.")
    
    # Show some statistics
    print("\n" + "="*70)
    print("LABELING STATISTICS")
    print("="*70)
    
    # Count perfect role matches
    perfect_matches = df[df['resume_role'] == df['job_role']]
    print(f"Same role pairs: {len(perfect_matches)} / {len(df)}")
    
    # Show role pair distribution for label 2
    if 2 in df['label'].values:
        strong_matches = df[df['label'] == 2]
        print(f"\nStrong Match role pairs:")
        role_pairs = strong_matches.groupby(['resume_role', 'job_role']).size()
        for (r_role, j_role), count in role_pairs.items():
            print(f"  {r_role} → {j_role}: {count}")

def main():
    """Main function"""
    # Try different possible locations for the file
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
        print("❌ Error: Could not find pairs_to_label.csv")
        print("   Looked in:")
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