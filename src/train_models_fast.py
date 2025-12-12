"""
CS-441 Final Project: Resume-Job Match Quality Prediction
FAST Training Script (No BERT - Trains in <2 minutes)

This script:
1. Loads labeled resume-job pairs
2. Extracts TF-IDF features only (FAST!)
3. Trains 3 classifiers
4. Evaluates and compares models
5. Saves results and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

class ResumeJobMatcher:
    """Fast training class - TF-IDF only"""
    
    def __init__(self, data_path='data/pairs_labeled.csv'):
        """Initialize"""
        print("="*70)
        print("RESUME-JOB MATCH QUALITY PREDICTION (FAST VERSION)")
        print("CS-441 Final Project")
        print("="*70)
        
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load labeled dataset"""
        print("\n[1/5] Loading data...")
        start_time = time.time()
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.data)} labeled pairs ({time.time()-start_time:.1f}s)")
        except FileNotFoundError:
            print(f"❌ Error: '{self.data_path}' not found!")
            return False
        
        # Validate columns
        required_cols = ['resume_text', 'job_text', 'label']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            print(f"❌ Error: Missing columns: {missing_cols}")
            return False
        
        # Remove unlabeled
        unlabeled = self.data['label'].isna().sum()
        if unlabeled > 0:
            print(f"⚠️  Removing {unlabeled} unlabeled pairs...")
            self.data = self.data.dropna(subset=['label'])
        
        # Convert labels to int
        self.data['label'] = self.data['label'].astype(int)
        
        # Display distribution
        print("\nLabel Distribution:")
        label_counts = self.data['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = ['Poor Match', 'Moderate Match', 'Strong Match'][label]
            percentage = (count / len(self.data)) * 100
            print(f"  {label} ({label_name}): {count} ({percentage:.1f}%)")
        
        return True
    
    def prepare_features(self):
        """Extract TF-IDF features (FAST!)"""
        print("\n[2/5] Extracting TF-IDF features...")
        start_time = time.time()
        
        # Combine texts
        self.data['combined_text'] = (
            self.data['resume_text'].fillna('') + ' ' + 
            self.data['job_text'].fillna('')
        )
        
        # Create vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,  # Reduced from 5000 for speed
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        
        # Fit and transform
        print("   Vectorizing text...")
        X_tfidf = self.tfidf_vectorizer.fit_transform(self.data['combined_text'])
        y = self.data['label'].values
        
        print(f"✅ Features: {X_tfidf.shape} ({time.time()-start_time:.1f}s)")
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Train: {len(self.y_train)} | Test: {len(self.y_test)}")
        
        return True
    
    def train_models(self):
        """Train 3 classifiers"""
        print("\n[3/5] Training models...")
        
        models_to_train = {
            'Logistic Regression': LogisticRegression(
                max_iter=500,  # Reduced for speed
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=50,  # Reduced from 100 for speed
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=50,  # Reduced from 100 for speed
                random_state=42
            )
        }
        
        for name, model in models_to_train.items():
            print(f"\n   Training {name}...")
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            print(f"   ✅ Done ({time.time()-start_time:.1f}s)")
    
    def evaluate_models(self):
        """Evaluate all models"""
        print("\n[4/5] Evaluating models...")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"MODEL: {name}")
            print('='*70)
            
            # Predict
            y_pred = model.predict(self.X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='macro', zero_division=0
            )
            
            # Store
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            # Print
            print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            print("\nPer-Class Performance:")
            print(classification_report(
                self.y_test, y_pred,
                target_names=['Poor', 'Moderate', 'Strong'],
                zero_division=0
            ))
    
    def plot_results(self):
        """Create visualizations"""
        print("\n[5/5] Generating visualizations...")
        
        # 1. Metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            values = [self.results[model][metric] for model in model_names]
            
            bars = ax.bar(range(len(model_names)), values, color='steelblue', alpha=0.8)
            ax.set_xlabel('Model', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric_name} Comparison', fontsize=13, fontweight='bold')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: model_comparison.png")
        plt.close()
        
        # 2. Confusion matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Poor', 'Mod', 'Strong'],
                yticklabels=['Poor', 'Mod', 'Strong'],
                ax=axes[idx],
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 12, 'fontweight': 'bold'}
            )
            axes[idx].set_title(f'{name}\nConfusion Matrix', 
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: confusion_matrices.png")
        plt.close()
        
        # 3. Summary table
        summary_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': [self.results[m]['accuracy'] for m in model_names],
            'Precision': [self.results[m]['precision'] for m in model_names],
            'Recall': [self.results[m]['recall'] for m in model_names],
            'F1-Score': [self.results[m]['f1'] for m in model_names]
        })
        summary_df = summary_df.sort_values('Accuracy', ascending=False)
        summary_df.to_csv('model_results_summary.csv', index=False)
        print("✅ Saved: model_results_summary.csv")
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(summary_df.to_string(index=False))
    
    def save_models(self):
        """Save models"""
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        # Save vectorizer
        joblib.dump(self.tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        print("✅ tfidf_vectorizer.pkl")
        
        # Save models
        for name, model in self.models.items():
            filename = name.replace(' ', '_').lower() + '.pkl'
            joblib.dump(model, filename)
            print(f"✅ {filename}")
    
    def show_best_model(self):
        """Show best model info"""
        print("\n" + "="*70)
        print("BEST MODEL")
        print("="*70)
        
        best_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_acc = self.results[best_name]['accuracy']
        best_f1 = self.results[best_name]['f1']
        
        print(f"\n🏆 {best_name}")
        print(f"   Accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
        print(f"   F1-Score: {best_f1:.4f}")
        
        # Show some predictions
        y_pred = self.results[best_name]['y_pred']
        correct = (self.y_test == y_pred).sum()
        total = len(self.y_test)
        
        print(f"\n   Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    
    def run_pipeline(self):
        """Execute pipeline"""
        total_start = time.time()
        
        # Run steps
        if not self.load_data():
            return False
        if not self.prepare_features():
            return False
        
        self.train_models()
        self.evaluate_models()
        self.plot_results()
        self.save_models()
        self.show_best_model()
        
        # Done!
        total_time = time.time() - total_start
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        print(f"\nTotal time: {total_time:.1f} seconds")
        
        print("\nGenerated files:")
        print("  📊 model_comparison.png")
        print("  📊 confusion_matrices.png")
        print("  📄 model_results_summary.csv")
        print("  💾 *.pkl (trained models)")
        
        print("\nYou're ready to submit! 🎉")
        
        return True

def main():
    """Main function"""
    matcher = ResumeJobMatcher('data/pairs_labeled.csv')
    success = matcher.run_pipeline()
    
    if not success:
        print("\n❌ Training failed. Check errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())