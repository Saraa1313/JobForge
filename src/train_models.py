"""
Our script:
1. Loads labeled resume-job pairs
2. Extracts features using TF-IDF and Sentence-BERT
3. Trains multiple classifiers
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
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
  

class ResumeJobMatcher:
    """Main class for training resume-job matching models"""
    
    def __init__(self, data_path='data/pairs_labeled.csv'):
        """Initialize and load data"""
     
        
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and validate the labeled dataset"""
     
        
        try:
            self.data = pd.read_csv(self.data_path)
            
        except FileNotFoundError:
           
            return False
        
       
        required_cols = ['resume_text', 'job_text', 'label']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
          
            return False
        
       
        unlabeled = self.data['label'].isna().sum()
        if unlabeled > 0:
            
            self.data = self.data.dropna(subset=['label'])
        
        self.data['label'] = self.data['label'].astype(int)
        
        label_counts = self.data['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = ['Poor Match', 'Moderate Match', 'Strong Match'][label]
            percentage = (count / len(self.data)) * 100
            print(f"  {label} ({label_name}): {count} ({percentage:.1f}%)")
        
        return True
    
    def prepare_tfidf_features(self):
        """Extract TF-IDF features from text"""
       
        
        self.data['combined_text'] = (
            self.data['resume_text'].fillna('') + ' ' + 
            self.data['job_text'].fillna('')
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words='english'
        )
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(self.data['combined_text'])
        y = self.data['label'].values
        
       
        
        # Split into train/test
        self.X_train_tfidf, self.X_test_tfidf, self.y_train, self.y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        return True
    
    def prepare_bert_features(self):
        """Extract Sentence-BERT embeddings"""
        if not BERT_AVAILABLE:
          
            return False
        
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Encoding resumes")
        resume_embeddings = self.bert_model.encode(
            self.data['resume_text'].fillna('').tolist(),
            show_progress_bar=True,
            batch_size=32
        )
        
        print("Encoding job descriptions")
        job_embeddings = self.bert_model.encode(
            self.data['job_text'].fillna('').tolist(),
            show_progress_bar=True,
            batch_size=32
        )
        
        X_bert = np.concatenate([resume_embeddings, job_embeddings], axis=1)
        
        from sklearn.metrics.pairwise import cosine_similarity
        cosine_sim = np.array([
            cosine_similarity([resume_embeddings[i]], [job_embeddings[i]])[0][0]
            for i in range(len(resume_embeddings))
        ]).reshape(-1, 1)
        
        X_bert = np.concatenate([X_bert, cosine_sim], axis=1)
       
        y = self.data['label'].values
        self.X_train_bert, self.X_test_bert, _, _ = train_test_split(
            X_bert, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return True
    
    def train_tfidf_models(self):
        """Train classifiers on TF-IDF features"""
      
        
        models_to_train = {
            'Logistic Regression (TF-IDF)': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest (TF-IDF)': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting (TF-IDF)': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        }
        
        for name, model in models_to_train.items():
            print(f"\n   Training {name}...")
            model.fit(self.X_train_tfidf, self.y_train)
            self.models[name] = model
            print(f"{name} - trained")
    
    def train_bert_models(self):
        """Train classifiers on BERT features"""
        if not BERT_AVAILABLE or not hasattr(self, 'X_train_bert'):
            return
        
        
        # Training Logistic Regression on BERT features
        lr_bert = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        lr_bert.fit(self.X_train_bert, self.y_train)
        self.models['Logistic Regression (BERT)'] = lr_bert
    
    def evaluate_models(self):
        """Evaluate all trained models"""
       
        
        for name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"MODEL: {name}")
            print('='*70)
            
            if 'BERT' in name:
                X_test = self.X_test_bert
            else:
                X_test = self.X_test_tfidf
            
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='macro', zero_division=0
            )
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred)
            }
            
            
            print(f"\nAccuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            
            print("\nClassification Report:")
            print(classification_report(
                self.y_test, y_pred,
                target_names=['Poor Match', 'Moderate Match', 'Strong Match'],
                zero_division=0
            ))
    
    def plot_results(self):
        """Create visualizations of results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            values = [self.results[model][metric] for model in model_names]
            
            bars = ax.bar(range(len(model_names)), values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Model', fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Poor', 'Moderate', 'Strong'],
                yticklabels=['Poor', 'Moderate', 'Strong'],
                ax=axes[idx],
                cbar_kws={'label': 'Count'}
            )
            axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
            axes[idx].set_ylabel('True Label', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        summary_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': [self.results[m]['accuracy'] for m in model_names],
            'Precision': [self.results[m]['precision'] for m in model_names],
            'Recall': [self.results[m]['recall'] for m in model_names],
            'F1-Score': [self.results[m]['f1'] for m in model_names]
        })
        summary_df = summary_df.sort_values('F1-Score', ascending=False)
        summary_df.to_csv('model_results_summary.csv', index=False)
        print("Saved")
        
     
    
    def show_example_predictions(self, n=5):
        """Show example predictions from the best model"""
       
        
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['f1'])
        print(f"\nBest Model: {best_model_name}")
        print(f"F1-Score: {self.results[best_model_name]['f1']:.4f}\n")
        
       
        y_pred = self.results[best_model_name]['y_pred']
        
        indices = np.random.choice(len(self.y_test), min(n, len(self.y_test)), replace=False)
        
        label_names = {0: 'Poor Match', 1: 'Moderate Match', 2: 'Strong Match'}
        
        for i, idx in enumerate(indices, 1):
            true_label = self.y_test[idx]
            pred_label = y_pred[idx]
            
            print(f"Example {i}:")
            print(f"  True Label:      {true_label} ({label_names[true_label]})")
            print(f"  Predicted Label: {pred_label} ({label_names[pred_label]})")
            print(f"  Correct: {'Yes' if true_label == pred_label else 'No'}")
            print()
    
    def save_models(self):
        """Save trained models"""
        
        joblib.dump(self.tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    
        
        for name, model in self.models.items():
            filename = name.replace(' ', '_').replace('(', '').replace(')', '').lower() + '.pkl'
            joblib.dump(model, filename)
            print(f"Saved: {filename}")
    
    def run_pipeline(self):
        """Execute the complete training pipeline"""
        if not self.load_data():
            return False
        
        if not self.prepare_tfidf_features():
            return False
        
        self.prepare_bert_features()
        
        self.train_tfidf_models()
        self.train_bert_models()
        
        self.evaluate_models()
        
        self.plot_results()
        
        self.show_example_predictions()
        
        self.save_models()
        
        return True

def main():
    """Main execution function"""
    matcher = ResumeJobMatcher('data/pairs_labeled.csv')
    success = matcher.run_pipeline()
    
    if not success:
        print("\nTraining failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
