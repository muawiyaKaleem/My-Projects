import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SentimentSVMClassifier:
    def __init__(self, csv_file_path):
        """
        Initialize the SVM Sentiment Classifier
        
        Args:
            csv_file_path (str): Path to the CSV file containing sentiment data
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        
    def load_and_prepare_data(self):
        """Load CSV data and prepare it for training"""
        print("Loading data from CSV...")
        self.data = pd.read_csv(self.csv_file_path)
        
        # Display basic info about the dataset
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Check for missing values in key columns
        if 'cleaned_text_no_stopwords' in self.data.columns:
            text_column = 'cleaned_text_no_stopwords'
        elif 'cleaned_text' in self.data.columns:
            text_column = 'cleaned_text'
        else:
            text_column = 'text'
            
        print(f"Using text column: {text_column}")
        
        # Remove rows with missing text or sentiment labels
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=[text_column, 'sentiment_label'])
        final_count = len(self.data)
        
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} rows with missing data")
        
        # Display sentiment distribution
        print("\nSentiment Label Distribution:")
        print(self.data['sentiment_label'].value_counts())
        
        return text_column
    
    def prepare_features_and_labels(self, text_column, max_features=5000, test_size=0.2):
        """
        Prepare features using TF-IDF and encode labels
        
        Args:
            text_column (str): Name of the text column to use
            max_features (int): Maximum number of features for TF-IDF
            test_size (float): Proportion of data to use for testing
        """
        print(f"\nPreparing features with TF-IDF (max_features={max_features})...")
        
        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        
        # Prepare text data
        X_text = self.data[text_column].astype(str)
        
        # Transform text to TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(X_text)
        
        # Encode sentiment labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.data['sentiment_label'])
        
        print(f"Feature matrix shape: {X_tfidf.shape}")
        print(f"Label classes: {list(self.label_encoder.classes_)}")
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_tfidf, y_encoded, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_encoded
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
    
    def train_svm_model(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Train SVM model
        
        Args:
            kernel (str): SVM kernel type
            C (float): Regularization parameter
            gamma (str/float): Kernel coefficient
        """
        print(f"\nTraining SVM model with kernel='{kernel}', C={C}, gamma={gamma}...")
        
        # Initialize SVM classifier
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed!")
    
    def evaluate_model(self):
        """Evaluate the trained model and return metrics"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics for training set
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        train_precision = precision_score(self.y_train, y_train_pred, average='weighted')
        train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted')
        
        # Calculate metrics for test set
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred, average='weighted')
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        
        # Print results
        print(f"TRAINING SET METRICS:")
        print(f"  Accuracy:  {train_accuracy:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall:    {train_recall:.4f}")
        print(f"  F1-Score:  {train_f1:.4f}")
        
        print(f"\nTEST SET METRICS:")
        print(f"  Accuracy:  {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall:    {test_recall:.4f}")
        print(f"  F1-Score:  {test_f1:.4f}")
        
        # Detailed classification report
        print(f"\nDETAILED CLASSIFICATION REPORT (Test Set):")
        class_names = self.label_encoder.classes_
        print(classification_report(self.y_test, y_test_pred, target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - SVM Sentiment Classifier')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_svm.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
    
    def save_model(self, model_filename='svm_sentiment_model.joblib', 
                   vectorizer_filename='tfidf_vectorizer.joblib',
                   label_encoder_filename='label_encoder.joblib'):
        """
        Save the trained model, vectorizer, and label encoder
        
        Args:
            model_filename (str): Filename for the model
            vectorizer_filename (str): Filename for the vectorizer
            label_encoder_filename (str): Filename for the label encoder
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        print(f"\nSaving model components...")
        
        # Save model
        joblib.dump(self.model, model_filename)
        print(f"  Model saved as: {model_filename}")
        
        # Save vectorizer
        joblib.dump(self.vectorizer, vectorizer_filename)
        print(f"  Vectorizer saved as: {vectorizer_filename}")
        
        # Save label encoder
        joblib.dump(self.label_encoder, label_encoder_filename)
        print(f"  Label encoder saved as: {label_encoder_filename}")
        
        print("All components saved successfully!")
    
    def predict_new_text(self, text_list):
        """
        Predict sentiment for new text samples
        
        Args:
            text_list (list): List of text samples to predict
            
        Returns:
            list: Predictions with confidence scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Transform text using the fitted vectorizer
        X_new = self.vectorizer.transform(text_list)
        
        # Make predictions
        predictions = self.model.predict(X_new)
        probabilities = self.model.predict_proba(X_new)
        
        # Convert predictions back to original labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        results = []
        for i, (text, label, probs) in enumerate(zip(text_list, predicted_labels, probabilities)):
            max_prob = np.max(probs)
            results.append({
                'text': text,
                'predicted_sentiment': label,
                'confidence': max_prob,
                'probabilities': dict(zip(self.label_encoder.classes_, probs))
            })
        
        return results

def main():
    """Main function to run the SVM sentiment classifier"""
    # Initialize classifier with CSV file
    csv_file = "F:\Assessment\data\comments_sentiments.csv"  
    
    try:
        classifier = SentimentSVMClassifier(csv_file)
        
        # Step 1: Load and prepare data
        text_column = classifier.load_and_prepare_data()
        
        # Step 2: Prepare features and labels
        classifier.prepare_features_and_labels(text_column, max_features=5000, test_size=0.2)
        
        # Step 3: Train SVM model
        classifier.train_svm_model(kernel='rbf', C=1.0, gamma='scale')
        
        # Step 4: Evaluate model
        metrics = classifier.evaluate_model()
        
        # Step 5: Save model
        classifier.save_model()
        
        # Step 6: Demo prediction on new text
        print("\n" + "="*50)
        print("DEMO: PREDICTING NEW TEXT SAMPLES")
        print("="*50)
        
        sample_texts = [
            "This is absolutely amazing! I love it!",
            "This is terrible and disappointing.",
            "It's okay, nothing special but not bad either.",
            "Best product ever! Highly recommend!",
            "Waste of money, complete garbage."
        ]
        
        predictions = classifier.predict_new_text(sample_texts)
        
        for pred in predictions:
            print(f"Text: '{pred['text']}'")
            print(f"  Predicted: {pred['predicted_sentiment']} (confidence: {pred['confidence']:.3f})")
            print()
        
        # Save summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"svm_model_report_{timestamp}.txt"
        
        with open(report_filename, 'w') as f:
            f.write("SVM SENTIMENT CLASSIFIER - MODEL REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset: {csv_file}\n")
            f.write(f"Training samples: {classifier.X_train.shape[0]}\n")
            f.write(f"Test samples: {classifier.X_test.shape[0]}\n")
            f.write(f"Features: {classifier.X_train.shape[1]}\n")
            f.write(f"Classes: {list(classifier.label_encoder.classes_)}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Test Accuracy:  {metrics['test_accuracy']:.4f}\n")
            f.write(f"Test Precision: {metrics['test_precision']:.4f}\n")
            f.write(f"Test Recall:    {metrics['test_recall']:.4f}\n")
            f.write(f"Test F1-Score:  {metrics['test_f1']:.4f}\n\n")
            
            f.write("MODEL CONFIGURATION:\n")
            f.write(f"Algorithm: Support Vector Machine (SVM)\n")
            f.write(f"Kernel: RBF\n")
            f.write(f"C parameter: 1.0\n")
            f.write(f"Gamma: scale\n")
            f.write(f"Vectorizer: TF-IDF (max_features=5000)\n")
        
        print(f"Model report saved as: {report_filename}")
        
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found!")
        print("Please make sure your CSV file is in the same directory and update the filename in the script.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()