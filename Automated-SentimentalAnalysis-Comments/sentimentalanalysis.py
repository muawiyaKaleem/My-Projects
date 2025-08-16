
import pandas as pd
import numpy as np
import re
import string
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

# Suppress warning messages for cleaner output
warnings.filterwarnings('ignore')

# Download required NLTK data files if not already present
# These are needed for tokenization and stopword removal
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords corpus...")
    nltk.download('stopwords')

class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis class that uses multiple approaches
    to classify text sentiment as Positive, Negative, or Neutral.
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer with VADER and TextBlob models."""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text to be cleaned
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string if not already
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (social media specific)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text while preserving sentiment-carrying words.
        
        Args:
            text (str): Text to process
            
        Returns:
            str: Text with stopwords removed
        """
        if not text:
            return text
            
        # Tokenize text
        word_tokens = word_tokenize(text)
        
        # Keep important sentiment words even if they're in stopwords
        sentiment_words = {'not', 'no', 'nor', 'but', 'however', 'although', 'though'}
        
        # Filter out stopwords except sentiment-carrying ones
        filtered_text = [word for word in word_tokens 
                        if word not in self.stop_words or word in sentiment_words]
        
        return ' '.join(filtered_text)
    
    def get_textblob_sentiment(self, text):
        """
        Get sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            tuple: (sentiment_label, polarity_score, subjectivity_score)
        """
        if not text:
            return 'Neutral', 0.0, 0.0
            
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment based on polarity
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        return sentiment, polarity, subjectivity
    
    def get_vader_sentiment(self, text):
        """
        Get sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).
        VADER is particularly good for social media text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            tuple: (sentiment_label, compound_score, scores_dict)
        """
        if not text:
            return 'Neutral', 0.0, {'pos': 0.0, 'neu': 1.0, 'neg': 0.0, 'compound': 0.0}
            
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Classify sentiment based on compound score
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        return sentiment, compound, scores
    
    def get_ensemble_sentiment(self, text):
        """
        Combine TextBlob and VADER predictions for more robust sentiment analysis.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            tuple: (final_sentiment, confidence_score, detailed_scores)
        """
        if not text:
            return 'Neutral', 0.0, {}
            
        # Get predictions from both models
        tb_sentiment, tb_polarity, tb_subjectivity = self.get_textblob_sentiment(text)
        vader_sentiment, vader_compound, vader_scores = self.get_vader_sentiment(text)
        
        # Create detailed scores dictionary
        detailed_scores = {
            'textblob_sentiment': tb_sentiment,
            'textblob_polarity': tb_polarity,
            'textblob_subjectivity': tb_subjectivity,
            'vader_sentiment': vader_sentiment,
            'vader_compound': vader_compound,
            'vader_pos': vader_scores['pos'],
            'vader_neu': vader_scores['neu'],
            'vader_neg': vader_scores['neg']
        }
        
        # Ensemble logic: if both models agree, use that prediction
        if tb_sentiment == vader_sentiment:
            final_sentiment = tb_sentiment
            # Calculate confidence based on agreement and score magnitudes
            confidence = (abs(tb_polarity) + abs(vader_compound)) / 2
        else:
            # If models disagree, use the one with stronger confidence
            if abs(tb_polarity) > abs(vader_compound):
                final_sentiment = tb_sentiment
                confidence = abs(tb_polarity) * 0.7  # Reduced confidence due to disagreement
            else:
                final_sentiment = vader_sentiment
                confidence = abs(vader_compound) * 0.7
        
        # Normalize confidence to 0-1 range
        confidence = min(confidence, 1.0)
        
        return final_sentiment, confidence, detailed_scores

def analyze_comments_sentiment(csv_file_path, output_file_path=None):
    """
    Main function to analyze sentiment of comments from CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file containing comments
        output_file_path (str): Path to save the results (optional)
        
    Returns:
        pd.DataFrame: DataFrame with original data plus sentiment analysis results
    """
    
    print("Starting Sentiment Analysis Pipeline...")
    
    # Initialize sentiment analyzer class instance
    analyzer = SentimentAnalyzer()
    
    # Load the comments data from CSV file
    # Try to read the CSV and handle any potential errors
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded {len(df)} comments from {csv_file_path}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None
    
    # Identify the comment column in the CSV file
    # We'll look for common column names that might contain text comments
    comment_columns = ['comment', 'text', 'content', 'review', 'message']
    comment_col = None
    
    # Check if any of the standard column names exist
    for col in comment_columns:
        if col in df.columns:
            comment_col = col
            break
    
    # If no standard column found, find the first text-like column
    if comment_col is None:
        for col in df.columns:
            if df[col].dtype == 'object':  # Object dtype usually indicates text
                comment_col = col
                break
    
    # If still no suitable column found, exit with error
    if comment_col is None:
        print("Error: No suitable text column found in the CSV file")
        return None
    
    print(f"Using column '{comment_col}' for sentiment analysis")
    
    # Clean the text data using our preprocessing functions
    print("Cleaning text data...")
    df['cleaned_text'] = df[comment_col].apply(analyzer.clean_text)
    df['cleaned_text_no_stopwords'] = df['cleaned_text'].apply(analyzer.remove_stopwords)
    
    # Perform sentiment analysis using ensemble approach
    print("Analyzing sentiment with ensemble approach...")
    
    # Initialize list to store sentiment analysis results
    sentiment_results = []
    
    # Process each comment individually
    for idx, text in enumerate(df['cleaned_text']):
        # Show progress every 50 comments to track processing
        if idx % 50 == 0:
            print(f"   Processed {idx}/{len(df)} comments...")
            
        # Get sentiment prediction and confidence from ensemble method
        sentiment, confidence, detailed = analyzer.get_ensemble_sentiment(text)
        
        # Create result dictionary with all sentiment metrics
        result = {
            'sentiment_label': sentiment,
            'confidence_score': confidence,
            **detailed  # Unpack detailed scores from both models
        }
        sentiment_results.append(result)
    
    # Combine original data with sentiment analysis results
    sentiment_df = pd.DataFrame(sentiment_results)
    result_df = pd.concat([df, sentiment_df], axis=1)
    
    # Generate and display summary statistics
    print("\nSentiment Analysis Results:")
    sentiment_counts = result_df['sentiment_label'].value_counts()
    total_comments = len(result_df)
    
    # Show distribution of sentiments with percentages
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total_comments) * 100
        print(f"   {sentiment}: {count} ({percentage:.1f}%)")
    
    # Display average confidence score across all predictions
    print(f"   Average Confidence: {result_df['confidence_score'].mean():.3f}")
    
    # Save results to output file if path is provided
    if output_file_path:
        result_df.to_csv(output_file_path, index=False)
        print(f"Results saved to {output_file_path}")
    
    # Display sample results for verification
    print("\nSample Results:")
    sample_cols = [comment_col, 'sentiment_label', 'confidence_score', 
                   'textblob_polarity', 'vader_compound']
    available_cols = [col for col in sample_cols if col in result_df.columns]
    print(result_df[available_cols].head(10).to_string(index=False))
    
    return result_df

def get_sentiment_insights(df):
    """
    Generate additional insights and statistics from sentiment analysis results.
    This function extracts meaningful patterns and highlights from the analyzed data.
    
    Args:
        df (pd.DataFrame): DataFrame containing sentiment analysis results
        
    Returns:
        dict: Dictionary containing various insights and statistics
    """
    insights = {}
    
    # Calculate basic statistics about the sentiment analysis
    insights['total_comments'] = len(df)
    insights['sentiment_distribution'] = df['sentiment_label'].value_counts().to_dict()
    insights['average_confidence'] = df['confidence_score'].mean()
    
    # Find the most positive and negative comments based on TextBlob polarity scores
    insights['most_positive'] = df.loc[df['textblob_polarity'].idxmax()]
    insights['most_negative'] = df.loc[df['textblob_polarity'].idxmin()]
    
    # Analyze high confidence predictions (confidence > 0.7)
    high_confidence = df[df['confidence_score'] > 0.7]
    insights['high_confidence_count'] = len(high_confidence)
    insights['high_confidence_distribution'] = high_confidence['sentiment_label'].value_counts().to_dict()
    
    return insights

if __name__ == "__main__":
    # Configuration section - Update these paths according to your file structure
    INPUT_CSV = r"F:\Assessment\data\raw_comments.csv"  
    OUTPUT_CSV = r"F:\Assessment\data\comments_sentiments.csv" 
    
    # Display header for the sentiment analysis process
    print("=" * 60)
    print("        SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Execute the main sentiment analysis function
    results_df = analyze_comments_sentiment(INPUT_CSV, OUTPUT_CSV)
    
    # If analysis was successful, generate additional insights
    if results_df is not None:
        print("\n" + "=" * 40)
        print("        ADDITIONAL INSIGHTS")
        print("=" * 40)
        
        # Generate comprehensive insights from the results
        insights = get_sentiment_insights(results_df)
        
        # Display high confidence predictions breakdown
        print(f"\nHigh Confidence Predictions ({len(results_df[results_df['confidence_score'] > 0.7])} comments):")
        for sentiment, count in insights['high_confidence_distribution'].items():
            print(f"   {sentiment}: {count}")
        
        # Show the most positive comment found
        print(f"\nMost Positive Comment (Score: {insights['most_positive']['textblob_polarity']:.3f}):")
        comment_text = insights['most_positive'].get('comment', insights['most_positive'].get('text', 'N/A'))
        print(f"   \"{comment_text[:100]}...\"")
        
        # Show the most negative comment found
        print(f"\nMost Negative Comment (Score: {insights['most_negative']['textblob_polarity']:.3f}):")
        comment_text = insights['most_negative'].get('comment', insights['most_negative'].get('text', 'N/A'))
        print(f"   \"{comment_text[:100]}...\"")
        
        # Final success message
        print("\nSentiment analysis completed successfully!")
        print(f"Results saved to: {OUTPUT_CSV}")
    
    else:
        # Error message if analysis failed
        print("Sentiment analysis failed. Please check your input file and try again.")