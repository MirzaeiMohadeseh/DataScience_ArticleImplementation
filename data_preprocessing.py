# data_preprocessing_single_file.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from collections import Counter

print("=" * 70)
print(" پیش‌پردازش دیتاست پلاستیک - یک فایل خروجی")
print("=" * 70)

class TextPreprocessor:
    def __init__(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            print(" داده‌های NLTK آماده")
        except:
            print(" استفاده از روش جایگزین")
        
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """پاکسازی کامل متن - حذف @ها و wwwها"""
        text = str(text)
        
        text = re.sub(r'[^\w\s]', ' ', text)
        
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        text = re.sub(r'\d+', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        text = text.lower()
        
        return text
    
    def correct_typos(self, text):
        """اصلاح خطاهای تایپی رایج"""
        typo_corrections = {
            'teh': 'the', 'plastik': 'plastic', 'enviroment': 'environment',
            'recyling': 'recycling', 'goverment': 'government', 'importent': 'important',
            'polution': 'pollution', 'ocen': 'ocean', 'peopl': 'people'
        }
        
        for typo, correction in typo_corrections.items():
            text = re.sub(r'\b' + typo + r'\b', correction, text)
        
        return text
    
    def tokenize_text(self, text):
        """توکن‌سازی"""
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        filtered_tokens = []
        for token in tokens:
            if (token not in self.stop_words and 
                len(token) > 2 and 
                token.isalpha()):
                stemmed = self.stemmer.stem(token)
                filtered_tokens.append(stemmed)
        
        return filtered_tokens

def main():
    """تابع اصلی"""
    
    try:
        df = pd.read_csv('random_plastic_comments_500.csv')
        print(f" دیتاست لود شد: {len(df)} کامنت")
    except FileNotFoundError:
        print(" فایل دیتاست یافت نشد.")
        return
    
     
    preprocessor = TextPreprocessor()
    
    print(" در حال پیش‌پردازش...")
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    df['corrected_text'] = df['cleaned_text'].apply(preprocessor.correct_typos)
    df['tokens'] = df['corrected_text'].apply(preprocessor.tokenize_text)
    
    df = df[df['tokens'].apply(len) > 0]
    print(f" پیش‌پردازش کامل شد: {len(df)} کامنت")
    
    all_tokens = []
    for tokens in df['tokens']:
        all_tokens.extend(tokens)
    
    word_freq = Counter(all_tokens)
    top_words = word_freq.most_common(10)
    
    print(f"\n ۱۰ کلمه پرتکرار:")
    for i, (word, freq) in enumerate(top_words, 1):
        print(f"   {i:2d}. {word:<15} {freq:>3} بار")
    
    words, counts = zip(*top_words)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(words, counts, color='skyblue')
    plt.title('Top 10 Most Frequent Words', fontsize=16, fontweight='bold')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    df.to_csv('plastic_dataset_processed.csv', index=False, encoding='utf-8-sig')
    print(f"\n فقط یک فایل ذخیره شد: 'plastic_dataset_processed.csv'")
    print(f" شامل {len(df)} کامنت پردازش شده")

if __name__ == "__main__":
    main()