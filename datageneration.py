import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import random

print("=" * 70)
print("ایجاد دیتاست ۵۰۰ تایی کاملاً تصادفی")
print("=" * 70)

def add_noise_and_issues(text):
    """اضافه کردن انواع نویز و مشکلات به متن"""
    text = str(text)
    
    emojis = ['😊', '😡', '😢', '😂', '🤔', '👍', '👎', '❤️', '🔥', '💩', '🌍', '🐠', '🚮']
    if random.random() < 0.3:
        text += " " + random.choice(emojis)
    
    hashtags = ['#plasticpollution', '#savetheoceans', '#environment', '#climatechange', 
                '#recycling', '#zerowaste', '#ecofriendly', '#sustainability']
    if random.random() < 0.25:
        text += " " + random.choice(hashtags)
    
    mentions = ['@GreenPeace', '@BBCEnvironment', '@UNEnvironment', '@DavidAttenborough']
    if random.random() < 0.2:
        text = random.choice(mentions) + " " + text
    
    urls = ['https://bit.ly/3plasticfacts', 'http://www.greenpeace.org/plastic']
    if random.random() < 0.15:
        text += " " + random.choice(urls)
    
    if random.random() < 0.4:
        common_typos = {
            'the': 'teh', 'plastic': 'plastik', 'environment': 'enviroment',
            'recycling': 'recyling', 'government': 'goverment', 'important': 'importent'
        }
        for correct, typo in common_typos.items():
            if correct in text.lower() and random.random() < 0.3:
                text = text.replace(correct, typo)
                break
    
    if random.random() < 0.25:
        words = text.split()
        if len(words) > 2:
            random_index = random.randint(0, len(words)-1)
            words[random_index] = words[random_index].upper()
            text = " ".join(words)
    
    if random.random() < 0.35:
        extra_punctuation = ['!!!', '...', '??', '?!', '!?']
        text += random.choice(extra_punctuation)
    
    return text

def create_random_plastic_dataset():
    """ایجاد ۵۰۰ کامنت کاملاً تصادفی"""
    
    base_comments = [
        "Plastic pollution is destroying our oceans and marine life",
        "Single-use plastic should be banned completely",
        "Great to see biodegradable alternatives becoming available",
        "I switched to reusable bags and bottles",
        "The plastic industry is destroying our environment",
        "Microplastics are in our food and water this is terrifying",
        "Recycling is not working most plastic never gets recycled",
        "Supermarkets use too much plastic packaging",
        "The new plastic tax is a good step forward",
        "Companies using recycled plastic deserve support",
        "Community beach cleanups are very rewarding",
        "Plastic bottles take hundreds of years to decompose",
        "Different plastics have different recycling codes",
        "Many countries have banned single-use plastic bags",
        "Plastic production has increased dramatically",
        "We need better solutions for plastic waste management",
        "Plastic in fashion is a growing problem",
        "Government policies need to be stronger",
        "Young activists are leading the movement",
        "The Great Pacific Garbage Patch is mostly plastic",
        "I reduced my plastic consumption significantly",
        "Oil prices affect plastic recycling economics",
        "Seeing plastic waste in nature is heartbreaking",
        "Education about plastic waste is crucial",
        "Innovations in recycling technology give hope",
        "Plastic bags are convenient but harmful",
        "We must find balance between convenience and environment",
        "Plastic pollution affects everyone globally",
        "Corporate responsibility for plastic is important",
        "Local recycling programs vary too much",
        "Biodegradable plastics are not perfect but better",
        "Plastic waste in rivers flows to oceans",
        "Public awareness is increasing which is good",
        "More research needed on plastic alternatives",
        "Plastic packaging for food is often unnecessary",
        "International cooperation needed for plastic crisis",
        "Plastic recycling rates are disappointingly low",
        "Consumer choices can drive change in industry",
        "Plastic pollution costs billions in cleanup",
        "Marine animals suffer the most from plastic",
        "We need circular economy approaches for plastic",
        "Plastic waste exports to other countries wrong",
        "Innovative materials can replace plastic",
        "Plastic production should be regulated strictly",
        "Everyone has responsibility to reduce plastic",
        "Plastic problem requires global solution",
        "Reusable products are the future",
        "Plastic awareness campaigns are effective",
        "Waste management systems need improvement",
        "Plastic pollution is preventable with effort"
    ]
    
    expanded_data = []
    
    for i in range(500):
        base_comment = random.choice(base_comments)
        
        messy_text = add_noise_and_issues(base_comment)
        
        random_date = datetime(2017, 1, 1) + timedelta(
            days=random.randint(0, 2555)  
        )
        
        sources = ['BBC News', 'Guardian ',  'Mail Online']
        
        expanded_data.append({
            'comment_id': f"comment_{i+1:03d}",
            'text': messy_text,
            'source': random.choice(sources),
            'date': random_date,
            'likes': random.randint(0, 350),
            'shares': random.randint(0, 150),
            'word_count': len(messy_text.split())
        })
    
    df = pd.DataFrame(expanded_data)
    
    print(f" دیتاست تصادفی ایجاد شد: {len(df)} کامنت")
    print(f" نمونه‌ای از کامنت‌ها:")
    for i in range(3):
        print(f"   {i+1}. {df.iloc[i]['text']}")
    
    return df

def analyze_dataset_stats(df):
    """آنالیز آماری دیتاست بدون افشای احساسات"""
    print("\n" + "=" * 70)
    print(" آمار دیتاست (بدون اطلاعات احساسات)")
    print("=" * 70)
    
    print(f" آمار کلی:")
    print(f"   • تعداد کل کامنت‌ها: {len(df):,}")
    print(f"   • بازه زمانی: {df['date'].min().strftime('%Y-%m-%d')} تا {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   • میانگین طول کامنت: {df['text'].str.len().mean():.1f} کاراکتر")
    print(f"   • میانگین تعداد کلمات: {df['word_count'].mean():.1f}")
    print(f"   • مجموع لایک‌ها: {df['likes'].sum():,}")
    print(f"   • مجموع اشتراک‌ها: {df['shares'].sum():,}")
    
    print(f"\n توزیع منابع:")
    source_stats = df['source'].value_counts()
    for source, count in source_stats.items():
        percentage = (count / len(df)) * 100
        print(f"   • {source}: {count} کامنت ({percentage:.1f}%)")
    
    print(f"\n توزیع سالیانه:")
    df['year'] = df['date'].dt.year
    yearly_stats = df['year'].value_counts().sort_index()
    for year, count in yearly_stats.items():
        print(f"   • {year}: {count} کامنت")
    
    return df

def main():
    """تابع اصلی برای ایجاد دیتاست"""
    
    plastic_dataset = create_random_plastic_dataset()
    
    analyzed_data = analyze_dataset_stats(plastic_dataset)
    
    print("\n" + "=" * 70)
    print(" ۱۰ نمونه تصادفی از کامنت‌ها")
    print("=" * 70)
    
    samples = analyzed_data.sample(10, random_state=42)
    for idx, row in samples.iterrows():
        print(f"\n کامنت {idx+1}:")
        print(f"    منبع: {row['source']}")
        print(f"    تاریخ: {row['date'].strftime('%Y-%m-%d')}")
        print(f"    لایک: {row['likes']} |  اشتراک: {row['shares']}")
        print(f"    متن: \"{row['text']}\"")
        print("-" * 60)
    
    try:
        analyzed_data.to_csv('random_plastic_comments_500.csv', index=False, encoding='utf-8-sig')
        print(f"\n دیتاست در فایل 'random_plastic_comments_500.csv' ذخیره شد")
        
    except Exception as e:
        print(f"\n خطا در ذخیره‌سازی: {e}")
    
    print("\n " + "=" * 70)
    print(" دیتاست ۵۰۰ تایی کاملاً تصادفی ایجاد شد!")
    print(" درصد احساسات مشخص نیست - نیاز به تحلیل دارد")
    print(" متن‌ها نیاز به پیش‌پردازش دارند")
    print("=" * 70)
    
    return analyzed_data

if __name__ == "__main__":
    dataset = main()