import pandas as pd
import pyarrow.parquet as pq
import re
from collections import Counter
import json

def analyze_dataset(name, df, text_col):
    texts = df[text_col].dropna().astype(str).tolist()
    
    print(f"\n{'='*50}\nDataset: {name}\n{'='*50}")
    print(f"Total rows: {len(df)}")
    print(f"Column Names: {list(df.columns)}")
    print(f"\n--- First 50 texts ---")
    for i, t in enumerate(texts[:50]):
        print(f"{i+1:02d}: {t}")
        
    all_text = " ".join(texts)
    
    # 1. Unique characters
    char_counts = Counter(all_text)
    print("\n--- All Unique Characters (sorted) ---")
    print(" ".join(sorted(char_counts.keys())))
    
    # 2. Non-alphabetic character frequency
    print("\n--- Non-alphabetic Character Frequency ---")
    # karakalpak alphabet approx: a-z A-Z plus á ǵ ı ń ó ú ş q ƣ
    # just look at non-word or basic
    non_alpha = {k:v for k,v in char_counts.items() if not k.isalpha()}
    for k, v in sorted(non_alpha.items(), key=lambda x: -x[1]):
        print(f"Char: {repr(k)} (Unicode: U+{ord(k):04X}) -> {v}")
        
    # 3. Unicode Anomalies (combining chars, control chars)
    print("\n--- Unicode Anomalies ---")
    anomalies = {k:v for k,v in char_counts.items() if unicodedata.category(k).startswith('C') or unicodedata.category(k).startswith('M')}
    if not anomalies: print("None found in characters.")
    else:
        for k, v in anomalies.items(): print(f"Char: {repr(k)} (Unicode: U+{ord(k):04X}) -> {v}")

    # 4. Pattern matching
    patterns = {
        "Numbers": r'\d+',
        "Dates": r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',
        "URLs/Emails": r'http[s]?://\S+|www\.\S+|\S+@\S+',
        "Percentages": r'\d+\s*%',
        "Currency": r'[$€£¥₽]|sum',
        "Units": r'\d+\s*(kg|cm|mm|m|km|g|ml|l|sec|min)\b',
        "Hyphens": r'-',
        "Apostrophes": r'[\'‘’]',
        "Special Symbols": r'[&@#*+~^\\|<>]'
    }
    
    print("\n--- Pattern Matches (up to 5 examples each) ---")
    for pat_name, pat_regex in patterns.items():
        matches = [t for t in texts if re.search(pat_regex, t)]
        print(f"{pat_name}: {len(matches)} texts found")
        for m in matches[:5]:
            print(f"  - {m}")
            
    # 5. Length distribution
    lengths = [len(t) for t in texts]
    if lengths:
        print(f"\n--- Length Distribution ---")
        print(f"Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.2f}")

import unicodedata

# 1. Mendeley CSV
csv_path = "/home/dilmurod/job/tts_project/data/Karakalpak_Speech_Corpus_mendeley_v1/train.csv"
try:
    df_csv = pd.read_csv(csv_path)
    # determine text col
    text_col = 'raw_text' if 'raw_text' in df_csv.columns else df_csv.columns[1] 
    analyze_dataset("Mendeley CSV", df_csv, text_col)
except Exception as e:
    print(f"Error reading {csv_path}: {e}")

# 2. karakalpak-speech-corpus parquet
pq_path1 = "/home/dilmurod/job/tts_project/data/karakalpak-speech-corpus/test-00000-of-00001.parquet"
try:
    df_pq1 = pd.read_parquet(pq_path1)
    text_col = 'raw_text' if 'raw_text' in df_pq1.columns else ('text' if 'text' in df_pq1.columns else df_pq1.columns[1])
    analyze_dataset("karakalpak-speech-corpus", df_pq1, text_col)
except Exception as e:
    print(f"Error reading {pq_path1}: {e}")

# 3. karakalpak-audio-dataset parquet
pq_path2 = "/home/dilmurod/job/tts_project/data/karakalpak-audio-dataset/test-00000-of-00001.parquet"
try:
    df_pq2 = pd.read_parquet(pq_path2)
    text_col = 'raw_text' if 'raw_text' in df_pq2.columns else ('text' if 'text' in df_pq2.columns else df_pq2.columns[1])
    analyze_dataset("karakalpak-audio-dataset", df_pq2, text_col)
except Exception as e:
    print(f"Error reading {pq_path2}: {e}")

