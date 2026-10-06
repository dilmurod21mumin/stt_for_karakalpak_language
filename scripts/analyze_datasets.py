import pandas as pd
import pyarrow.parquet as pq
import re
from collections import Counter
import unicodedata

def generate_report(name, df, text_col, out_file):
    texts = df[text_col].dropna().astype(str).tolist()
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {name}\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(f"Column Names: {list(df.columns)}\n\n")
        
        f.write(f"--- First 50 texts ---\n")
        for i, t in enumerate(texts[:50]):
            f.write(f"{i+1:02d}: {t}\n")
            
        all_text = " ".join(texts)
        char_counts = Counter(all_text)
        
        f.write("\n--- All Unique Characters (sorted) ---\n")
        f.write(" ".join(sorted(char_counts.keys())) + "\n")
        
        f.write("\n--- Non-alphabetic Character Frequency ---\n")
        non_alpha = {k:v for k,v in char_counts.items() if not k.isalpha()}
        for k, v in sorted(non_alpha.items(), key=lambda x: -x[1]):
            f.write(f"Char: {repr(k)} (Unicode: U+{ord(k):04X}) -> {v}\n")
            
        f.write("\n--- Unicode Anomalies ---\n")
        anomalies = {k:v for k,v in char_counts.items() if unicodedata.category(k).startswith('C') or unicodedata.category(k).startswith('M') or k in ['ʼ', '‘', '’', '`', 'ʻ', '´']}
        if not anomalies: 
            f.write("None found.\n")
        else:
            for k, v in anomalies.items(): 
                f.write(f"Char: {repr(k)} (Unicode: U+{ord(k):04X}) -> {v}\n")

        patterns = {
            "Numbers": r'\d+',
            "Dates": r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',
            "URLs/Emails": r'http[s]?://\S+|www\.\S+|\S+@\S+',
            "Percentages": r'\d+\s*%',
            "Currency": r'[$€£¥₽]|sum\b',
            "Units": r'\d+\s*(kg|cm|mm|m|km|g|ml|l|sec|min)\b',
            "Hyphens": r'-',
            "Apostrophes": r'[\'‘’`ʼ]',
            "Special Symbols": r'[&@#*+~^\\|<>]',
            "Abbreviations": r'\b[A-Z]{2,}\b'
        }
        
        f.write("\n--- Pattern Matches (up to 5 examples each) ---\n")
        for pat_name, pat_regex in patterns.items():
            matches = [t for t in texts if re.search(pat_regex, t)]
            f.write(f"{pat_name}: {len(matches)} texts found\n")
            for m in matches[:5]:
                f.write(f"  - {m}\n")
                
        lengths = [len(t) for t in texts]
        if lengths:
            f.write(f"\n--- Length Distribution ---\n")
            f.write(f"Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.2f}\n")

csv_path = "/home/dilmurod/job/tts_project/data/Karakalpak_Speech_Corpus_mendeley_v1/train.csv"
df_csv = pd.read_csv(csv_path)
generate_report("Mendeley CSV", df_csv, 'text' if 'text' in df_csv.columns else df_csv.columns[1], '/home/dilmurod/job/tts_project/report_csv.txt')

pq_path1 = "/home/dilmurod/job/tts_project/data/karakalpak-speech-corpus/test-00000-of-00001.parquet"
df_pq1 = pd.read_parquet(pq_path1)
generate_report("karakalpak-speech-corpus", df_pq1, 'raw_text' if 'raw_text' in df_pq1.columns else 'text', '/home/dilmurod/job/tts_project/report_pq1.txt')

pq_path2 = "/home/dilmurod/job/tts_project/data/karakalpak-audio-dataset/test-00000-of-00001.parquet"
df_pq2 = pd.read_parquet(pq_path2)
generate_report("karakalpak-audio-dataset", df_pq2, 'sentence_latin', '/home/dilmurod/job/tts_project/report_pq2.txt')
