import re
import unicodedata
import os
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterable, Tuple, Set

# uv run python text_normalizer.py /home/dilmurod/job/tts_project/data/cleaned/karakalpak_speech_corpus/parquet_files

# Karakalpak Number to Word mapping
NUM_WORDS = {
    0: "nol", 1: "bir", 2: "eki", 3: "úsh", 4: "tórt", 5: "bes",
    6: "altı", 7: "jeti", 8: "segiz", 9: "toǵız", 10: "on",
    20: "jigirma", 30: "otız", 40: "qırq", 50: "eliw",
    60: "alpıs", 70: "jetpis", 80: "seksen", 90: "toqsan"
}

def num_to_kk_words(n: int) -> str:
    """Helper function to convert integers to Karakalpak spoken words."""
    if n == 0: return NUM_WORDS[0]
    
    # ONLY drop "bir" if the number is exactly a standalone base. 
    # If it has remainders (like 140), it will safely keep "bir".
    if n == 100: return "júz"
    if n == 1_000: return "mıń"
    if n == 1_000_000: return "million"
    if n == 1_000_000_000: return "milliard"
    
    words = []
    
    if n >= 1_000_000_000:
        b = n // 1_000_000_000
        words.append(num_to_kk_words(b) + " milliard")
        n %= 1_000_000_000
    if n >= 1_000_000:
        m = n // 1_000_000
        words.append(num_to_kk_words(m) + " million")
        n %= 1_000_000
    if n >= 1_000:
        k = n // 1_000
        words.append(num_to_kk_words(k) + " mıń")
        n %= 1_000
    if n >= 100:
        h = n // 100
        words.append(num_to_kk_words(h) + " júz")
        n %= 100
    if n >= 10:
        t = (n // 10) * 10
        words.append(NUM_WORDS[t])
        n %= 10
    if n > 0:
        words.append(NUM_WORDS[n])
        
    return " ".join(words)


ORDINAL_ENDINGS = {
    "bir": "birinshi",
    "eki": "ekinshi",
    "úsh": "úshinshi",
    "tórt": "tórtinshi",
    "bes": "besinshi",
    "altı": "altınshı",
    "jeti": "jetinshi",
    "segiz": "segizinshi",
    "toǵız": "toǵızınshı",
    "on": "onınshı",
    "jigirma": "jigirmanshı",
    "otız": "otızınshı",
    "qırq": "qırqınshı",
    "eliw": "eliwinshi",
    "alpıs": "alpısınshı",
    "jetpis": "jetpisinshi",
    "seksen": "sekseninshi",
    "toqsan": "toqsanınshı",
    "júz": "júzinshi",
    "mıń": "mıńınshı",
    "million": "millionınshı",
    "milliard": "milliardınshı",
}

def num_to_kk_ordinal(n: int) -> str:
    """Converts an integer into a Karakalpak spoken ordinal."""
    words = num_to_kk_words(n).split()
    if not words: return ""
    last_word = words[-1]
    words[-1] = ORDINAL_ENDINGS.get(last_word, last_word + "inshi")
    return " ".join(words)

# Cyrillic to Latin mapping for Karakalpak
CYR_TO_LATIN = {
    'а': 'a', 'ә': 'á', 'б': 'b', 'в': 'v', 'г': 'g', 'ғ': 'ǵ', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'j', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'қ': 'q', 'л': 'l', 'м': 'm',
    'н': 'n', 'ң': 'ń', 'о': 'o', 'ө': 'ó', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
    'у': 'u', 'ү': 'ú', 'ф': 'f', 'х': 'x', 'ҳ': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh',
    'щ': 'sh', 'ъ': '', 'ы': 'ı', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
}

#============================================================
# BASIC TEXT NORMALIZATION
#============================================================

def normalize_unicode(text: str) -> str:
    """Input: raw text -> Output: Unicode-normalized text."""
    if text is None: return ""
    return unicodedata.normalize('NFC', text)

def normalize_karakalpak_orthography(text: str) -> str:
    """Input: Unicode-normalized text -> Output: canonical Karakalpak text."""
    # Convert Cyrillic to Latin
    res = []
    for char in text:
        lower_char = char.lower()
        if lower_char in CYR_TO_LATIN:
            is_upper = char.isupper()
            mapped = CYR_TO_LATIN[lower_char]
            res.append(mapped.capitalize() if is_upper else mapped)
        else:
            res.append(char)
    text = "".join(res)
    
    # Standardize legacy ASCII-based Karakalpak Latin letters (e.g., g', o') to modern forms
    replacements = {
        r"g'|g`": "ǵ", r"G'|G`": "Ǵ",
        r"o'|o`": "ó", r"O'|O`": "Ó",
        r"n'|n`": "ń", r"N'|N`": "Ń",
        r"a'|a`": "á", r"A'|A`": "Á",
        r"u'|u`": "ú", r"U'|U`": "Ú",
        r"i'|i`": "í", r"I'|I`": "Í"
    }
    for pattern, rep in replacements.items():
        text = re.sub(pattern, rep, text)
    return text

def normalize_case(text: str) -> str:
    """Input: text -> Output: lowercase text."""
    return text.lower()

def normalize_whitespace(text: str) -> str:
    """Input: text -> Output: text with normalized whitespace."""
    return " ".join(text.split())


#============================================================
# SPECIAL TOKEN / SEMANTIC NORMALIZATION
#============================================================

def normalize_urls(text: str) -> str:
    """Input: text containing URLs -> Output: normalized URLs."""
    return re.sub(r'https?://\S+|www\.\S+', ' silteme ', text)

def normalize_emails(text: str) -> str:
    """Input: text containing email addresses -> Output: normalized email forms."""
    return re.sub(r'\S+@\S+', ' elektron pochta ', text)

def normalize_usernames_and_mentions(text: str) -> str:
    """Input: text containing @mentions/#hashtags -> Output: normalized forms."""
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    return text

def normalize_dates(text: str) -> str:
    """Input: text containing dates -> Output: spoken-form dates."""
    # Convert DD.MM.YYYY or DD/MM/YYYY into spaced digits
    return re.sub(r'\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b', r'\1 \2 \3', text)

def normalize_times(text: str) -> str:
    """Input: text containing times -> Output: spoken-form times."""
    # Convert HH:MM into HH MM
    return re.sub(r'\b(\d{1,2}):(\d{2})\b', r'\1 \2', text)

def normalize_roman_numerals(text: str) -> str:
    """Input: text containing Roman numerals -> Output: normalized spoken forms."""
    # Roman to decimal conversion for typical cases (I to XX)
    romans = {
        r'\bXX\b': '20', r'\bXIX\b': '19', r'\bXVIII\b': '18', r'\bXVII\b': '17',
        r'\bXVI\b': '16', r'\bXV\b': '15', r'\bXIV\b': '14', r'\bXIII\b': '13',
        r'\bXII\b': '12', r'\bXI\b': '11', r'\bX\b': '10', r'\bIX\b': '9',
        r'\bVIII\b': '8', r'\bVII\b': '7', r'\bVI\b': '6', r'\bV\b': '5',
        r'\bIV\b': '4', r'\bIII\b': '3', r'\bII\b': '2', r'\bI\b': '1'
    }
    for r, d in romans.items():
        text = re.sub(r, d, text)
    return text

def normalize_ordinals(text: str) -> str:
    """Input: text containing ordinal notation -> Output: spoken ordinal forms."""
    # 1. Handle explicit written suffixes (e.g., 1-shi, 2-nshi, 5-ınshı)
    text = re.sub(
        r'\b(\d+)-(?:shi|shı|nshi|nshı|inshi|ınshı)\b',
        lambda m: num_to_kk_ordinal(int(m.group(1))),
        text, flags=re.IGNORECASE
    )
    
    # 2. Handle number-word bindings (e.g., 2026-jıl -> 2026-ınshı jıl)
    # The regex strictly matches letters after the hyphen. 
    # A number-number format like 5-10 will safely be ignored here.
    text = re.sub(
        r'\b(\d+)-([a-zA-ZáóǵńúíıÁÓǴŃÚÍI]+)\b',
        lambda m: f"{num_to_kk_ordinal(int(m.group(1)))} {m.group(2)}",
        text
    )
    
    return text


def normalize_fractions(text: str) -> str:
    """Input: text containing fractions -> Output: spoken fraction forms."""
    # 1/2 -> 1 den 2
    return re.sub(r'\b(\d+)/(\d+)\b', r'\1 den \2', text)

def normalize_ranges(text: str) -> str:
    """Input: text containing numeric ranges -> Output: spoken range forms."""
    # 10-15 -> 10 15
    return re.sub(r'\b(\d+)\s*-\s*(\d+)\b', r'\1 \2', text)

def normalize_percentages(text: str) -> str:
    """Input: text containing % expressions -> Output: spoken percentage forms."""
    return text.replace('%', ' procent ')

def normalize_currency(text: str) -> str:
    """Input: text containing currency symbols/codes -> Output: spoken currency forms."""
    return text.replace('$', ' dollar ').replace('€', ' evro ')

def normalize_units(text: str) -> str:
    """Expand measurement-unit abbreviations into full Karakalpak words."""

    units = {
    "kg": "kilogramm",
    "km": "kilometr",
    "cm": "santimetr",
    "sm": "santimetr",
    "mm": "millimetr",
    "ml": "millilitr",
    "g": "gramm",
    "gr": "gramm",
    }

    for unit, spoken in units.items():
        text = re.sub(
            rf"(\d+(?:[.,]\d+)?)\s*{unit}\b",
            rf"\1 {spoken}",
            text,
            flags=re.IGNORECASE,
        )

    return text

def normalize_numbers(text: str) -> str:
    """Input: text containing numeric expressions -> Output: spoken number forms."""
    
    # 1. Handle float/decimal numbers first (e.g., 6,5 or 6.5 -> altı bútin bes)
    # Matches any digits separated by a comma or a dot.
    text = re.sub(
        r'\b(\d+)[.,](\d+)\b', 
        lambda m: f"{num_to_kk_words(int(m.group(1)))} bútin {num_to_kk_words(int(m.group(2)))}", 
        text
    )
    
    # 2. Handle all remaining standard integers
    text = re.sub(
        r'\d+', 
        lambda m: num_to_kk_words(int(m.group())), 
        text
    )
    
    return text

def normalize_abbreviations(text: str) -> str:
    """Input: text containing abbreviations -> Output: normalized/expanded forms."""
    
    # 1. Un-glue numeric abbreviations attached directly to numbers (e.g., 10mln -> 10 mln)
    text = re.sub(r'\b(\d+)(mln|mlrd)\b\.?', r'\1 \2', text, flags=re.IGNORECASE)
    
    # 2. Expand all standard abbreviations (handles optional dots like mln.)
    abbrs = {
        'mln': 'million',
        'mlrd': 'milliard',
        'mıń': 'mıń' # Just in case they abbreviate thousands similarly
    }
    
    for a, exp in abbrs.items():
        text = re.sub(fr'\b{a}\b\.?', exp, text, flags=re.IGNORECASE)
        
    return text

def normalize_initials(text: str) -> str:
    """Input: text containing initials such as A.Hasanov -> Output: normalized initials."""
    return re.sub(r'\b([A-Za-zÁ-Úá-ú])\.', r'\1 ', text)

def normalize_acronyms(text: str) -> str:
    """Input: text containing acronyms -> Output: configured spoken/expanded forms."""
    # If all caps sequence left, space them out (e.g. UZ -> U Z)
    return re.sub(r'\b([A-Z]{2,})\b', lambda m: " ".join(m.group(1)), text)

def normalize_math_symbols(text: str) -> str:
    """Input: text containing mathematical/scientific symbols -> Output: spoken forms."""
    return text.replace('+', ' qosıw ').replace('-', ' alıw ').replace('=', ' teń ').replace('*', ' kóbeytiw ')


#============================================================
# PUNCTUATION / FINAL CLEANUP
#============================================================

def normalize_repeated_punctuation(text: str) -> str:
    """Input: text with repeated punctuation -> Output: normalized punctuation."""
    return re.sub(r'([.?!,])\1+', r'\1', text)

def normalize_hyphens(text: str) -> str:
    """Input: text with remaining non-semantic hyphens -> Output: normalized text."""
    # Keep intra-word hyphens but remove standalone/dangling hyphens
    return re.sub(r'\s+-\s+', ' ', text)

def normalize_apostrophes(text: str) -> str:
    """Input: text with remaining apostrophes -> Output: normalized text."""
    # Orthography handles valid letter apostrophes (o', g'). Remove remaining dangling ones.
    return re.sub(r"['`‘’]", "", text)

def remove_punctuation(text: str) -> str:
    """Input: semantically normalized text -> Output: punctuation-normalized text."""
    # Keep only alphanumeric and whitespace
    return re.sub(r'[^\w\s]', ' ', text)


#============================================================
# VALIDATION
#============================================================

def validate_characters(text: str) -> Set[str]:
    """Input: normalized text -> Output: set of unexpected/invalid characters."""
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzáóǵńúíı ")
    return set(text) - allowed_chars

def is_empty_transcript(text: str) -> bool:
    """Input: normalized text -> Output: True if transcript is unusable/empty."""
    return len(text.strip()) == 0

def validate_text(text: str) -> Tuple[bool, str | None]:
    """Input: normalized text -> Output: (is_valid, reason)."""
    if is_empty_transcript(text):
        return False, "Empty transcript"
    invalid = validate_characters(text)
    if invalid:
        return False, f"Contains invalid characters: {invalid}"
    return True, None


#============================================================
# COMPLETE TEXT PIPELINE
#============================================================

def normalize_text(text: str) -> str:
    """
    Input: raw_text
    Output: final normalized ASR transcript.
    """
    if not text:
        return ""
        
    text = normalize_unicode(text)
    text = normalize_karakalpak_orthography(text)
    
    text = normalize_urls(text)
    text = normalize_emails(text)
    text = normalize_usernames_and_mentions(text)
    
    text = normalize_dates(text)
    text = normalize_times(text)
    text = normalize_roman_numerals(text)
    text = normalize_fractions(text)
    text = normalize_ranges(text)
    text = normalize_percentages(text)
    text = normalize_currency(text)
    text = normalize_units(text)
    text = normalize_ordinals(text)
    text = normalize_numbers(text)
    
    text = normalize_abbreviations(text)
    text = normalize_initials(text)
    text = normalize_acronyms(text)
    text = normalize_math_symbols(text)
    
    text = normalize_repeated_punctuation(text)
    text = normalize_hyphens(text)
    text = normalize_apostrophes(text)
    text = remove_punctuation(text)
    
    text = normalize_case(text)
    text = normalize_whitespace(text)
    
    return text


#============================================================
# DATASET / PARQUET PROCESSING
#============================================================

def normalize_parquet_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """
    Input: PyArrow RecordBatch containing: id | audio | raw_text
    Output: PyArrow RecordBatch containing: id | audio | raw_text | text
    """
    raw_texts = batch.column("raw_text").to_pylist()
    normalized_texts = [normalize_text(t) if t else "" for t in raw_texts]
    
    # Append the new "text" column
    text_array = pa.array(normalized_texts, type=pa.string())
    
    # Check if 'text' column already exists to prevent duplication on re-runs
    if "text" in batch.schema.names:
        idx = batch.schema.get_field_index("text")
        return batch.set_column(idx, "text", text_array)
    
    return batch.append_column("text", text_array)

def normalize_parquet_file(
    input_path: Path,
    output_path: Path, # Kept for signature compatibility, but overridden to ensure in-place
    batch_size: int = 5_000,
) -> int:
    """
    Input: input Parquet path
    Output: number of processed rows
    Strictly modifies the file IN PLACE as requested.
    """
    # Force output_path to match input_path to fulfill strict user requirement
    output_path = input_path
    
    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow
    
    # Define updated schema
    if "text" not in schema.names:
        schema = schema.append(pa.field("text", pa.string()))
        
    total_processed = 0
    
    # Write to a temporary file first to prevent corruption if process is interrupted
    fd, temp_path = tempfile.mkstemp(suffix=".parquet", dir=input_path.parent)
    os.close(fd)
    
    try:
        with pq.ParquetWriter(temp_path, schema) as writer:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                processed_batch = normalize_parquet_batch(batch)
                writer.write_batch(processed_batch)
                total_processed += processed_batch.num_rows
                
        # Safely overwrite the original file
        os.replace(temp_path, output_path)
    except Exception as e:
        # Cleanup temp file on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
        
    return total_processed

def iter_parquet_files(input_path: Path) -> Iterable[Path]:
    """
    Input: Parquet file or directory
    Output: iterator of Parquet file paths
    """
    if input_path.is_file() and input_path.suffix == '.parquet':
        yield input_path
    elif input_path.is_dir():
        yield from input_path.rglob("*.parquet")

def get_dataset_name(input_path: Path) -> str:
    """
    Input: input file/folder path
    Output: dataset name
    """
    return input_path.stem if input_path.is_file() else input_path.name

def normalize_dataset(
    input_path: Path,
    output_root: Path, # Kept for signature compatibility
    batch_size: int = 5_000,
) -> int:
    """
    Input: dataset file/folder
    Output: total number of processed rows
    """
    total_processed = 0
    for file_path in iter_parquet_files(input_path):
        # We ignore output_root and explicitly overwrite in-place
        total_processed += normalize_parquet_file(file_path, file_path, batch_size)
    return total_processed


#============================================================
# CLI
#============================================================

def main() -> None:
    """Input: command-line arguments -> Output: processed dataset."""
    import argparse
    parser = argparse.ArgumentParser(description="Karakalpak ASR Text Normalizer")
    parser.add_argument("input_path", type=Path, help="Path to input parquet file or directory")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for pyarrow processing")
    
    args = parser.parse_args()
    
    if not args.input_path.exists():
        print(f"Error: {args.input_path} does not exist.")
        return

    print(f"Normalizing dataset at: {args.input_path}")
    print("Files will be strictly modified in-place.")
    total = normalize_dataset(args.input_path, args.input_path, args.batch_size)
    print(f"Done! Successfully normalized {total} rows.")

if __name__ == "__main__":
    main()