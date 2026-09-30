# Karakalpak Alphabet (Modern Latin vs. Legacy Cyrillic)

This table maps the current official 34-letter Karakalpak Latin alphabet (adopted post-2016) to its legacy Cyrillic equivalents, along with pronunciation notes.

| Current Latin | Legacy Cyrillic | Notes / Pronunciation |
| :--- | :--- | :--- |
| **A a** | А а | /a/ |
| **Á á** | Ә ә | /æ/ (Front 'a', similar to 'a' in *cat*) |
| **B b** | Б б | /b/ |
| **C c** | Ц ц | /ts/ (Only in loanwords) |
| **Ch ch** | Ч ч | /tʃ/ (Digraph, treated as a single letter) |
| **D d** | Д д | /d/ |
| **E e** | Е е / Э э | /e/ |
| **F f** | Ф ф | /f/ (Only in loanwords) |
| **G g** | Г г | /g/ |
| **Ǵ ǵ** | Ғ ғ | /ʁ/ (Voiced uvular fricative, similar to French 'r') |
| **H h** | Ҳ ҳ | /h/ (Glottal fricative) |
| **X x** | Х х | /x/ (Voiceless velar fricative, like 'ch' in *Bach*) |
| **I i** | И и | /i/ |
| **Í ı** | Ы ы | /ɯ/ (Dotless 'i', pronounced deep in the throat) |
| **J j** | Ж ж | /ʒ/ or /dʒ/ |
| **K k** | К к | /k/ |
| **Q q** | Қ қ | /q/ (Voiceless uvular stop, deeper than 'k') |
| **L l** | Л л | /l/ |
| **M m** | М м | /m/ |
| **N n** | Н н | /n/ |
| **Ń ń** | Ң ң | /ŋ/ (Like 'ng' in *sing*) |
| **O o** | О о | /o/ |
| **Ó ó** | Ө ө | /œ/ (Front 'o') |
| **P p** | П п | /p/ |
| **R r** | Р р | /r/ |
| **S s** | С с | /s/ |
| **Sh sh** | Ш ш | /ʃ/ (Digraph, treated as a single letter) |
| **T t** | Т т | /t/ |
| **U u** | У у | /u/ |
| **Ú ú** | Ү ү | /y/ (Front 'u') |
| **V v** | В в | /v/ (Only in loanwords) |
| **W w** | Ў ў | /w/ (Bilabial glide, like 'w' in *water*) |
| **Y y** | Й й | /j/ (Consonant 'y' as in *yes*) |
| **Z z** | З з | /z/ |

## Data Processing Notes for STT:
* **Apostrophe datasets:** The 1994–2016 Latin alphabet used apostrophes (e.g., `G' g'`, `N' n'`, `O' o'`). You may need a normalization script to convert `a'` to `á`, `o'` to `ó`, etc., for older texts.
* **Cyrillic soft/hard signs:** The Cyrillic soft sign (Ь ь) and hard sign (Ъ ъ) have no direct equivalents in the modern Latin alphabet. Cyrillic Я and Ю are mapped to `Ya` and `Yu` respectively.