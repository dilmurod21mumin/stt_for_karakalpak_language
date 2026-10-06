"""Comprehensive tests for text_normalize.py.

Covers every normalization category defined in the module.
Run with:  python -m pytest test_text_normalizer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add both the project root and a potential 'scripts' folder to Python's path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

# Import directly from the text_normalize module
try:
    from text_normalizer import (
        num_to_kk_words,
        normalize_unicode,
        normalize_karakalpak_orthography,
        normalize_case,
        normalize_whitespace,
        normalize_urls,
        normalize_emails,
        normalize_usernames_and_mentions,
        normalize_dates,
        normalize_times,
        normalize_roman_numerals,
        normalize_ordinals,
        normalize_fractions,
        normalize_ranges,
        normalize_percentages,
        normalize_currency,
        normalize_units,
        normalize_numbers,
        normalize_abbreviations,
        normalize_initials,
        normalize_acronyms,
        normalize_math_symbols,
        normalize_repeated_punctuation,
        normalize_hyphens,
        normalize_apostrophes,
        remove_punctuation,
        validate_characters,
        validate_text,
        is_empty_transcript,
        normalize_text
    )
except ImportError:
    pytest.skip("text_normalize.py not found in path", allow_module_level=True)


# ============================================================================
# num_to_kk_words
# ============================================================================

class TestNumToKkWords:
    def test_zero(self):
        assert num_to_kk_words(0) == "nol"

    def test_single_digits(self):
        assert num_to_kk_words(1) == "bir"
        assert num_to_kk_words(5) == "bes"
        assert num_to_kk_words(9) == "toǵız"

    def test_teens(self):
        assert num_to_kk_words(10) == "on"
        assert num_to_kk_words(11) == "on bir"
        assert num_to_kk_words(19) == "on toǵız"

    def test_tens(self):
        assert num_to_kk_words(20) == "jigirma"
        assert num_to_kk_words(25) == "jigirma bes"
        assert num_to_kk_words(99) == "toqsan toǵız"

    def test_hundreds(self):
        assert "júz" in num_to_kk_words(100)
        assert num_to_kk_words(200) == "eki júz"
        assert num_to_kk_words(140) == "bir júz qırq"
        assert num_to_kk_words(305) == "úsh júz bes"

    def test_thousands(self):
        assert num_to_kk_words(1000) == "mıń"
        assert num_to_kk_words(2000) == "eki mıń"
        assert num_to_kk_words(1997) == "bir mıń toǵız júz toqsan jeti"
        assert num_to_kk_words(2026) == "eki mıń jigirma altı"

    def test_complex_thousands(self):
        # Additional edge cases for inner zeros
        assert num_to_kk_words(1001) == "bir mıń bir"
        assert num_to_kk_words(1024) == "bir mıń jigirma tórt"

    def test_large_numbers(self):
        assert "million" in num_to_kk_words(1_000_000)
        assert "milliard" in num_to_kk_words(1_000_000_000)

    def test_float_number(self):
        result = normalize_numbers("6.5")
        assert result == "altı bútin bes"

    def test_decimal_comma(self):
        result = normalize_numbers("3,14")
        assert result == "úsh bútin on tórt"

    def test_thousands_separator(self):
        result = normalize_numbers("1000")
        assert result == "mıń"

    def test_thousands_and_decimals_combined(self):
        result = normalize_numbers("1500.5")
        assert result == "bir mıń bes júz bútin bes"

# ============================================================================
# Unicode normalization
# ============================================================================

class TestUnicodeNormalization:
    def test_nfc(self):
        # Composed vs decomposed forms
        text = "á"  # 'a' + combining acute
        result = normalize_unicode(text)
        assert "́" not in result


# ============================================================================
# Whitespace normalization
# ============================================================================

class TestWhitespace:
    def test_multiple_spaces(self):
        assert normalize_whitespace("  Sálem    qalaysız?   ") == "Sálem qalaysız?"

    def test_tabs(self):
        assert normalize_whitespace("a\tb") == "a b"

    def test_newlines(self):
        assert normalize_whitespace("a\nb") == "a b"


# ============================================================================
# Orthographic normalization
# ============================================================================

class TestOrthography:
    def test_apostrophe_g(self):
        assert normalize_karakalpak_orthography("g'alaba") == "ǵalaba"
        assert normalize_karakalpak_orthography("g`alaba") == "ǵalaba"

    def test_apostrophe_o(self):
        assert normalize_karakalpak_orthography("o'zbekiston") == "ózbekiston"

    def test_apostrophe_u(self):
        assert normalize_karakalpak_orthography("u'lken") == "úlken"

    def test_uppercase_variant(self):
        assert normalize_karakalpak_orthography("G'arbi") == "Ǵarbi"
        assert normalize_karakalpak_orthography("O'ZBEK") == "ÓZBEK"

    def test_cyrillic_to_latin(self):
        assert normalize_karakalpak_orthography("Салем") == "Salem"
        
    def test_mixed_cyrillic(self):
        # Checking if capital Cyrillic letters map correctly to capital Latin
        assert normalize_karakalpak_orthography("Қарақалпақстан") == "Qaraqalpaqstan"
        assert normalize_karakalpak_orthography("ҒӘРЕЗСИЗЛИК") == "ǴÁREZSIZLIK"

    def test_no_change_valid(self):
        text = "sálem qalaysız"
        assert normalize_karakalpak_orthography(text) == text


# ============================================================================
# Case normalization
# ============================================================================

class TestCase:
    def test_all_upper(self):
        assert normalize_case("SÁLEM") == "sálem"

    def test_title(self):
        assert normalize_case("Sálem") == "sálem"

    def test_already_lower(self):
        assert normalize_case("sálem") == "sálem"


# ============================================================================
# URL normalization
# ============================================================================

class TestURLs:
    def test_https_replaced(self):
        result = normalize_urls("visit https://example.com today")
        assert "example.com" not in result
        assert "silteme" in result

    def test_www_replaced(self):
        result = normalize_urls("visit www.example.com today")
        assert "example.com" not in result
        assert "silteme" in result


# ============================================================================
# Email normalization
# ============================================================================

class TestEmails:
    def test_replaced(self):
        result = normalize_emails("mail user@example.com please")
        assert "@" not in result
        assert "elektron pochta" in result


# ============================================================================
# Mentions and hashtags
# ============================================================================

class TestMentionsAndHashtags:
    def test_remove_mentions(self):
        assert "@user" not in normalize_usernames_and_mentions("hello @user")

    def test_remove_hashtags(self):
        assert "#topic" not in normalize_usernames_and_mentions("hello #topic")


# ============================================================================
# Date normalization
# ============================================================================

class TestDates:
    def test_day_month_year(self):
        result = normalize_dates("01-09-2026")
        assert "01 09 2026" in result

    def test_numeric_date_slash(self):
        result = normalize_dates("01/09/2026")
        assert "01 09 2026" in result

    def test_numeric_date_dot(self):
        result = normalize_dates("01.09.2026")
        assert "01 09 2026" in result


# ============================================================================
# Time normalization
# ============================================================================

class TestTimes:
    def test_basic(self):
        result = normalize_times("12:30")
        assert "12 30" in result
        
    def test_morning_time(self):
        result = normalize_times("09:05")
        assert "09 05" in result


# ============================================================================
# Roman Numerals
# ============================================================================

class TestRomanNumerals:
    def test_romans(self):
        assert normalize_roman_numerals("XX") == "20"
        assert normalize_roman_numerals("V") == "5"

    def test_complex_romans(self):
        assert normalize_roman_numerals("XIV") == "14"
        assert normalize_roman_numerals("IX") == "9"


# ============================================================================
# Fraction normalization
# ============================================================================

class TestFractions:
    def test_general(self):
        result = normalize_fractions("1/2")
        assert "1 den 2" in result

    def test_complex_fraction(self):
        result = normalize_fractions("3/4")
        assert "3 den 4" in result


# ============================================================================
# Range normalization
# ============================================================================

class TestRanges:
    def test_basic(self):
        result = normalize_ranges("5-10")
        assert "5 10" in result

    def test_with_spaces(self):
        result = normalize_ranges("5 - 10")
        assert "5 10" in result


# ============================================================================
# Percentage normalization
# ============================================================================

class TestPercentages:
    def test_integer(self):
        result = normalize_percentages("25%")
        assert "25 procent" in result or "25  procent " in result


# ============================================================================
# Currency normalization
# ============================================================================

class TestCurrency:
    def test_dollar(self):
        result = normalize_currency("$10")
        assert "dollar" in result

    def test_euro(self):
        result = normalize_currency("€100")
        assert "evro" in result


# ============================================================================
# Unit normalization
# ============================================================================

class TestUnits:
    def test_kg(self):
        result = normalize_units("10 kg")
        assert "10 kilogramm" in result

    def test_km(self):
        result = normalize_units("100 km")
        assert "100 kilometr" in result
        
    def test_more_units(self):
        assert "5 millilitr" in normalize_units("5 ml")
        assert "10 gramm" in normalize_units("10 gr")
        assert "15 santimetr" in normalize_units("15 sm")


# ============================================================================
# Number normalization
# ============================================================================
class TestNumbers:
    def test_integer(self):
        result = normalize_numbers("25")
        assert result == "jigirma bes"

    def test_large(self):
        result = normalize_numbers("1000000")
        assert "million" in result

    def test_float_number_dot(self):
        result = normalize_numbers("6.5")
        assert result == "altı bútin bes"

    def test_float_number_comma(self):
        result = normalize_numbers("6,5")
        assert result == "altı bútin bes"

    def test_decimal_comma_complex(self):
        # 12,500 should be read as 12 point 500, NOT 12 thousand 500
        result = normalize_numbers("12,500")
        assert result == "on eki bútin bes júz"

# ============================================================================
# Ordinal normalization
# ============================================================================

class TestOrdinals:
    def test_ordinal_noun(self):
        # number - noun
        result = normalize_ordinals("1-mektep")
        assert result == "birinshi mektep"

    def test_ordinal_year(self):
        # number - noun (large number)
        result = normalize_ordinals("2026-jıl")
        assert result == "eki mıń jigirma altınshı jıl"

    def test_explicit_suffix(self):
        # explicitly written suffixes
        assert normalize_ordinals("1-shi") == "birinshi"
        assert normalize_ordinals("2-nshi") == "ekinshi"

        assert normalize_ordinals("1-mart") == "birinshi mart"
        assert normalize_ordinals("2-aprel") == "ekinshi aprel"
        assert normalize_ordinals("5-sanli") == "besinshi sanli"
        assert normalize_ordinals("2026-jil") == "eki mıń jigirma altınshı jil"

    def test_number_number_ignored(self):
        # number - number (must do nothing here)
        result = normalize_ordinals("5-10")
        assert result == "5-10"

# ============================================================================
# Abbreviation normalization
# ============================================================================

class TestAbbreviations:
    def test_numeric_abbreviations(self):
        # Standard spacing
        assert normalize_abbreviations("10 mln") == "10 million"
        assert normalize_abbreviations("5 mlrd") == "5 milliard"
        
        # With trailing periods
        assert normalize_abbreviations("10 mln.") == "10 million"
        assert normalize_abbreviations("5 mlrd.") == "5 milliard"
        
        # Glued directly to the digit
        assert normalize_abbreviations("10mln") == "10 million"
        assert normalize_abbreviations("5mlrd.") == "5 milliard"


# ============================================================================
# Initials normalization
# ============================================================================
class TestInitials:
    def test_initials(self):
        result = normalize_initials("A.Hasanov")
        assert result == "A Hasanov"


# ============================================================================
# Acronym normalization
# ============================================================================

class TestAcronyms:
    def test_acronym_space(self):
        result = normalize_acronyms("UZ")
        assert result == "U Z"


# ============================================================================
# Symbol normalization
# ============================================================================

class TestSymbols:
    def test_plus(self):
        result = normalize_math_symbols("5 + 3")
        assert "qosıw" in result

    def test_equals(self):
        result = normalize_math_symbols("x = 5")
        assert "teń" in result

    def test_multiply(self):
        result = normalize_math_symbols("10 * 2")
        assert "kóbeytiw" in result

    def test_full_equation(self):
        result = normalize_math_symbols("2 + 2 = 4")
        assert "qosıw" in result
        assert "teń" in result


# ============================================================================
# Repeated punctuation
# ============================================================================

class TestRepeatedPunctuation:
    def test_exclamation(self):
        assert normalize_repeated_punctuation("Sálem!!!") == "Sálem!"

    def test_question(self):
        assert normalize_repeated_punctuation("ne???") == "ne?"


# ============================================================================
# Generic punctuation
# ============================================================================

class TestPunctuation:
    def test_comma_to_space(self):
        result = remove_punctuation("Sálem,qalaysız?")
        assert "," not in result

    def test_brackets(self):
        result = remove_punctuation("(test)")
        assert "(" not in result
        assert ")" not in result


# ============================================================================
# Residual hyphens
# ============================================================================

class TestResidualHyphens:
    def test_hyphen_to_space(self):
        assert normalize_hyphens("a - b") == "a b"


# ============================================================================
# Apostrophes
# ============================================================================

class TestApostrophes:
    def test_remove_remaining(self):
        assert normalize_apostrophes("it's") == "its"
        assert normalize_apostrophes("`hello`") == "hello"


# ============================================================================
# Validation
# ============================================================================

class TestValidation:
    def test_valid_text(self):
        is_valid, reason = validate_text("sálem qalaysız")
        assert is_valid is True
        assert reason is None

    def test_empty(self):
        is_valid, reason = validate_text("   ")
        assert is_valid is False
        assert reason == "Empty transcript"

    def test_unknown_chars_reported(self):
        is_valid, reason = validate_text("sálem ©")
        assert is_valid is False
        assert "©" in reason


# ============================================================================
# Empty Transcript Function
# ============================================================================

class TestEmptyTranscript:
    def test_is_empty(self):
        assert is_empty_transcript("") is True
        assert is_empty_transcript("   ") is True
        assert is_empty_transcript("a") is False


# ============================================================================
# Full pipeline: normalize_text
# ============================================================================

class TestNormalizeText:
    def test_basic_punctuation(self):
        result = normalize_text("Sálem, qalaysız?")
        assert result == "sálem qalaysız"

    def test_whitespace(self):
        result = normalize_text("  Sálem    qalaysız  ")
        assert result == "sálem qalaysız"

    def test_initials(self):
        result = normalize_text("A.Hasanov")
        assert "a hasanov" in result
        assert "." not in result

    def test_units_kg(self):
        result = normalize_text("10 kg")
        assert "on" in result
        assert "kilogramm" in result

    def test_percentages(self):
        result = normalize_text("25%")
        assert "jigirma bes" in result
        assert "procent" in result

    def test_currency_dollar(self):
        result = normalize_text("$10")
        assert "dollar" in result
        assert "on" in result

    def test_date_numeric(self):
        result = normalize_text("01-09-2026")
        # 01 09 2026 -> bir toǵız eki mıń jigirma altı
        assert "bir toǵız eki mıń jigirma altı" in result or "toǵız" in result

    def test_time(self):
        result = normalize_text("12:30")
        # 12 30 -> on eki otız
        assert "on eki" in result
        assert "otız" in result

    def test_fraction(self):
        result = normalize_text("1/2")
        assert "bir den eki" in result

    def test_range(self):
        result = normalize_text("5-10")
        assert "bes" in result
        assert "on" in result

    def test_url_removed(self):
        result = normalize_text("see https://example.com now")
        assert "example" not in result
        assert "silteme" in result

    def test_email_removed(self):
        result = normalize_text("mail user@example.com")
        assert "@" not in result
        assert "elektron pochta" in result

    def test_acronym_kept_or_spaced(self):
        result = normalize_text("UZ")
        # UZ -> U Z -> u z
        assert result == "u z"

    def test_repeated_punctuation(self):
        result = normalize_text("Sálem!!!")
        assert "!" not in result
        assert result == "sálem"

    def test_case_normalization(self):
        result = normalize_text("SÁLEM")
        assert result == "sálem"

    def test_preserves_karakalpak_chars(self):
        text = "áóúıǵń"
        result = normalize_text(text)
        assert result == "áóúıǵń"

    def test_idempotency(self):
        raw = "Sálem, 25% — $10!"
        first = normalize_text(raw)
        second = normalize_text(first)
        assert first == second, f"Not idempotent: {first!r} != {second!r}"

    def test_equation_full(self):
        # 2 + 2 = 4 -> eki qosıw eki teń tórt
        result = normalize_text("2 + 2 = 4")
        assert result == "eki qosıw eki teń tórt"
        
    def test_cyrillic_integration(self):
        # Салем! -> salem
        result = normalize_text("Салем!")
        assert result == "salem"


# ============================================================================
# Mixed sentences
# ============================================================================

class TestMixedSentences:
    def test_units_and_date(self):
        result = normalize_text("01-09-2026 — 10 kg, 25%!")
        assert "kilogramm" in result
        assert "procent" in result
        assert "!" not in result
        assert "," not in result

    def test_initials_and_numbers(self):
        result = normalize_text("A.Hasanov arrived on 01-09-2026")
        assert "hasanov" in result
        assert "." not in result

    def test_currency_and_percentages(self):
        result = normalize_text("$100 discount, 25% off")
        assert "dollar" in result
        assert "procent" in result

    def test_pure_karakalpak_text(self):
        raw = (
            "aqsa degen aqımaqqa qúdai qúlqımdı salmaǵan menen "
            "aqsa dámetetin ádbeket bolsań áida jolıń asıq maǵan"
        )
        result = normalize_text(raw)
        assert result == raw 

    def test_range_with_unit(self):
        result = normalize_text("10-20 kg")
        assert "kilogramm" in result
        assert "kg" not in result
        assert "on jigirma" in result

    def test_complex_mixed_2(self):
        # Tests fractions, time, unit, and cyrillic in one pass
        raw = "1/2 saattay waqıt ketti 10 ml suw, 09:15"
        result = normalize_text(raw)
        assert "bir den eki" in result
        assert "millilitr" in result
        assert "toǵız on bes" in result


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_whitespace(self):
        assert normalize_text("   \t\n  ") == ""

    def test_only_punctuation(self):
        assert normalize_text("...!!!???") == ""

    def test_single_number(self):
        result = normalize_text("7")
        assert result == "jeti"

    def test_very_long_number(self):
        result = normalize_text("1234567890")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_none_like_input(self):
        result = normalize_text("nan")
        assert result == "nan"

    def test_already_normalised(self):
        text = "sálem qalaysız"
        assert normalize_text(text) == text