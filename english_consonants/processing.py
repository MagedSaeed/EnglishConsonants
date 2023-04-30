import re
import string

ENGLISH_LETTERS = string.ascii_lowercase

def process_english(text):
    # add spaces between punctuations, if there is not
    text = text.lower()
    text = re.sub(
        r"""([.,!?()\/\\،"'\{\}\(\)\[\]؟<>`؛=+\-\*\&\^\%\$\#\@\!])""",
        r" \1 ",
        text,
    )
    # remove any non arabic character
    text = "".join(
        [c for c in text if c in ENGLISH_LETTERS or c.isspace()]
    )  # keep only english chars and spaces
    text = re.sub("\s{2,}", " ", text).strip()  # remove multiple spaces
    """
      interestingly, there is a difference betwen re.sub('\s+',' ',s) and re.sub('\s{2,}',' ',s)
      the first one remove newlines while the second does not.
    """
    return text.strip()


def strip_vowels(text):
    text_with_no_vowels = re.sub(r'[AEIOU]','',text,flags=re.IGNORECASE)
    return text_with_no_vowels