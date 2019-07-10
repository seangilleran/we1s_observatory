import regex as re

# Constants for the preprocessing functions
LINEBREAK_REGEX = re.compile(r"((\r\n)|[\n\v])+")
NONBREAKING_SPACE_REGEX = re.compile(r"(?!\n)\s+")
PREFIX_RE = re.compile(r"""^[\[\]\("'\.,;:-]""")
SUFFIX_RE = re.compile(r"""[\[\]\)"'\.,;:-]$""")
INFIX_RE = re.compile(r"""[-~]""")
SIMPLE_URL_RE = re.compile(r"""^https?://""")
MONTHS_RE = re.compile(
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|Dec(?:ember)?)"
)
