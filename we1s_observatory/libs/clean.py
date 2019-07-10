import unicodedata

from ftfy import fix_text

from we1s_observatory.libs import regex


def content_field_standardize(data):
    """Takes a dict (json data) and standardizes
    the location of its content field.
    """
    changed_content = False
    # lexis-nexis files -- move content-unscrubbed to content
    if "content_unscrubbed" in data:
        data["content"] = data.pop("content_unscrubbed")
        changed_content = True
    if "content_scrubbed" in data:
        # lexis-nexis -- remove useless scrubbed content fields
        if "content" in data:
            data.pop("content_scrubbed")
            changed_content = True
        # Reddit-collection files -- move content-scrubbed to content
        elif "content" not in data:
            data["content"] = data.pop("content_scrubbed")
            changed_content = True
    return changed_content


def remove_accents(text, method="unicode"):
    """Remove accents from any accented unicode characters in a string.

    Either transforms them into ascii equivalents or removes them entirely.
    Parameters:
    - text (str): raw text
    - method ({'unicode', 'ascii'}): if 'unicode', remove accented
        char for any unicode symbol with a direct ASCII equivalent; if 'ascii',
        remove accented char for any unicode symbol.
        NB: the 'ascii' method is notably faster but less effective than 'unicode'.
    Returns:
        str
    Raises:
        ValueError: if ``method`` is not in {'unicode', 'ascii'}
    """
    if method == "unicode":
        return "".join(
            c
            for c in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(c)
        )
    elif method == "ascii":
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("ascii")
        )
    else:
        msg = '`method` must be either "unicode" and "ascii", not {}'.format(method)
        raise ValueError(msg)


def scrub(text, unicode_normalization="NFC", accent_removal_method="unicode"):
    """Normalize whitespace and and bad unicode, and remove accents.

    Parameters:
    - unicode_normalization: The ftfy.fix_text() `normalization` parameter.
    - accent_removal_method: The Doc.remove_accents() `method` parameter.

    """
    # Change multiple spaces to one and multiple line breaks to one.
    # Also strip leading/trailing whitespace.
    text = regex.NONBREAKING_SPACE_REGEX.sub(
        " ", regex.LINEBREAK_REGEX.sub(r"\n", text)
    ).strip()
    # Combine characters and diacritics written using separate code points
    text = fix_text(text, normalization=unicode_normalization)
    text = remove_accents(text, method=accent_removal_method)
    return text
