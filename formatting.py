import re


def __strip(text: str) -> str:
    """
    Strips leading and trailing whitespace from the text attribute.

    This method is used to remove leading and trailing whitespace characters
    from the string value stored in the `text` attribute.

    Args:
        text (str): The input string to be stripped.

    Returns:
        str: A new string resulting from the stripped text.
    """
    return text.strip()


def __remove_citations(text: str) -> str:
    """
    Removes bracketed numerical citations from the text property.

    Returns:
      A string with the citations removed.
    """
    # This regular expression finds one or more digits enclosed in square brackets.
    # \[\d+\]:
    # \[ and \] match the literal square brackets.
    # \d+ matches one or more digits (0-9).
    citation_pattern = r"\[\d+\]"

    # re.sub() finds all occurrences of the pattern and replaces them with an empty string.
    return re.sub(citation_pattern, "", text)


def format_for_tts(text: str) -> str:
    """
    Applies all formatting steps to the text property.

    Returns:
      A formatted string.
    """
    out = __strip(text)
    out = __remove_citations(out)
    return out

