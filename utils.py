import re

def chunk_text(text: str, chunk_size: int) -> list[str]:
    """
    Splits the text into chunks of a specified size, respecting sentence boundaries.
    OpenAI's TTS API has a limit of 4096 characters. We chunk to stay below this.

    Args:
        text: The input text to be chunked.
        chunk_size: The desired maximum size of each chunk.

    Returns:
        A list of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current_chunk = ""

    # Split text by multiple sentence delimiters while preserving them
    sentence_pattern = r"([.!?]+\s*)"
    parts = re.split(sentence_pattern, text)

    # Recombine sentences with their delimiters
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i]
        delimiter = parts[i + 1] if i + 1 < len(parts) else ""
        if sentence.strip():  # Only add non-empty sentences
            sentences.append(sentence + delimiter)

    # Handle the last part if it doesn't have a delimiter
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If a single sentence exceeds chunk_size, split it by words
        if len(sentence) > chunk_size:
            words = sentence.split()
            word_chunk = ""
            for word in words:
                if len(word_chunk) + len(word) + 1 <= chunk_size:
                    word_chunk += (" " if word_chunk else "") + word
                else:
                    if word_chunk:
                        chunks.append(word_chunk)
                    word_chunk = word
            if word_chunk:
                chunks.append(word_chunk)
            continue

        # Check if adding this sentence would exceed the chunk size
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # If the chunk is full, store it and start a new one
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    # Add the last remaining chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks