import os
import re
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Initialize local embedding model
try:
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    MODEL_AVAILABLE = True
    print("Local BGE-base model loaded successfully")
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"Warning: Failed to load BGE model. Embeddings will be disabled. Error: {e}")


def get_embedding(text: str):
    """Get embedding for text using local BGE model"""
    if not MODEL_AVAILABLE:
        return None
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def get_embeddings_batch(texts: List[str]) -> List[Any]:
    """Embed many chunks in one batched call instead of one .encode() per
    chunk — at full-corpus scale (tens of thousands of judgments) unbatched
    per-chunk calls turn into hundreds of thousands of individual GPU calls,
    dominated by per-call overhead rather than actual compute."""
    if not MODEL_AVAILABLE or not texts:
        return [None] * len(texts)
    try:
        embeddings = embedding_model.encode(
            texts, convert_to_numpy=True, batch_size=128, show_progress_bar=False
        )
        return [e.tolist() for e in embeddings]
    except Exception as e:
        print(f"Error generating batch embeddings: {e}")
        return [None] * len(texts)


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', pdf_path, '-'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0:
            print(f"pdftotext error: {result.stderr}")
        if result.stdout:
            print(f"pdftotext got {len(result.stdout)} chars")
            return result.stdout
    except FileNotFoundError:
        print("pdftotext not found - install poppler-utils")
    except Exception as e:
        print(f"pdftotext failed: {e}")
    return ""


def clean_judgment_text(text: str) -> str:
    """
    Remove repetitive JUDIS boilerplate headers that pdftotext picks up on
    every page. These appear in two forms depending on how pdftotext lays
    out the columns:

      Form 1 (single line):
        http://JUDIS.NIC.IN    SUPREME COURT OF INDIA    Page N of N

      Form 2 (split across lines):
        http://JUDIS.NIC.IN
        SUPREME COURT OF INDIA    Page N of N
    """
    # Form 1 – all on one line
    cleaned = re.sub(
        r'http://JUDIS\.NIC\.IN\s+SUPREME COURT OF INDIA\s+Page \d+ of \d+\s*',
        '',
        text
    )
    # Form 2 – URL alone on a line
    cleaned = re.sub(r'http://JUDIS\.NIC\.IN\s*\n?', '', cleaned)
    # Form 2 – "SUPREME COURT OF INDIA  Page N of N" leftover
    cleaned = re.sub(r'SUPREME COURT OF INDIA\s+Page \d+ of \d+\s*\n?', '', cleaned)

    # Collapse runs of blank lines created by the removals
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def parse_judgment_metadata(text: str, filename: str) -> Dict[str, Any]:
    """
    Extract metadata from judgment text.
    Expects already-cleaned text — no internal cleaning is done here.
    """
    metadata = {
        'petitioner': 'Unknown',
        'respondent': 'Unknown',
        'court': 'Supreme Court of India',
        'date_of_judgment': None,
        'bench': [],
        'citations': {}
    }

    header_lines = text.split('\n')[:20]

    petitioner_name = None
    respondent_name = None

    for line_index, current_line in enumerate(header_lines):
        stripped_line = current_line.strip()

        if stripped_line == 'PETITIONER:':
            petitioner_name = _extract_name_after_header(
                header_lines, line_index + 1,
                ['RESPONDENT:', 'Vs.', 'vs.', 'V.']
            )
        elif stripped_line == 'RESPONDENT:':
            respondent_name = _extract_name_after_header(
                header_lines, line_index + 1,
                ['DATE OF JUDGMENT:', 'BENCH:', 'ACT:']
            )

    if petitioner_name:
        metadata['petitioner'] = petitioner_name
    if respondent_name:
        metadata['respondent'] = respondent_name

    date_pattern = r'DATE OF JUDGMENT:\s*(\d{2}/\d{2}/\d{4})'
    date_match = re.search(date_pattern, text)
    if date_match:
        date_str = date_match.group(1)
        try:
            date_obj = datetime.strptime(date_str, '%d/%m/%Y')
            metadata['date_of_judgment'] = date_obj.strftime('%Y-%m-%d')
        except Exception:
            pass

    bench_pattern = r'BENCH:\s*\[?([^\]\n]+)'
    bench_match = re.search(bench_pattern, text)
    if bench_match:
        bench_text = bench_match.group(1).strip()
        judges = re.findall(
            r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z])?)\s*\.?\s*J\.?',
            bench_text
        )
        if judges:
            metadata['bench'] = judges

    return metadata


def _extract_name_after_header(
    lines: List[str], start_index: int, stop_markers: List[str]
) -> str | None:
    """Extract the name that appears on the line(s) after a header keyword."""
    for next_line_index in range(start_index, len(lines)):
        line_content = lines[next_line_index].strip()
        if line_content in stop_markers:
            break
        if (
            line_content
            and not line_content.startswith('ACT:')
            and not line_content.startswith('BENCH:')
            and not line_content.startswith('Vs.')
            and not line_content.startswith('vs.')
            and not line_content.startswith('V.')
        ):
            return line_content
    return None


def chunk_judgment_text(text: str) -> List[Tuple[str, str]]:
    """Split judgment text into logical chunks based on common legal judgment structure."""

    section_identifiers = {
        'facts': [
            r'(?i)facts of the case',
            r'(?i)\bfacts\b',
            r'(?i)background',
            r'(?i)facts and circumstances',
            r'(?i)the facts are',
            r'(?i)brief facts'
        ],
        'issues': [
            r'(?i)issues? arising',
            r'(?i)questions? for consideration',
            r'(?i)points? for determination',
            r'(?i)the issue is',
            r'(?i)issues framed'
        ],
        'arguments': [
            r'(?i)arguments? advanced',
            r'(?i)submissions?',
            r'(?i)contentions?',
            r'(?i)learned counsel',
            r'(?i)learned senior counsel',
            r'(?i)learned attorney'
        ],
        'ratio': [
            r'(?i)ratio decidendi',
            r'(?i)legal principles',
            r'(?i)principles laid down',
            r'(?i)the law is',
            r'(?i)the legal position',
            r'(?i)precedent',
            r'(?i)jurisprudence'
        ],
        'judgment': [
            r'(?i)\bjudgment\b',
            r'(?i)\border\b',
            r'(?i)\bdecision\b',
            r'(?i)\bconclusion\b',
            r'(?i)\bholding\b',
            r'(?i)we hold',
            r'(?i)we conclude',
            r'(?i)\baccordingly\b',
            r'(?i)in the result'
        ]
    }

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []

    # Long stretches of text with no recognized section header (common in
    # older judgments) would otherwise accumulate into a single oversized
    # chunk — cap it so retrieval stays granular and the full chunk can
    # actually fit in the LLM's context window later.
    max_chunk_chars = 3000

    current_section = 'facts'
    current_section_text = []
    current_section_length = 0

    for paragraph in paragraphs:
        identified_section = _identify_section_type(paragraph, section_identifiers)

        if identified_section and identified_section != current_section and current_section_text:
            chunks.append((current_section, '\n'.join(current_section_text)))
            current_section_text = []
            current_section_length = 0
            current_section = identified_section

        if current_section_text and current_section_length + len(paragraph) > max_chunk_chars:
            chunks.append((current_section, '\n'.join(current_section_text)))
            current_section_text = []
            current_section_length = 0

        if len(paragraph) > max_chunk_chars:
            # A single "paragraph" (blank-line-delimited span) can itself be
            # huge when pdftotext doesn't preserve blank lines well — split
            # it on word boundaries so nothing bypasses the size cap.
            words = paragraph.split()
            piece = []
            piece_length = 0
            for word in words:
                if piece_length + len(word) + 1 > max_chunk_chars and piece:
                    chunks.append((current_section, ' '.join(piece)))
                    piece = []
                    piece_length = 0
                piece.append(word)
                piece_length += len(word) + 1
            if piece:
                current_section_text = [' '.join(piece)]
                current_section_length = piece_length
            continue

        current_section_text.append(paragraph)
        current_section_length += len(paragraph)

    if current_section_text:
        chunks.append((current_section, '\n'.join(current_section_text)))

    if not chunks:
        chunks = _create_size_based_chunks(text)

    return chunks


def _identify_section_type(
    paragraph: str, section_patterns: Dict[str, List[str]]
) -> str | None:
    """Identify which section a paragraph belongs to based on patterns."""
    for section_type, patterns in section_patterns.items():
        for pattern in patterns:
            if re.search(pattern, paragraph):
                return section_type
    return None


def _create_size_based_chunks(text: str) -> List[Tuple[str, str]]:
    """Create fixed-size chunks when no clear section headings are found."""
    words = text.split()
    chunk_size = 500
    chunks = []

    for start_index in range(0, len(words), chunk_size):
        chunk_words = words[start_index:start_index + chunk_size]
        chunk_text = ' '.join(chunk_words)

        position_ratio = start_index / len(words) if words else 0
        if position_ratio < 0.3:
            section = 'facts'
        elif position_ratio < 0.6:
            section = 'issues'
        elif position_ratio < 0.8:
            section = 'arguments'
        else:
            section = 'judgment'

        chunks.append((section, chunk_text))

    return chunks


def ingest_judgment_from_pdf(pdf_path: str, conn) -> int | None:
    """Ingest a single PDF judgment into the database, using a connection
    shared across the whole ingestion run (see main()) rather than opening
    a fresh one per document — at full-corpus scale that's tens of
    thousands of avoidable connection setup/teardown round-trips."""
    print(f"Processing {pdf_path}...")

    # 1. Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
        return None

    # 2. Clean BEFORE anything else — metadata parsing, chunking, and DB storage
    #    all operate on the same clean text.
    text = clean_judgment_text(raw_text)
    print(f"Text length after cleaning: {len(text)}")
    print(f"Text preview: {text[:200]}...")

    # 3. Parse metadata from clean text
    metadata = parse_judgment_metadata(text, os.path.basename(pdf_path))

    cur = conn.cursor()

    try:
        # 4. Insert judgment — store the clean text, not the raw text
        cur.execute(
            """
            INSERT INTO judgments
                (petitioner, respondent, court, date_of_judgment, bench, citations, judgment_text)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                metadata['petitioner'],
                metadata['respondent'],
                metadata['court'],
                metadata['date_of_judgment'],
                metadata['bench'],
                json.dumps(metadata['citations']),
                text,           # <-- clean text
            )
        )

        result = cur.fetchone()
        judgment_id = result[0] if result else None

        # 5. Chunk the clean text
        chunks = chunk_judgment_text(text)

        valid_sections = {'facts', 'issues', 'arguments', 'ratio', 'judgment'}

        normalized_chunks = [
            (section if section in valid_sections else 'judgment', chunk_text)
            for section, chunk_text in chunks
        ]

        # 5b. Insert all of this judgment's chunks in one multi-row INSERT
        # instead of one round-trip per chunk. execute_values with fetch=True
        # returns the RETURNING rows in the same order the input rows were
        # given, so chunk_ids lines up positionally with normalized_chunks.
        chunk_rows = execute_values(
            cur,
            """
            INSERT INTO judgment_chunks (judgment_id, section, content)
            VALUES %s
            RETURNING chunk_id
            """,
            [(judgment_id, section, chunk_text) for section, chunk_text in normalized_chunks],
            fetch=True,
        )
        chunk_ids = [row[0] for row in chunk_rows]
        chunk_texts = [chunk_text for _, chunk_text in normalized_chunks]

        # 6. Generate embeddings for every chunk in this judgment in one
        # batched call instead of one .encode() per chunk.
        embeddings = get_embeddings_batch(chunk_texts)

        embedding_rows = [
            (chunk_id, embedding)
            for chunk_id, embedding in zip(chunk_ids, embeddings)
            if embedding is not None
        ]
        missing = len(chunk_ids) - len(embedding_rows)
        if missing:
            print(f"Warning: {missing} chunk(s) had no embedding generated (BGE model unavailable)")

        if embedding_rows:
            execute_values(
                cur,
                """
                INSERT INTO judgment_embeddings (chunk_id, embedding)
                VALUES %s
                """,
                embedding_rows,
            )

        conn.commit()
        print(f"Successfully ingested {pdf_path} with {len(chunks)} chunks")
        return judgment_id

    except Exception as e:
        conn.rollback()
        print(f"Error processing {pdf_path}: {e}")
        return None
    finally:
        cur.close()


def main():
    """Process all PDFs in the fixtures/sample_judgments/ directory."""
    test_dir = Path("fixtures/sample_judgments")
    if not test_dir.exists():
        print("Test directory not found!")
        return

    pdf_files = list(test_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in test directory!")
        return

    print(f"Found {len(pdf_files)} PDF files to process...")

    conn = psycopg2.connect("dbname=legal_rag")
    try:
        successful = 0
        for pdf_file in pdf_files:
            judgment_id = ingest_judgment_from_pdf(str(pdf_file), conn)
            if judgment_id:
                successful += 1

        print(
            f"\nProcessing complete. "
            f"Successfully ingested {successful}/{len(pdf_files)} judgments."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
