import os
import re
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any
import psycopg2
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

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdftotext (poppler) - handles scanned PDFs too"""
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', pdf_path, '-'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"pdftotext failed: {e}")
    return ""

def parse_judgment_metadata(text: str, filename: str) -> Dict[str, Any]:
    """Extract metadata from judgment text"""
    metadata = {
        'petitioner': 'Unknown',
        'respondent': 'Unknown', 
        'court': 'Supreme Court of India',
        'date_of_judgment': None,
        'bench': [],
        'citations': {}
    }
    
    cleaned_text = re.sub(r'http://JUDIS\.NIC\.IN\s+', '', text)
    cleaned_text = re.sub(r'SUPREME COURT OF INDIA\s+Page \d+ of \d+\s+', '', cleaned_text)
    
    header_lines = cleaned_text.split('\n')[:20]
    
    petitioner_name = None
    respondent_name = None
    
    for line_index, current_line in enumerate(header_lines):
        stripped_line = current_line.strip()
        
        if stripped_line == 'PETITIONER:':
            petitioner_name = _extract_name_after_header(header_lines, line_index + 1, ['RESPONDENT:', 'Vs.', 'vs.', 'V.'])
        elif stripped_line == 'RESPONDENT:':
            respondent_name = _extract_name_after_header(header_lines, line_index + 1, ['DATE OF JUDGMENT:', 'BENCH:', 'ACT:'])
    
    if petitioner_name:
        metadata['petitioner'] = petitioner_name
    if respondent_name:
        metadata['respondent'] = respondent_name
    
    date_pattern = r'DATE OF JUDGMENT:\s*(\d{2}/\d{2}/\d{4})'
    date_match = re.search(date_pattern, cleaned_text)
    if date_match:
        date_str = date_match.group(1)
        try:
            date_obj = datetime.strptime(date_str, '%d/%m/%Y')
            metadata['date_of_judgment'] = date_obj.strftime('%Y-%m-%d')
        except:
            pass
    
    bench_pattern = r'BENCH:\s*\[?([^\]\n]+)'
    bench_match = re.search(bench_pattern, cleaned_text)
    if bench_match:
        bench_text = bench_match.group(1).strip()
        judges = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z])?)\s*\.?\s*J\.?', bench_text)
        if judges:
            metadata['bench'] = judges
    
    return metadata

def _extract_name_after_header(lines: List[str], start_index: int, stop_markers: List[str]) -> str | None:
    """Extract name that appears after a header line"""
    for next_line_index in range(start_index, len(lines)):
        line_content = lines[next_line_index].strip()
        if line_content in stop_markers:
            break
        if (line_content and 
            not line_content.startswith('ACT:') and 
            not line_content.startswith('BENCH:') and
            not line_content.startswith('Vs.') and
            not line_content.startswith('vs.') and
            not line_content.startswith('V.')):
            return line_content
    return None

def chunk_judgment_text(text: str) -> List[Tuple[str, str]]:
    """Split judgment text into logical chunks based on common legal judgment structure"""
    
    section_identifiers = {
        'facts': [
            r'(?i)facts of the case',
            r'(?i)facts',
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
            r'(?i)judgment',
            r'(?i)order',
            r'(?i)decision',
            r'(?i)conclusion',
            r'(?i)holding',
            r'(?i)we hold',
            r'(?i)we conclude',
            r'(?i)accordingly',
            r'(?i)in the result'
        ]
    }
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    
    current_section = 'facts'
    current_section_text = []
    
    for paragraph in paragraphs:
        identified_section = _identify_section_type(paragraph, section_identifiers)
        
        if identified_section and identified_section != current_section and current_section_text:
            chunks.append((current_section, '\n'.join(current_section_text)))
            current_section_text = []
            current_section = identified_section
        
        current_section_text.append(paragraph)
    
    if current_section_text:
        chunks.append((current_section, '\n'.join(current_section_text)))
    
    if not chunks:
        chunks = _create_size_based_chunks(text)
    
    return chunks

def _identify_section_type(paragraph: str, section_patterns: Dict[str, List[str]]) -> str | None:
    """Identify which section a paragraph belongs to based on patterns"""
    for section_type, patterns in section_patterns.items():
        for pattern in patterns:
            if re.search(pattern, paragraph):
                return section_type
    return None

def _create_size_based_chunks(text: str) -> List[Tuple[str, str]]:
    """Create chunks of reasonable size when no clear sections are found"""
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

def ingest_judgment_from_pdf(pdf_path: str) -> int | None:
    """Ingest a single PDF judgment into the database"""
    print(f"Processing {pdf_path}...")
    
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"Warning: No text extracted from {pdf_path}")
        return None
    
    # Parse metadata
    metadata = parse_judgment_metadata(text, os.path.basename(pdf_path))
    
    conn = psycopg2.connect("dbname=legal_rag")
    cur = conn.cursor()
    
    try:
        # Insert judgment
        cur.execute("""
        INSERT INTO judgments
        (petitioner, respondent, court, date_of_judgment, bench, citations, judgment_text)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
        """, (
            metadata['petitioner'],
            metadata['respondent'],
            metadata['court'],
            metadata['date_of_judgment'],
            metadata['bench'],
            json.dumps(metadata['citations']),
            text
        ))
        
        result = cur.fetchone()
        judgment_id = result[0] if result else None
        
        # Create and insert chunks
        chunks = chunk_judgment_text(text)
        
        for section, chunk_text in chunks:
            # Validate section
            valid_sections = ['facts', 'issues', 'arguments', 'ratio', 'judgment']
            if section not in valid_sections:
                section = 'judgment'  # Default fallback
            
            cur.execute("""
            INSERT INTO judgment_chunks (judgment_id, section, content)
            VALUES (%s, %s, %s)
            RETURNING chunk_id
            """, (judgment_id, section, chunk_text))
            
            chunk_result = cur.fetchone()
            chunk_id = chunk_result[0] if chunk_result else None
            
            # Generate and store embedding
            embedding = get_embedding(chunk_text)
            
            if embedding is not None:
                cur.execute("""
                INSERT INTO judgment_embeddings (chunk_id, embedding)
                VALUES (%s, %s)
                """, (chunk_id, embedding))
            else:
                print(f"Warning: No embedding generated for chunk {chunk_id} (BGE model unavailable)")
        
        conn.commit()
        print(f"Successfully ingested {pdf_path} with {len(chunks)} chunks")
        return judgment_id
        
    except Exception as e:
        conn.rollback()
        print(f"Error processing {pdf_path}: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def main():
    """Main function to process all PDFs in test directory"""
    test_dir = Path("test")
    if not test_dir.exists():
        print("Test directory not found!")
        return
    
    pdf_files = list(test_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in test directory!")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process...")
    
    successful = 0
    for pdf_file in pdf_files:
        judgment_id = ingest_judgment_from_pdf(str(pdf_file))
        if judgment_id:
            successful += 1
    
    print(f"\nProcessing complete. Successfully ingested {successful}/{len(pdf_files)} judgments.")

if __name__ == "__main__":
    main()
