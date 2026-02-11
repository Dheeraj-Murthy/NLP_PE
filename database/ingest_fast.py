import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import PyPDF2
import psycopg2
from datetime import datetime
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import time
from psycopg2 import pool
import numpy as np

# Initialize connection pool
MAX_WORKERS = min(multiprocessing.cpu_count(), 12)  # Limit to 12 workers to avoid overwhelming
print(f"🚀 Using {MAX_WORKERS} workers for parallel processing")

try:
    connection_pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=4,
        maxconn=MAX_WORKERS + 2,
        dbname="legal_rag"
    )
    print("✓ Database connection pool created")
except Exception as e:
    print(f"❌ Failed to create connection pool: {e}")
    exit(1)

# Initialize local embedding model
try:
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    MODEL_AVAILABLE = True
    print("✓ Local BGE-base model loaded successfully")
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"❌ Warning: Failed to load BGE model. Error: {e}")
    exit(1)

def get_batch_embeddings(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Generate embeddings for multiple texts at once (much faster)"""
    if not MODEL_AVAILABLE or not texts:
        return [None] * len(texts)
    
    try:
        # Process in batches to manage memory
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embeddings = embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                batch_size=batch_size,
                normalize_embeddings=True
            )
            all_embeddings.extend([emb.tolist() for emb in embeddings])
        
        return all_embeddings
    except Exception as e:
        print(f"❌ Error generating batch embeddings: {e}")
        return [None] * len(texts)

def extract_text_from_pdf_fast(pdf_path: str) -> str:
    """Fast PDF text extraction using optimized PyPDF2 settings"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file, strict=False)  # Disable strict mode for speed
            
            # Extract text from all pages
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"
                except:
                    continue  # Skip problematic pages
                    
    except Exception as e:
        print(f"❌ Error extracting {pdf_path}: {e}")
        return ""
    
    return text

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

def batch_insert_chunks_and_embeddings(conn, judgment_id, chunks):
    with conn.cursor() as cur:
        try:
            sections = [section for section, _ in chunks]
            contents = [content for _, content in chunks]

            cur.execute("""
                INSERT INTO judgment_chunks (judgment_id, section, content)
                SELECT %s, s, c
                FROM unnest(%s::text[], %s::text[]) AS t(s, c)
                RETURNING chunk_id
                """, (judgment_id, sections, contents))

            chunk_ids = [row[0] for row in cur.fetchall()]

            # embeddings
            embeddings = get_batch_embeddings(contents)
            embedding_data = [
                (cid, emb) for cid, emb in zip(chunk_ids, embeddings) if emb is not None
            ]

            if embedding_data:
                cur.executemany("""
                    INSERT INTO judgment_embeddings (chunk_id, embedding)
                    VALUES (%s, %s)
                    """, embedding_data)

                conn.commit()
                return chunk_ids

        except Exception as e:
            conn.rollback()
            raise

def ingest_judgment_from_pdf_fast(pdf_path: str) -> Tuple[int, bool]:
    """Fast ingestion of a single PDF judgment"""
    start_time = time.time()
    
    try:
        # Extract text
        text = extract_text_from_pdf_fast(pdf_path)
        if not text.strip():
            print(f"⚠️  No text extracted from {pdf_path}")
            return None, False
        
        # Parse metadata
        metadata = parse_judgment_metadata(text, os.path.basename(pdf_path))
        
        # Get connection from pool
        conn = connection_pool.getconn()
        try:
            with conn.cursor() as cur:
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
                
                if judgment_id:
                    # Create and insert chunks in batch
                    chunks = chunk_judgment_text(text)
                    
                    # Validate sections
                    valid_sections = ['facts', 'issues', 'arguments', 'ratio', 'judgment']
                    validated_chunks = []
                    for section, chunk_text in chunks:
                        if section not in valid_sections:
                            section = 'judgment'
                        validated_chunks.append((section, chunk_text))
                    
                    chunk_ids = batch_insert_chunks_and_embeddings(conn, judgment_id, validated_chunks)
                    
                    processing_time = time.time() - start_time
                    print(f"✓ {os.path.basename(pdf_path)}: {len(chunk_ids)} chunks in {processing_time:.1f}s")
                    return judgment_id, bool(chunk_ids)
                else:
                    return None, False
                    
        finally:
            connection_pool.putconn(conn)
            
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return None, False

def main():
    """Main function to process all PDFs in parallel"""
    start_time = time.time()
    
    test_dir = Path("test")
    if not test_dir.exists():
        print("❌ Test directory not found!")
        return
    
    pdf_files = list(test_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in test directory!")
        return
    
    print(f"🚀 Starting fast ingestion of {len(pdf_files)} PDF files using {MAX_WORKERS} workers...")
    print(f"⏱️  Estimated time: {len(pdf_files) / (MAX_WORKERS * 2):.1f} minutes")
    
    successful = 0
    failed = 0
    
    # Process PDFs in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(ingest_judgment_from_pdf_fast, str(pdf_file)): pdf_file 
            for pdf_file in pdf_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                judgment_id, success = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Task failed for {pdf_file}: {e}")
                failed += 1
    
    total_time = time.time() - start_time
    
    print(f"\n" + "="*60)
    print(f"🎉 FAST INGESTION COMPLETE")
    print(f"✅ Successfully processed: {successful}/{len(pdf_files)} judgments")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"🚀 Average: {total_time/len(pdf_files):.2f} seconds per judgment")
    print(f"📊 Rate: {len(pdf_files)/total_time:.2f} judgments per second")
    print(f"🔥 Speedup vs sequential: ~{MAX_WORKERS}x faster")
    print("="*60)

if __name__ == "__main__":
    main()
