import psycopg2
import os
from sentence_transformers import SentenceTransformer

# Initialize BGE-base-en-v1.5 model
try:
    embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    MODEL_AVAILABLE = True
    print("Local BGE-base model loaded successfully")
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"Warning: Failed to load BGE model. Error: {e}")
    exit(1)

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

def semantic_search(query, k=5):
    query_embedding = get_embedding(query)
    
    if query_embedding is None:
        print("Failed to generate query embedding")
        return []

    conn = psycopg2.connect("dbname=legal_rag")
    cur = conn.cursor()

    cur.execute("""
        SELECT
            j.id AS judgment_id,
            jc.section,
            LEFT(jc.content, 400) AS preview,
            1 - (je.embedding <=> %s::vector) AS similarity
        FROM judgment_embeddings je
        JOIN judgment_chunks jc ON jc.chunk_id = je.chunk_id
        JOIN judgments j ON j.id = jc.judgment_id
        ORDER BY je.embedding <=> %s::vector
        LIMIT %s;
    """, (query_embedding, query_embedding, k))

    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

if __name__ == "__main__":
    results = semantic_search("education Bill of karnataka", k=5)
    
    for r in results:
        print(f"Section: {r[1]}")
        print(f"Similarity: {r[3]:.4f}")
        print(f"Preview: {r[2]}")
        print("-" * 40)