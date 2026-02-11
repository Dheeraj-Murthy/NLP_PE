import os
import psycopg2
import json
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer

class LegalRetriever:
    def __init__(self, db_connection_string: str = "dbname=legal_rag"):
        self.db_connection_string = db_connection_string
        self.embedding_model = None
        self._load_embedding_model()
    
    
    def _load_embedding_model(self):
        try:
            self.embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
            print("✓ BGE-base model loaded for retrieval")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def get_query_embedding(self, query: str) -> List[float]:
        if not self.embedding_model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {e}")
    
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        top_k: int = 8,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        query_embedding = self.get_query_embedding(query)
        
        conn = psycopg2.connect(self.db_connection_string)
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT 
                    jc.content,
                    j.petitioner,
                    j.respondent,
                    j.court,
                    EXTRACT(YEAR FROM j.date_of_judgment) as year,
                    jc.section,
                    jc.chunk_id,
                    1 - (je.embedding <=> %s::vector) as similarity
                FROM judgment_embeddings je
                JOIN judgment_chunks jc ON jc.chunk_id = je.chunk_id
                JOIN judgments j ON j.id = jc.judgment_id
                WHERE 1 - (je.embedding <=> %s::vector) >= %s
                ORDER BY je.embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, similarity_threshold, query_embedding, top_k))
            
            results = cur.fetchall()
            
            retrieved_chunks = []
            for row in results:
                content, petitioner, respondent, court, year, section, chunk_id, similarity = row
                
                case_name = self._format_case_name(petitioner, respondent)
                para_id = f"¶{chunk_id}"
                
                retrieved_chunks.append({
                    "text": content.strip(),
                    "case": case_name,
                    "court": court if court else "Unknown Court",
                    "year": int(year) if year else None,
                    "para": para_id,
                    "section": section,
                    "similarity": float(similarity)
                })
            
            return retrieved_chunks
            
        finally:
            cur.close()
            conn.close()
    

        def retrieve_candidate_chunks(
            self,
            query: str,
            candidate_k: int = 30,
            similarity_threshold: float = 0.2
        ) -> List[Dict[str, Any]]:
        
        #Stage-1 retrieval: fetch more candidates with lower threshold for reranking stage.
        
            query_embedding = self.get_query_embedding(query)
        
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()
        
            try:
                cur.execute("""
                    SELECT 
                        jc.content,
                        j.petitioner,
                        j.respondent,
                        j.court,
                        EXTRACT(YEAR FROM j.date_of_judgment) as year,
                        jc.section,
                        jc.chunk_id,
                        1 - (je.embedding <=> %s::vector) as similarity
                    FROM judgment_embeddings je
                    JOIN judgment_chunks jc ON jc.chunk_id = je.chunk_id
                    JOIN judgments j ON j.id = jc.judgment_id
                    WHERE 1 - (je.embedding <=> %s::vector) >= %s
                    ORDER BY je.embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, similarity_threshold, query_embedding, candidate_k))
            
                results = cur.fetchall()
            
                retrieved_chunks = []
                for row in results:
                    content, petitioner, respondent, court, year, section, chunk_id, similarity = row
                    
                    case_name = self._format_case_name(petitioner, respondent)
                    para_id = f"¶{chunk_id}"
                    
                    retrieved_chunks.append({
                        "text": content.strip(),
                        "case": case_name,
                        "court": court if court else "Unknown Court",
                        "year": int(year) if year else None,
                        "para": para_id,
                        "section": section,
                        "similarity": float(similarity)
                    })
            
                return retrieved_chunks
            
            finally:
                cur.close()
                conn.close()

    def _format_case_name(self, petitioner: Optional[str], respondent: Optional[str]) -> str:
        if not petitioner or petitioner == 'Unknown':
            petitioner = 'Unknown'
        if not respondent or respondent == 'Unknown':
            respondent = 'Unknown'
        
        petitioner = petitioner.strip().title()
        respondent = respondent.strip().title()
        
        max_length = 50
        if len(petitioner) > max_length:
            petitioner = petitioner[:max_length] + "..."
        if len(respondent) > max_length:
            respondent = respondent[:max_length] + "..."
        
        return f"{petitioner} v. {respondent}"
    
    def get_retrieval_stats(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        query_embedding = self.get_query_embedding(query)
        
        conn = psycopg2.connect(self.db_connection_string)
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    AVG(1 - (embedding <=> %s::vector)) as avg_similarity,
                    MAX(1 - (embedding <=> %s::vector)) as max_similarity,
                    MIN(1 - (embedding <=> %s::vector)) as min_similarity
                FROM judgment_embeddings
            """, (query_embedding, query_embedding, query_embedding))
            
            stats = cur.fetchone()
            
            cur.execute("""
                SELECT 
                    jc.section,
                    COUNT(*) as count,
                    AVG(1 - (je.embedding <=> %s::vector)) as avg_similarity
                FROM judgment_embeddings je
                JOIN judgment_chunks jc ON jc.chunk_id = je.chunk_id
                GROUP BY jc.section
                ORDER BY avg_similarity DESC
            """, (query_embedding,))
            
            section_stats = cur.fetchall()
            
            return {
                "total_chunks": stats[0] if stats else 0,
                "avg_similarity": float(stats[1]) if stats and stats[1] else 0,
                "max_similarity": float(stats[2]) if stats and stats[2] else 0,
                "min_similarity": float(stats[3]) if stats and stats[3] else 0,
                "section_distribution": [
                    {
                        "section": row[0],
                        "count": row[1],
                        "avg_similarity": float(row[2]) if row[2] else 0
                    }
                    for row in section_stats
                ]
            }
            
        finally:
            cur.close()
            conn.close()

if __name__ == "__main__":
    retriever = LegalRetriever()
    query = "education Bill of Karnataka"
    
    print(f"Query: {query}")
    print("=" * 50)
    
    results = retriever.retrieve_relevant_chunks(query, top_k=3)
    
    for i, chunk in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Case: {chunk['case']}")
        print(f"Court: {chunk['court']} ({chunk['year']})")
        print(f"Section: {chunk['section']}")
        print(f"Similarity: {chunk['similarity']:.4f}")
        print(f"Content: {chunk['text'][:200]}...")
    
    print("\n" + "=" * 50)
    print("Retrieval Statistics:")
    stats = retriever.get_retrieval_stats(query)
    print(f"Total chunks in DB: {stats['total_chunks']}")
    print(f"Average similarity: {stats['avg_similarity']:.4f}")
    print(f"Max similarity: {stats['max_similarity']:.4f}")
    print("\nSection distribution:")
    for section in stats['section_distribution']:
        print(f"  {section['section']}: {section['count']} chunks (avg sim: {section['avg_similarity']:.4f})")