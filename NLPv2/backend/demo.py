#!/usr/bin/env python3

from retriever import LegalRetriever
from prompt_builder import PromptBuilder
import json

def demo_rag_without_llm():
    print("🔍 Legal RAG Demo (Retrieval + Prompt Building)")
    print("=" * 60)
    
    retriever = LegalRetriever()
    prompt_builder = PromptBuilder()
    
    demo_queries = [
        "education Bill of Karnataka",
        "Supreme Court natural justice principles", 
        "contract law interpretation"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        print("🔍 Retrieving relevant chunks...")
        chunks = retriever.retrieve_relevant_chunks(query, top_k=3)
        
        if not chunks:
            print("❌ No relevant chunks found")
            continue
        
        print(f"✓ Found {len(chunks)} relevant chunks")
        
        print("\n📋 Top Retrieved Chunks:")
        for j, chunk in enumerate(chunks, 1):
            print(f"\n[{j}] {chunk['case']} ({chunk['court']}, {chunk['year']})")
            print(f"    Similarity: {chunk['similarity']:.4f}")
            print(f"    Section: {chunk['section']}")
            print(f"    Preview: {chunk['text'][:150]}...")
        
        print("\n📝 Generated RAG Prompt:")
        prompt = prompt_builder.build_rag_prompt(chunks, query)
        print("-" * 40)
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    demo_rag_without_llm()
