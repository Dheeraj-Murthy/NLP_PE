from rag_pipeline import LegalRAGPipeline
from reranker import CrossEncoderReranker
from prompt_builder import PromptBuilder

p = LegalRAGPipeline(load_llm=True, max_context_length=4000)
chunks = p.retriever.retrieve_hybrid_candidates(query='What are the grounds for grant of bail?', candidate_k=30, similarity_threshold=0.2)
top = CrossEncoderReranker().rerank('What are the grounds for grant of bail?', chunks, top_n=8)
prompt = PromptBuilder(4000).build_rag_prompt(top, 'What are the grounds for grant of bail?')
print("=== PROMPT (first 1500 chars) ===")
print(prompt[:1500])
print("=== END PROMPT ===")
print("Length of prompt:", len(prompt))
resp = p.llm.generate_response(prompt, max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True)
print("=== RAW RESPONSE ===")
print(repr(resp))
