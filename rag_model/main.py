#!/usr/bin/env python3

import sys
import argparse
from rag_pipeline import LegalRAGPipeline
from document_processor import DocumentProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Legal RAG System - Query legal documents using AI"
    )
    parser.add_argument("--query", "-q", type=str, help="Legal question to answer")
    parser.add_argument(
        "--document", "-d", type=str, help="Path to PDF/image document for OCR analysis"
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Disable database retrieval when using document mode",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode (single turn)",
    )
    parser.add_argument(
        "--chat", "-c", action="store_true", help="Chat mode (multi-turn with memory)"
    )
    parser.add_argument(
        "--test", "-t", action="store_true", help="Run built-in test queries"
    )
    parser.add_argument(
        "--retrieval-test", "-r", action="store_true", help="Test retrieval only"
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear chat history before starting",
    )
    parser.add_argument(
        "--top-k", type=int, default=8, help="Number of chunks to retrieve (default: 8)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Similarity threshold (default: 0.3)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Include debug information"
    )

    args = parser.parse_args()

    if not any(
        [
            args.query,
            args.document,
            args.interactive,
            args.chat,
            args.test,
            args.retrieval_test,
        ]
    ):
        parser.print_help()
        return

    print("🔍 Initializing Legal RAG System...")
    try:
        pipeline = LegalRAGPipeline(
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            load_llm=not args.retrieval_test,
        )

        if args.clear_history:
            pipeline.clear_chat_history()
            print("✓ Chat history cleared")

        print("✓ System ready")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    if args.test:
        run_test_queries(pipeline, args.debug)
    elif args.retrieval_test:
        run_retrieval_test(pipeline)
    elif args.document:
        run_document_query(
            pipeline, args.document, args.query, not args.no_retrieval, args.debug
        )
    elif args.chat:
        run_chat_mode(pipeline, args.debug)
    elif args.interactive:
        run_interactive_mode(pipeline, args.debug)
    elif args.query:
        run_single_query(pipeline, args.query, args.debug)


def run_single_query(pipeline, query, debug=False):
    print(f"\n📝 Query: {query}")
    print("-" * 50)

    result = pipeline.query(query, include_debug_info=debug)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"📊 Answer found: {'Yes' if result['answer_found'] else 'No'}")
    print(f"📈 Confidence: {result['confidence']:.2f}")
    print(
        f"⏱️  Time: {result['metrics']['total_time']}s (retrieval: {result['metrics']['retrieval_time']}s, generation: {result['metrics']['generation_time']}s)"
    )

    print(f"\n💬 Response:")
    print(result["answer"])

    if result["citations"]:
        print(f"\n📚 Citations:")
        for citation in result["citations"]:
            print(f"   • {citation}")

    if debug:
        print(f"\n🐛 Debug Info:")
        debug_info = result.get("debug", {})
        if "quality_metrics" in debug_info:
            metrics = debug_info["quality_metrics"]
            print(f"   • Answer quality: {metrics}")
        if "retrieved_chunks" in debug_info:
            print(
                f"   • Top retrieved chunk: {debug_info['retrieved_chunks'][0]['case'] if debug_info['retrieved_chunks'] else 'None'}"
            )


def run_test_queries(pipeline, debug=False):
    test_queries = [
        "What are the regulations for educational institutions in Karnataka?",
        "How does the Supreme Court interpret fundamental rights?",
        "What are the principles of natural justice?",
        "Explain the concept of due process in administrative law",
        "What are the requirements for valid contracts under Indian law?",
    ]

    print(f"\n🧪 Running {len(test_queries)} test queries...")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/{len(test_queries)} ---")
        run_single_query(pipeline, query, debug)
        print("\n" + "-" * 60)


def run_retrieval_test(pipeline):
    test_queries = [
        "education fees Karnataka",
        "Supreme Court fundamental rights",
        "natural justice principles",
        "contract law requirements",
    ]

    print(f"\n🔍 Testing retrieval with {len(test_queries)} queries...")
    print("=" * 60)

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = pipeline.test_retrieval_only(query)

        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue

        print(f"📊 Chunks found: {result['chunks_found']}")
        print(f"⏱️  Retrieval time: {result['retrieval_time']}s")

        if result["retrieved_chunks"]:
            print(
                f"📈 Top similarity: {result['retrieved_chunks'][0]['similarity']:.4f}"
            )
            print(f"📚 Top source: {result['retrieved_chunks'][0]['case']}")
            print(f"💬 Preview: {result['retrieved_chunks'][0]['text'][:100]}...")


def run_document_query(
    pipeline, document_path, query=None, include_retrieval=True, debug=False
):
    print(f"\n📄 Document: {document_path}")
    print("-" * 50)

    doc_processor = DocumentProcessor()
    status = doc_processor.get_status()

    if not status["ready"]:
        print("⚠️  OCR dependencies not installed.")
        print(
            f"   pytesseract: {'✓' if status['pytesseract'] else '✗ (run: pip install pytesseract)'}"
        )
        print(
            f"   pdf2image: {'✓' if status['pdf2image'] else '✗ (run: pip install pdf2image)'}"
        )
        print("\nInstalling dependencies:")
        print("  pip install pytesseract pdf2image")
        print("  # Also install tesseract OCR engine:")
        print("  # macOS: brew install tesseract")
        print("  # Ubuntu: sudo apt-get install tesseract-ocr")
        return

    if query:
        print(f"📝 Query: {query}")

    result = pipeline.query_with_document(
        document_path=document_path,
        user_query=query,
        include_retrieval=include_retrieval,
        include_debug_info=debug,
    )

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    doc_info = result.get("document", {})
    print(f"📋 Document type: {doc_info.get('type', 'unknown').upper()}")
    print(f"📄 Pages: {doc_info.get('pages', 'N/A')}")

    metrics = result.get("metrics", {})
    print(f"⏱️  Time: {metrics.get('total_time', 0)}s", end="")
    if metrics.get("retrieval_time"):
        print(
            f" (retrieval: {metrics['retrieval_time']}s, generation: {metrics['generation_time']}s)"
        )
    else:
        print(f" (generation: {metrics.get('generation_time', 0)}s)")

    print(f"\n💬 Response:")
    print(result["answer"])

    if result.get("citations"):
        print(f"\n📚 Citations:")
        for citation in result["citations"]:
            print(f"   • {citation}")

    if debug:
        print(f"\n🐛 Debug Info:")
        debug_info = result.get("debug", {})
        if "ocr_text_preview" in debug_info:
            print(f"   • OCR preview: {debug_info['ocr_text_preview'][:100]}...")
        if "retrieved_chunks" in debug_info and debug_info["retrieved_chunks"]:
            print(f"   • Top retrieved: {debug_info['retrieved_chunks'][0]['case']}")


def run_chat_mode(pipeline, debug=False):
    print("\n💬 Chat mode - Multi-turn conversation with memory")
    print("   Commands: 'quit' to exit, 'clear' to clear history, 'history' to view")
    print("=" * 60)

    while True:
        try:
            query = input("\n📝 You: ").strip()

            if not query:
                continue
            elif query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            elif query.lower() == "clear":
                pipeline.clear_chat_history()
                print("✓ Chat history cleared")
                continue
            elif query.lower() == "history":
                history = pipeline.get_chat_history()
                print("\n📜 Conversation History:")
                for msg in history:
                    role = "You" if msg["role"] == "user" else "Assistant"
                    print(f"  {role}: {msg['content'][:100]}...")
                continue

            result = pipeline.chat(query, include_debug_info=debug)

            if "error" in result:
                print(f"❌ Error: {result['error']}")
                continue

            print(f"\n🤖 Assistant:")
            print("-" * 40)
            print(result["answer"])

            if result.get("citations"):
                print(f"\n📚 Citations:")
                for citation in result["citations"]:
                    print(f"   • {citation}")

            print(
                f"\n📊 Confidence: {result['confidence']:.2f} | Time: {result['metrics']['total_time']}s"
            )

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def run_interactive_mode(pipeline, debug=False):
    print("\n🔄 Interactive mode - Enter 'quit' to exit")
    print("=" * 60)

    while True:
        try:
            query = input("\n📝 Enter your legal question: ").strip()

            if not query:
                continue
            elif query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            run_single_query(pipeline, query, debug)
            print("\n" + "-" * 60)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
