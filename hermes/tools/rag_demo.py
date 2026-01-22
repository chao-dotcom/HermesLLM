"""
RAG Demonstration Tool

Interactive demonstration of the RAG (Retrieval-Augmented Generation) system.
"""

from typing import Optional

import click
from loguru import logger

from hermes.rag.pipeline import RAGPipeline
from hermes.rag.retriever import VectorRetriever
from hermes.rag.query_expander import QueryExpander


@click.group()
def cli():
    """
    RAG Demonstration Tool
    
    Interactive tools for testing and demonstrating RAG capabilities.
    """
    pass


@cli.command(name="query")
@click.option("--query", "-q", required=True, help="Query text")
@click.option("--k", default=5, type=int, help="Number of documents to retrieve")
@click.option("--collection", default="documents", help="Vector DB collection")
@click.option("--expand", is_flag=True, help="Use query expansion")
@click.option("--rerank", is_flag=True, help="Use reranking")
@click.option("--max-tokens", default=512, type=int, help="Max tokens for generation")
@click.option("--temperature", default=0.7, type=float, help="Temperature")
@click.option("--verbose", is_flag=True, help="Verbose output")
def query_rag(
    query: str,
    k: int,
    collection: str,
    expand: bool,
    rerank: bool,
    max_tokens: int,
    temperature: float,
    verbose: bool,
):
    """
    Query the RAG system.
    
    Examples:
    
        \b
        # Basic query
        hermes rag query -q "What is machine learning?"
        
        \b
        # Query with expansion and reranking
        hermes rag query -q "Explain RAG" --expand --rerank -k 10
        
        \b
        # Verbose output
        hermes rag query -q "Deep learning basics" --verbose
    """
    logger.info(f"Query: {query}")
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline(
            collection_name=collection,
            use_reranking=rerank,
        )
        
        # Optionally expand query
        if expand:
            logger.info("Expanding query...")
            expander = QueryExpander()
            expanded_queries = expander.expand_multi_query(query)
            if verbose:
                logger.info(f"Expanded queries: {expanded_queries}")
        
        # Generate answer
        logger.info("Generating answer...")
        result = rag.generate_answer(
            query=query,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Display results
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result["answer"])
        print("\n" + "="*80)
        
        if verbose:
            print("\nRETRIEVED DOCUMENTS:")
            print("="*80)
            for i, doc in enumerate(result.get("retrieved_documents", []), 1):
                print(f"\n[{i}] Score: {getattr(doc, 'score', 'N/A')}")
                print(f"Content: {doc.content[:200]}...")
                if hasattr(doc, "metadata"):
                    print(f"Metadata: {doc.metadata}")
        
        logger.success("Query completed")
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise


@cli.command(name="retrieve")
@click.option("--query", "-q", required=True, help="Query text")
@click.option("--k", default=5, type=int, help="Number of documents to retrieve")
@click.option("--collection", default="documents", help="Vector DB collection")
@click.option("--expand", is_flag=True, help="Use multi-query expansion")
@click.option("--self-query", is_flag=True, help="Use self-query extraction")
def retrieve_only(
    query: str,
    k: int,
    collection: str,
    expand: bool,
    self_query: bool,
):
    """
    Retrieve documents without generation (retrieval only).
    
    Examples:
    
        \b
        # Basic retrieval
        hermes rag retrieve -q "machine learning"
        
        \b
        # With query expansion
        hermes rag retrieve -q "deep learning" --expand -k 10
    """
    logger.info(f"Retrieving documents for: {query}")
    
    try:
        # Initialize retriever
        retriever = VectorRetriever(collection_name=collection)
        
        # Optionally expand query
        queries = [query]
        if expand:
            logger.info("Expanding query...")
            expander = QueryExpander()
            queries = expander.expand_multi_query(query)
            logger.info(f"Generated {len(queries)} queries")
        
        # Retrieve documents
        logger.info("Retrieving documents...")
        if len(queries) > 1:
            documents = retriever.retrieve_multi_query(queries, k=k)
        else:
            documents = retriever.retrieve(query, k=k)
        
        # Display results
        print("\n" + "="*80)
        print(f"RETRIEVED {len(documents)} DOCUMENTS:")
        print("="*80)
        
        for i, doc in enumerate(documents, 1):
            print(f"\n[{i}] Score: {getattr(doc, 'score', 'N/A')}")
            print(f"Content: {doc.content[:300]}...")
            if hasattr(doc, "metadata"):
                print(f"Metadata: {doc.metadata}")
            print("-"*80)
        
        logger.success(f"Retrieved {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise


@cli.command(name="interactive")
@click.option("--collection", default="documents", help="Vector DB collection")
@click.option("--k", default=5, type=int, help="Number of documents to retrieve")
@click.option("--rerank", is_flag=True, help="Use reranking")
def interactive_mode(collection: str, k: int, rerank: bool):
    """
    Interactive RAG mode - continuous question answering.
    
    Examples:
    
        \b
        # Start interactive session
        hermes rag interactive
        
        \b
        # With reranking
        hermes rag interactive --rerank -k 10
    """
    logger.info("Starting interactive RAG mode")
    logger.info("Type 'exit' or 'quit' to stop")
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline(
            collection_name=collection,
            use_reranking=rerank,
        )
        
        print("\n" + "="*80)
        print("INTERACTIVE RAG MODE")
        print("="*80)
        print("Ask me anything! (Type 'exit' to quit)\n")
        
        while True:
            # Get user input
            try:
                query = input("\n🤔 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break
            
            if not query:
                continue
            
            if query.lower() in ["exit", "quit", "q"]:
                break
            
            # Generate answer
            try:
                result = rag.generate_answer(query=query, k=k)
                print(f"\n🤖 HermesLLM: {result['answer']}\n")
                print("-"*80)
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\n❌ Error: {e}\n")
        
        logger.info("Exiting interactive mode")
        
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")
        raise


@cli.command(name="demo")
@click.option("--author", default="Paul Iusztin", help="Author name for demo queries")
def run_demo(author: str):
    """
    Run a pre-configured RAG demonstration.
    
    Examples:
    
        \b
        # Run demo with default author
        hermes rag demo
        
        \b
        # Run demo with specific author
        hermes rag demo --author "John Doe"
    """
    logger.info("Running RAG demonstration")
    
    # Demo queries
    demo_queries = [
        f"My name is {author}. Could you draft a LinkedIn post discussing RAG systems?",
        f"What has {author} written about machine learning?",
        "Explain how RAG systems work with vector databases and LLMs",
    ]
    
    try:
        rag = RAGPipeline(use_reranking=True)
        
        print("\n" + "="*80)
        print("RAG SYSTEM DEMONSTRATION")
        print("="*80)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n\n[Query {i}]: {query}")
            print("-"*80)
            
            result = rag.generate_answer(query=query, k=5)
            
            print("\nAnswer:")
            print(result["answer"])
            print("\n" + "="*80)
        
        logger.success("Demo completed")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    cli()
