"""
Example usage of RelationshipExtractor.
Demonstrates various ways to use the relationship extraction system.
"""

import os
from dotenv import load_dotenv
from relationship_extractor import RelationshipExtractor

# Load environment variables
load_dotenv()

# Example 1: Simple extraction
def example_simple():
    """Extract relationships from a single short text."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Extraction")
    print("="*70)
    
    text = """
    Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976.
    Steve Jobs served as CEO for many years. He is no longer alive today.
    The company is headquartered in Cupertino, California.
    Apple did NOT start in a traditional office building.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return
    
    extractor = RelationshipExtractor(api_key=api_key)
    
    print("\nInput Text:")
    print(text)
    
    print("\nExtracting relationships (simple mode)...")
    df = extractor.extract(text, include_implicit=False)
    
    print(f"\nExtracted {len(df)} relationships:\n")
    print(df.to_string(index=False))
    
    # Save results
    df.to_csv("/tmp/example1_simple.csv", index=False)
    print(f"\nResults saved to /tmp/example1_simple.csv")


# Example 2: Multi-pass extraction with implicit relationships
def example_multipass():
    """Extract with multi-pass approach including implicit relationships."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Pass Extraction (with implicit)")
    print("="*70)
    
    text = """
    John Smith is a software engineer at Google. Google is headquartered 
    in Mountain View. John works on the Cloud Platform team. The team 
    has 50 engineers. Mountain View is located in California.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return
    
    extractor = RelationshipExtractor(api_key=api_key)
    
    print("\nInput Text:")
    print(text)
    
    print("\nExtracting with multi-pass approach...")
    df = extractor.extract(text, include_implicit=True)
    
    print(f"\nExtracted {len(df)} relationships:\n")
    
    # Display with more details
    for idx, row in df.iterrows():
        print(f"{idx+1}. {row['subject']:20} -> {row['predicate']:20} -> {row['object']:20}")
        print(f"   Source: {row['source_quote'][:60]}...")
        print(f"   Negated: {row['negated']}\n")


# Example 3: Cross-chunk relationships (long text)
def example_cross_chunk():
    """Demonstrate cross-chunk relationship extraction."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Cross-Chunk Relationship Detection")
    print("="*70)
    
    text = """
    The company was founded in 1998. Its founders were very ambitious.
    
    The headquarters is located in a small town. The town is known for tech.
    
    Years later, the company expanded globally. It now has offices in 50 countries.
    
    The founder is still involved in strategic decisions. He believes in innovation.
    The innovative culture has made the company successful.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return
    
    extractor = RelationshipExtractor(api_key=api_key)
    
    print("\nLong Text (simulating multiple chunks):")
    print(text[:200] + "...\n")
    
    print("Extracting relationships across chunks...")
    df = extractor.extract(text, include_implicit=True)
    
    print(f"\nExtracted {len(df)} relationships (local + cross-chunk):\n")
    
    for idx, row in df.iterrows():
        print(f"{idx+1}. {row['subject']} -> {row['predicate']} -> {row['object']}")


# Example 4: Pronoun resolution example
def example_pronoun_resolution():
    """Show pronoun resolution in action."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Pronoun Resolution")
    print("="*70)
    
    text = """
    Steve Jobs founded Apple in 1976. He served as CEO for many years.
    He led the company through many innovations. His vision transformed
    the technology industry. He believed in design excellence.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return
    
    extractor = RelationshipExtractor(api_key=api_key)
    
    print("\nOriginal Text (with pronouns):")
    print(text)
    
    print("\nExtracting relationships (pronouns will be resolved automatically)...")
    df = extractor.extract(text, include_implicit=False)
    
    print(f"\nExtracted {len(df)} relationships:\n")
    
    # Show how pronouns were resolved
    subjects = df['subject'].unique()
    print(f"Unique subjects (pronoun resolution applied): {list(subjects)}\n")
    
    for idx, row in df.iterrows():
        print(f"-> {row['subject']:20} {row['predicate']:20} {row['object']:20}")


# Example 5: Negation detection
def example_negation_detection():
    """Demonstrate negation flagging."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Negation Detection")
    print("="*70)
    
    text = """
    Steve Jobs founded Apple. He did NOT work at Microsoft.
    Apple is headquartered in California. Apple is NOT located in Texas.
    The company makes iPhones. It does NOT make only phones.
    """
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return
    
    from relationship_extractor import RelationshipExtractor as RE
    
    extractor = RE(api_key=api_key)
    
    print("\nText with negations:")
    print(text)
    
    print("\nExtracting and analyzing negations...")
    
    # For this, we need to modify extraction to show negations
    chunks = extractor.chunk_text(text)
    all_rels = []
    
    for chunk in chunks:
        rels = extractor._extract_from_chunk(chunk, pass_number=1)
        all_rels.extend(rels)
    
    print(f"\nFound {len(all_rels)} relationships:\n")
    
    # Show positive vs negated
    positive = [r for r in all_rels if r.get('negated') != True]
    negated = [r for r in all_rels if r.get('negated') == True]
    
    print(f"Positive Relationships ({len(positive)}):")
    for rel in positive:
        print(f"  [+] {rel['subject']} -> {rel['predicate']} -> {rel['object']}")
    
    print(f"\nNegated Relationships ({len(negated)}):")
    for rel in negated:
        print(f"  [-] {rel['subject']} -> {rel['predicate']} -> {rel['object']}")


# Example 6: Batch processing multiple texts
def example_batch_processing():
    """Process multiple texts at once."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Batch Processing")
    print("="*70)
    
    texts = [
        "Google was founded by Larry Page and Sergey Brin in 1998.",
        "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
    ]
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return
    
    extractor = RelationshipExtractor(api_key=api_key)
    
    print(f"\nProcessing {len(texts)} texts...")
    results = extractor.extract_batch(texts, include_implicit=False)
    
    print(f"\nResults:\n")
    
    for text_id, df in results.items():
        print(f"\n{text_id}:")
        if not df.empty:
            for idx, row in df.iterrows():
                print(f"  -> {row['subject']} -> {row['predicate']} -> {row['object']}")
        else:
            print("  (No relationships found)")


# Main execution
if __name__ == "__main__":
    print("\n")
    print("="*70)
    print("RELATIONSHIP EXTRACTOR - EXAMPLES")
    print("="*70)
    
    try:
        # Run examples
        example_simple()
        example_pronoun_resolution()
        example_negation_detection()
        example_multipass()
        example_cross_chunk()
        example_batch_processing()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure OPENAI_API_KEY is set in your .env file.")
