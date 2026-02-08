import arxiv
import os

# Query -> (Title Keywords, Author Keywords)
queries = [
    ("Toy Models of Superposition", ["Superposition"], ["Elhage", "Hume", "Olsson"]),
    ("The Geometry of Truth", ["Geometry", "Truth"], ["Marks", "Tegmark"]),
    ("Language Models Represent Space and Time", ["Space", "Time"], ["Gurnee", "Tegmark"]),
    ("Linearity of Relation Decoding", ["Linearity", "Relation"], []),
    ("Polysemanticity", ["Polysemanticity"], [])
]

save_dir = "papers"
client = arxiv.Client()

for query_text, title_kw, author_kw in queries:
    print(f"Searching for: {query_text}")
    search = arxiv.Search(
        query = query_text,
        max_results = 5,
        sort_by = arxiv.SortCriterion.Relevance
    )
    
    found = False
    for result in client.results(search):
        # Check title match
        title_match = all(kw.lower() in result.title.lower() for kw in title_kw)
        
        # Check author match (if provided)
        author_match = True
        if author_kw:
             author_match = any(any(akw.lower() in a.name.lower() for akw in author_kw) for a in result.authors)
        
        if title_match and author_match:
            print(f"MATCH FOUND: {result.title}")
            print(f"PDF URL: {result.pdf_url}")
            
            filename = f"{save_dir}/{result.entry_id.split('/')[-1]}_{query_text.replace(' ', '_')[:50]}.pdf"
            if not os.path.exists(filename):
                print(f"Downloading to {filename}...")
                result.download_pdf(dirpath=save_dir, filename=filename.split('/')[-1])
            else:
                print(f"File {filename} already exists.")
            found = True
            break
        else:
            print(f"Skipping: {result.title} (No match)")
            
    if not found:
        print(f"Could not find a good match for: {query_text}")
