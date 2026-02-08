import arxiv
import os

queries = [
    "Toy Models of Superposition",
    "Linearity of Relation Decoding in Transformer Language Models",
    "The Geometry of Truth: Mechanism of True/False Determination in LLMs",
    "Language Models Represent Space and Time",
    "Polysemanticity and Capacity in Neural Networks"
]

save_dir = "papers"
client = arxiv.Client()

for query in queries:
    print(f"Searching for: {query}")
    search = arxiv.Search(
        query = query,
        max_results = 1,
        sort_by = arxiv.SortCriterion.Relevance
    )
    
    try:
        result = next(client.results(search))
        print(f"Found: {result.title}")
        print(f"PDF URL: {result.pdf_url}")
        
        filename = f"{save_dir}/{result.entry_id.split('/')[-1]}_{query.replace(' ', '_')[:50]}.pdf"
        if not os.path.exists(filename):
            print(f"Downloading to {filename}...")
            result.download_pdf(dirpath=save_dir, filename=filename.split('/')[-1])
        else:
            print(f"File {filename} already exists.")
            
    except StopIteration:
        print(f"Paper not found on arXiv: {query}")
    except Exception as e:
        print(f"Error downloading {query}: {e}")
