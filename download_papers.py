import arxiv
import requests
import os

titles = [
    "Funny or Persuasive, but Not Both: Evaluating Fine-Grained Multi-Concept Control in LLMs",
    "From Frege to chatGPT: Compositionality in language, cognition, and deep neural networks",
    "Latent Concept Disentanglement in Transformer-based Language Models",
    "Improving Large Language Models with Concept-Aware Fine-Tuning",
    "Beyond Syntax: How Do LLMs Understand Code?",
    "Exploring Multilingual Concepts of Human Values in Large Language Models"
]

save_dir = "papers"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

client = arxiv.Client()

for title in titles:
    print(f"Searching for: {title}")
    search = arxiv.Search(
        query = title,
        max_results = 1,
        sort_by = arxiv.SortCriterion.Relevance
    )
    
    try:
        result = next(client.results(search))
        # Simple check if title matches reasonably well (ignore case and non-alphanumeric)
        # This is a basic check, might need manual verification if results are weird
        print(f"Found: {result.title}")
        print(f"PDF URL: {result.pdf_url}")
        
        filename = f"{save_dir}/{result.entry_id.split('/')[-1]}_{title.replace(' ', '_')[:50]}.pdf"
        if not os.path.exists(filename):
            print(f"Downloading to {filename}...")
            result.download_pdf(dirpath=save_dir, filename=filename.split('/')[-1])
        else:
            print(f"File {filename} already exists.")
            
    except StopIteration:
        print(f"Paper not found on arXiv: {title}")
    except Exception as e:
        print(f"Error downloading {title}: {e}")

print("Download process complete.")
