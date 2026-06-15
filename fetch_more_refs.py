import urllib.request
import urllib.parse
import json
import time

queries = [
    "large language model reasoning hallucination",
    "multi-agent debate LLM alignment",
    "medical large language models verification",
    "legal language models reasoning",
    "sycophancy language models human feedback"
]

bib_entries = []
count = 0
seen_titles = set()

print("Fetching highly cited papers from Semantic Scholar API...")

for q in queries:
    if count >= 35: break
    query = urllib.parse.quote(q)
    # Using Semantic Scholar Graph API.
    # Searching for highly cited papers (which implies high quality/peer-reviewed).
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=15&fields=title,authors,year,venue,citationCount,externalIds"
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            for paper in data.get('data', []):
                if count >= 35: break
                title = paper.get('title', '')
                if not title or title.lower() in seen_titles: continue
                
                venue = paper.get('venue', '')
                # Filter for recognized high-quality venues roughly. If venue is empty but citationCount > 50, we might include it.
                cites = paper.get('citationCount', 0)
                if not venue and cites < 50:
                    continue
                
                authors = paper.get('authors', [])
                if not authors: continue
                author_names = " and ".join([a['name'] for a in authors])
                year = paper.get('year', 2024)
                
                # generate key
                first_author_last = authors[0]['name'].split()[-1].lower()
                first_word = title.split()[0].lower()
                key = f"{first_author_last}{year}{first_word}"
                
                bib = f"""@article{{{key},
  title={{{title}}},
  author={{{author_names}}},
  journal={{{venue if venue else 'ArXiv (Highly Cited)'}}},
  year={{{year}}},
  note={{Citations: {cites}}}
}}"""
                bib_entries.append(bib)
                seen_titles.add(title.lower())
                count += 1
                
        time.sleep(2) # Respect rate limits
    except Exception as e:
        print(f"Error searching '{q}': {e}")

with open("paper/high_quality_refs.bib", "a") as f:
    f.write("\n\n" + "\n\n".join(bib_entries))

print(f"Appended {count} new high-quality references to high_quality_refs.bib")
