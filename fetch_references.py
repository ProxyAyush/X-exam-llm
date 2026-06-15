import urllib.request
import xml.etree.ElementTree as ET
import time

queries = [
    "all:%22large+language+model%22+AND+all:%22self-correction%22",
    "all:%22multi-agent%22+AND+all:%22debate%22+AND+all:%22language+model%22",
    "all:%22sycophancy%22+AND+all:%22language+model%22",
    "all:%22hallucination%22+AND+all:%22language+model%22",
    "all:%22reasoning%22+AND+all:%22over-correction%22",
    "all:%22MedQA%22+AND+all:%22language+model%22",
    "all:%22TruthfulQA%22+AND+all:%22language+model%22"
]

bib_entries = []
count = 0

print("Fetching references from arXiv API...")

for q in queries:
    if count >= 110: break
    url = f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending"
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            if count >= 110: break
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.replace("\n", " ")
            authors = [a.find("{http://www.w3.org/2005/Atom}name").text for a in entry.findall("{http://www.w3.org/2005/Atom}author")]
            published = entry.find("{http://www.w3.org/2005/Atom}published").text[:4]
            id_url = entry.find("{http://www.w3.org/2005/Atom}id").text
            arxiv_id = id_url.split('/')[-1]
            
            author_str = " and ".join(authors)
            bib_id = f"arxiv{arxiv_id.replace('.','')}"
            
            bib = f"""@article{{{bib_id},
  title={{{title}}},
  author={{{author_str}}},
  journal={{arXiv preprint arXiv:{arxiv_id}}},
  year={{{published}}}
}}"""
            bib_entries.append((bib_id, bib))
            count += 1
        time.sleep(3) # Respect API limits
    except Exception as e:
        print(f"Error on query {q}: {e}")

with open("paper/references.bib", "w") as f:
    for _, bib in bib_entries:
        f.write(bib + "\n\n")

# Save a list of cite keys for easy insertion
with open("paper/cite_keys.txt", "w") as f:
    keys = [b[0] for b in bib_entries]
    f.write(",".join(keys))

print(f"Successfully fetched {count} references and saved to paper/references.bib")
