import re

def parse_bib_to_md(bib_path, md_out_path):
    with open(bib_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Very rudimentary bibtex parser
    entries = re.findall(r'@\w+\{(.*?)\n(.*?)\n\}', content, re.DOTALL)
    
    formatted_refs = []
    
    for _, body in entries:
        title = re.search(r'title\s*=\s*[\{"](.*?)(?:[\}"]|\n)', body)
        author = re.search(r'author\s*=\s*[\{"](.*?)(?:[\}"]|\n)', body)
        year = re.search(r'year\s*=\s*[\{"]?(\d{4})[\}"]?', body)
        journal = re.search(r'(?:journal|booktitle)\s*=\s*[\{"](.*?)(?:[\}"]|\n)', body)
        
        t = title.group(1).replace('{', '').replace('}', '').strip() if title else "Unknown Title"
        a = author.group(1).replace('{', '').replace('}', '').strip() if author else "Unknown Author"
        y = year.group(1).strip() if year else "n.d."
        j = journal.group(1).replace('{', '').replace('}', '').strip() if journal else "arXiv / Preprint"
        
        # Format authors nicely (truncate if too many)
        authors_list = [name.strip() for name in a.split(' and ')]
        if len(authors_list) > 3:
            a_formatted = f"{authors_list[0]} et al."
        else:
            a_formatted = ", ".join(authors_list)
            
        md_ref = f"- **{a_formatted} ({y}).** *{t}.* {j}."
        formatted_refs.append(md_ref)
    
    # Sort alphabetically by author
    formatted_refs.sort()
    
    with open(md_out_path, 'w', encoding='utf-8') as f:
        f.write("## 5. Full References\n")
        f.write("\n".join(formatted_refs))

parse_bib_to_md("paper/high_quality_refs.bib", "formatted_refs.md")
print("Done formatting references.")
