import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from urllib.parse import urljoin, quote
import re

# Configuration
# ... (Imports remain same)

# Configuration
OUTPUT_DIR = "D:/gujarati_project_data/dataset/raw_text"
WIKISOURCE_BASE_URL = "https://gu.wikisource.org"
MAX_DEPTH = 3  # Deeper crawl
MAX_PAGES_PER_SOURCE = 1000 # Production limit

# REAL Data Sources mapped to Dialects
DIALECT_SOURCES = {
    "saurashtra": [
        "https://gu.wikisource.org/wiki/" + quote("શ્રેણી:ઝવેરચંદ_મેઘાણી"), # Zaverchand Meghani
        "https://gu.wikisource.org/wiki/" + quote("શ્રેણી:લોકસાહિત્ય"),    # Folk Literature
        "https://gu.wikisource.org/wiki/" + quote("શ્રેણી:સૌરાષ્ટ્રની_રસધાર"), # Saurashtra Ni Rasdhar specifically
    ],
    "standard": [
        "https://gu.wikisource.org/wiki/" + quote("શ્રેણી:ગાંધીજી"),       # M.K. Gandhi
        "https://gu.wikisource.org/wiki/" + quote("શ્રેણી:નવલકથા"),        # Novels
        "https://gu.wikisource.org/wiki/" + quote("શ્રેણી:નર્મદ"),         # Narmad
    ],
    "kathiawadi": [
        "https://gu.wikisource.org/wiki/" + quote("શ્રેણી:દુહા_છંદ"),      # Duha/Chand
        "https://gu.wikisource.org/wiki/" + quote("શ્રેણી:ભજન"),           # Bhajans often have dialect
    ]
}

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_soup(url):
    try:
        headers = {
            'User-Agent': 'GujaratiDialectBot/2.0 (Recursive Crawler; mailto:researcher@example.com)'
        }
        time.sleep(0.5) # Polite delay
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def crawl_category(category_url, depth=0, visited=None):
    """
    Recursively finds pages in a category and its subcategories.
    Returns a set of page URLs.
    """
    if visited is None:
        visited = set()
    
    if depth > MAX_DEPTH or category_url in visited:
        return set()
    
    print(f"[{depth}] Crawling Category: {category_url}")
    visited.add(category_url)
    soup = get_soup(category_url)
    if not soup:
        return set()

    found_pages = set()
    
    # 1. Get Pages in this Category
    page_div = soup.find('div', id='mw-pages')
    if page_div:
        for a_tag in page_div.find_all('a', href=True):
            full_url = urljoin(WIKISOURCE_BASE_URL, a_tag['href'])
            found_pages.add(full_url)
    
    # 2. Get Subcategories (Recursion)
    subcat_div = soup.find('div', id='mw-subcategories')
    if subcat_div:
        for a_tag in subcat_div.find_all('a', href=True):
            subcat_url = urljoin(WIKISOURCE_BASE_URL, a_tag['href'])
            found_pages.update(crawl_category(subcat_url, depth + 1, visited))

    return found_pages

def scrape_text_from_page(page_url):
    soup = get_soup(page_url)
    if not soup:
        return None, None
    
    # Title
    header = soup.find('h1', id='firstHeading')
    title = header.get_text().strip() if header else "Unknown"
    
    # Content div
    content_div = soup.find('div', class_='mw-parser-output')
    if not content_div:
        return title, ""
        
    # Cleanup
    for tag in content_div(['script', 'style', 'table', 'div']):
        if tag.name == 'div' and 'poem' in (tag.get('class') or []):
            continue # KEEP poems (often in div class='poem')
        if tag.get('class') and 'navbox' in tag.get('class'):
            tag.decompose()
    
    # Extraction Strategy:
    # 1. Paragraphs <p>
    # 2. Poems <div class="poem"> or <poem> tag (Wikisource extension)
    # 3. Line breaks <br> in poems need handling
    
    text_parts = []
    
    # Handle standard paragraphs
    for p in content_div.find_all('p'):
        text_parts.append(p.get_text().strip())
        
    # Handle Poems (Critical for literature/dialect)
    # Wikisource often uses <div class="poem"> or specialized tags
    poems = content_div.find_all('div', class_='poem')
    for poem in poems:
        # replace <br> with newlines for structure preservation
        for br in poem.find_all('br'):
            br.replace_with('\n')
        text_parts.append(poem.get_text().strip())

    text_content = "\n".join([t for t in text_parts if t])
    
    # Clean citations
    text_content = re.sub(r'\[\d+\]', '', text_content)
    
    return title, text_content

def main():
    setup_directories()
    all_data = []
    
    for dialect, start_urls in DIALECT_SOURCES.items():
        print(f"\n=== Dialect: {dialect.upper()} ===")
        unique_pages = set()
        
        # 1. Gather all unique pages via recursive crawl
        for url in start_urls:
            pages = crawl_category(url)
            unique_pages.update(pages)
            
        print(f"Found {len(unique_pages)} unique pages total (limit {MAX_PAGES_PER_SOURCE}).")
        
        # 2. Scrape content
        count = 0
        for link in list(unique_pages):
            if count >= MAX_PAGES_PER_SOURCE: 
                break
                
            print(f"  Scraping ({count+1}): {link}")
            title, text = scrape_text_from_page(link)
            
            if text and len(text) > 300: # Stricter length filter
                all_data.append({
                    'source': 'wikisource',
                    'url': link,
                    'title': title,
                    'text_raw': text,
                    'dialect_label': dialect,
                    'confidence_score': 0.95 
                })
                count += 1
            
    if all_data:
        df = pd.DataFrame(all_data)
        import time
        timestamp = int(time.time())
        output_file = os.path.join(OUTPUT_DIR, f"wikisource_dialect_corpus_v2_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        print(f"\nSUCCESS: Saved {len(df)} dialect-labeled documents to {output_file}")
        print(df['dialect_label'].value_counts())
    else:
        print("\nWARNING: No data collected.")

if __name__ == "__main__":
    main()
