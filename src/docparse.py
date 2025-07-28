import pymupdf  
import os
from collections import defaultdict

def parse_pdf_to_sections(pdf_path, doc_filename):
    """
    Parses a PDF file and extracts text content structured into sections.
    A simple heuristic based on font size is used to identify section titles.
    """
    sections = []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error opening or processing {pdf_path}: {e}")
        return []

    current_title = "Introduction" # Default title for text before the first heading
    current_text = ""
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=11)["blocks"]
        if not blocks:
            continue

        # Simple heuristic: find the most common font size as body text
        font_sizes = defaultdict(int)
        for b in blocks:
            if 'lines' in b:
                for l in b['lines']:
                    for s in l['spans']:
                        font_sizes[round(s['size'])] += 1
        
        body_font_size = max(font_sizes, key=font_sizes.get) if font_sizes else 10

        for b in blocks:
            if b['type'] == 0 and 'lines' in b: # It's a text block
                block_text = ""
                block_font_size = 0
                span_count = 0
                
                for l in b['lines']:
                    for s in l['spans']:
                        block_text += s['text'] + " "
                        block_font_size += s['size']
                        span_count += 1
                
                avg_font_size = block_font_size / span_count if span_count > 0 else 0
                block_text = block_text.strip()

                # Title heuristic: larger font, fewer words, not ending with a period.
                is_title = (
                    avg_font_size > (body_font_size + 2) and
                    len(block_text.split()) < 15 and
                    not block_text.endswith('.')
                )

                if is_title and block_text:
                    # Save the previous section
                    if current_text.strip():
                        sections.append({
                            "document": doc_filename,
                            "page_number": page_num + 1,
                            "section_title": current_title,
                            "section_text": current_text.strip()
                        })
                    
                    # Start a new section
                    current_title = block_text
                    current_text = ""
                else:
                    current_text += block_text + " "

    # Add the last section
    if current_text.strip():
        sections.append({
            "document": doc_filename,
            "page_number": page_num + 1,
            "section_title": current_title,
            "section_text": current_text.strip()
        })
        
    doc.close()
    return sections


