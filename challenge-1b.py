import os
import json
import re
from pathlib import Path
from docling.document_converter import DocumentConverter
from bs4 import BeautifulSoup
import numpy as np
from typing import List, Dict, Any, Optional
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime
import ollama

def convert_pdf_to_html_and_extract_headings(pdf_path: str, output_dir: str = "output"):
    """
    Converts PDF to HTML using Docling and extracts H1-H6 headings with page numbers.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory where output files will be saved.
    Returns:
        dict: Extracted headings data in the same format as file02.json
    """
    if not Path(pdf_path).is_file():
        print(f"Error: PDF file not found at {pdf_path}")
        return None

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Converting '{pdf_path}' to HTML using Docling...")
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        
        # Get HTML content
        html_content = result.document.export_to_html()
        
        # Save HTML to file
        html_file_path = Path(output_dir) / f"{Path(pdf_path).stem}.html"
        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML saved to: {html_file_path}")
        
        # Extract headings from HTML
        headings_data = extract_headings_from_html(html_content, pdf_path)
        
        # Save headings as JSON
        json_file_path = Path(output_dir) / f"{Path(pdf_path).stem}.json"
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(headings_data, f, indent=4, ensure_ascii=False)
        print(f"Headings JSON saved to: {json_file_path}")
        
        return headings_data

    except Exception as e:
        print(f"Error during conversion: {e}")
        return None

def extract_headings_from_html(html_content: str, pdf_path: str):
    """
    Extract H1-H6 headings from HTML content and verify page numbers by searching in PDF content
    
    Args:
        html_content (str): HTML content from PDF conversion
        pdf_path (str): Original PDF file path for title extraction and content search
    Returns:
        dict: Formatted headings data with accurate page numbers
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract document title (try multiple methods)
    title = extract_document_title(soup, pdf_path)
    
    # Extract PDF content page by page for searching
    print("Extracting PDF content for heading verification...")
    pdf_content_by_page = extract_pdf_content_by_pages(pdf_path)
    
    # Find all heading tags (H1-H6)
    heading_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    
    outline = []
    last_found_page = 0  # Track the last successfully found page
    
    for heading in heading_tags:
        # Get heading level (H1, H2, etc.)
        level = heading.name.upper()
        
        # Get heading text (clean up whitespace)
        text = heading.get_text(strip=True)
        
        if text and len(text) > 2:  # Only process meaningful text
            # Search for the heading text in PDF content to get accurate page number
            page_number = search_heading_in_pdf_content(text, pdf_content_by_page, last_found_page)
            
            # If not found, try fallback methods
            if page_number is None:
                page_number = estimate_page_from_surrounding_headings(text, outline, pdf_content_by_page, last_found_page)
            
            # Update last found page if we got a valid result
            if page_number and page_number > last_found_page:
                last_found_page = page_number
            
            outline.append({
                "level": level,
                "text": text,
                "page": page_number
            })
            
            # Log the result
            if page_number:
                print(f"✓ Found '{text[:50]}...' on page {page_number}")
            else:
                print(f"⚠ Could not locate '{text[:50]}...' - using fallback")
    
    # Create the final structure matching file02.json format
    headings_data = {
        "title": title,
        "outline": outline
    }
    
    return headings_data

def extract_pdf_content_by_pages(pdf_path: str):
    """
    Extract PDF content page by page using LangChain PyPDF loader
    
    Returns:
        dict: {page_number: content_text}
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        
        pdf_content_by_page = {}
        
        for i, doc in enumerate(documents):
            page_num = i + 1
            page_content = doc.page_content
            pdf_content_by_page[page_num] = page_content
        
        print(f"Extracted content from {len(documents)} pages for search")
        return pdf_content_by_page
        
    except ImportError:
        print("LangChain not available, falling back to basic page estimation")
        return {}
    except Exception as e:
        print(f"Error extracting PDF content: {e}")
        return {}

def search_heading_in_pdf_content(heading_text: str, pdf_content_by_page: dict, last_found_page: int = 0):
    """
    Search for heading text in PDF content to find the correct page number
    Uses smart search strategy: skip first occurrence, start from last found page + 1
    
    Args:
        heading_text: The heading text to search for
        pdf_content_by_page: Dictionary of PDF content by page number
        last_found_page: Page number of the last found heading (to start search after)
    Returns:
        int: Page number where heading is found, or None if not found
    """
    if not pdf_content_by_page:
        return None
    
    # Clean and prepare the heading text for search
    clean_heading = clean_text_for_search(heading_text)
    
    # Remove numbers and special characters from heading for better matching
    clean_heading_no_numbers = remove_numbers_and_special_chars(clean_heading)
    heading_words = clean_heading_no_numbers.split()
    
    # Determine starting page for search
    start_page = last_found_page + 1 if last_found_page > 0 else 1
    
    # Get pages to search (from start_page onwards)
    pages_to_search = [p for p in sorted(pdf_content_by_page.keys()) if p >= start_page]
    
    print(f"  Searching '{heading_text[:30]}...' -> '{clean_heading_no_numbers[:30]}...' starting from page {start_page}")
    
    # Strategy 1: Exact text match with occurrence counting (using cleaned text)
    exact_matches = []
    for page_num in pages_to_search:
        content = pdf_content_by_page[page_num]
        clean_content = clean_text_for_search(content)
        if clean_heading_no_numbers in clean_content:
            exact_matches.append(page_num)
    
    if exact_matches:
        # If only one occurrence found, return it immediately
        if len(exact_matches) == 1:
            selected_page = exact_matches[0]
            print(f"    Found single exact match on page {selected_page}")
        # If multiple occurrences and this is first search, skip first occurrence
        elif last_found_page == 0 and len(exact_matches) > 1:
            selected_page = exact_matches[1]  # Pick second occurrence
            print(f"    Found exact matches on pages {exact_matches}, picking second: {selected_page}")
        else:
            selected_page = exact_matches[0]  # Pick first available
            print(f"    Found exact match on page {selected_page}")
        return selected_page
    
    # Strategy 2: Partial match with most words (with occurrence logic)
    if len(heading_words) >= 2:
        partial_matches = []
        for page_num in pages_to_search:
            content = pdf_content_by_page[page_num]
            clean_content = clean_text_for_search(content)
            
            # Check if majority of words appear in the content
            word_matches = sum(1 for word in heading_words if word in clean_content)
            match_ratio = word_matches / len(heading_words)
            
            if match_ratio >= 0.7:  # At least 70% of words match
                partial_matches.append((page_num, match_ratio))
        
        if partial_matches:
            # Sort by match ratio (best first)
            partial_matches.sort(key=lambda x: x[1], reverse=True)
            
            # If only one match found, return it immediately
            if len(partial_matches) == 1:
                selected_page = partial_matches[0][0]
                print(f"    Found single partial match on page {selected_page}")
            # If multiple matches and this is first search, skip first occurrence
            elif last_found_page == 0 and len(partial_matches) > 1:
                selected_page = partial_matches[1][0]  # Pick second best
                print(f"    Found partial matches, picking second best: page {selected_page}")
            else:
                selected_page = partial_matches[0][0]  # Pick best
                print(f"    Found partial match on page {selected_page}")
            return selected_page
    
    # Strategy 3: Sequential word matching for multi-word headings
    if len(heading_words) >= 3:
        sequence_matches = []
        for page_num in pages_to_search:
            content = pdf_content_by_page[page_num]
            clean_content = clean_text_for_search(content)
            
            # Check for consecutive word sequences (using cleaned heading words)
            for i in range(len(heading_words) - 2):
                phrase = " ".join(heading_words[i:i+3])
                if phrase in clean_content:
                    sequence_matches.append(page_num)
                    break  # Only count once per page
        
        if sequence_matches:
            # If only one match found, return it immediately
            if len(sequence_matches) == 1:
                selected_page = sequence_matches[0]
                print(f"    Found single sequence match on page {selected_page}")
            # If multiple matches and this is first search, skip first occurrence
            elif last_found_page == 0 and len(sequence_matches) > 1:
                selected_page = sequence_matches[1]  # Pick second occurrence
                print(f"    Found sequence matches on pages {sequence_matches}, picking second: {selected_page}")
            else:
                selected_page = sequence_matches[0]  # Pick first available
                print(f"    Found sequence match on page {selected_page}")
            return selected_page
    
    print(f"    No matches found for '{heading_text[:30]}...'")
    return None

def estimate_page_from_surrounding_headings(heading_text: str, existing_outline: list, pdf_content_by_page: dict, last_found_page: int = 0):
    """
    Estimate page number based on surrounding headings that were successfully found
    
    Args:
        heading_text: The heading text that couldn't be found
        existing_outline: List of already processed headings
        pdf_content_by_page: PDF content by page
        last_found_page: The last successfully found page number
    Returns:
        int: Estimated page number
    """
    if not existing_outline:
        return 1
    
    # Use the last found page as reference if available
    if last_found_page > 0:
        # Try next page after the last found heading
        max_page = max(pdf_content_by_page.keys()) if pdf_content_by_page else 6
        next_page = min(last_found_page + 1, max_page)
        print(f"    Estimating page {next_page} based on last found page {last_found_page}")
        return next_page
    
    # Fallback: Get the last successfully found heading's page number from outline
    last_outline_page = None
    for heading in reversed(existing_outline):
        if heading.get("page"):
            last_outline_page = heading["page"]
            break
    
    if last_outline_page:
        # Try next page after the last found heading
        max_page = max(pdf_content_by_page.keys()) if pdf_content_by_page else 6
        next_page = min(last_outline_page + 1, max_page)
        print(f"    Estimating page {next_page} based on outline")
        return next_page
    
    # Final fallback to page 1
    return 1

def remove_numbers_and_special_chars(text: str):
    """
    Remove numbers and special characters from text, keeping only letters and spaces
    """
    # Remove numbers (digits)
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters except spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_text_for_search(text: str):
    """
    Clean text for better search matching
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common punctuation but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra spaces again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_document_title(soup, pdf_path):
    """
    Extract document title using multiple strategies
    """
    # Strategy 1: Look for HTML title tag
    title_tag = soup.find('title')
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)
    
    # Strategy 2: Look for first H1 tag
    first_h1 = soup.find('h1')
    if first_h1 and first_h1.get_text(strip=True):
        return first_h1.get_text(strip=True)
    
    # Strategy 3: Look for document header/title patterns
    for tag in soup.find_all(['div', 'p', 'span']):
        text = tag.get_text(strip=True)
        if len(text) > 10 and len(text) < 100:
            # Check if it looks like a title (has title-case or important keywords)
            if (text.istitle() or 
                any(word in text.lower() for word in ['overview', 'guide', 'manual', 'document', 'report'])):
                return text
    
    # Strategy 4: Use filename as fallback
    return Path(pdf_path).stem.replace('_', ' ').replace('-', ' ').title()

def extract_page_number(heading_element, soup):
    """
    Extract page number for a heading using multiple strategies
    """
    # Strategy 1: Look for page attributes or data attributes
    if heading_element.get('data-page'):
        try:
            return int(heading_element.get('data-page'))
        except:
            pass
    
    # Strategy 2: Look for page information in parent elements
    parent = heading_element.parent
    for _ in range(3):  # Check up to 3 parent levels
        if parent:
            if parent.get('data-page'):
                try:
                    return int(parent.get('data-page'))
                except:
                    pass
            parent = parent.parent
    
    # Strategy 3: Look for page markers in nearby text
    # Check siblings and nearby elements for page indicators
    for sibling in heading_element.find_next_siblings():
        text = sibling.get_text()
        page_match = re.search(r'page\s+(\d+)|p\.?\s*(\d+)', text.lower())
        if page_match:
            try:
                return int(page_match.group(1) or page_match.group(2))
            except:
                pass
    
    # Strategy 4: Look for page div/section containers
    page_container = heading_element.find_parent(['div', 'section'], class_=re.compile(r'page'))
    if page_container:
        page_text = page_container.get('class', [])
        for class_name in page_text:
            page_match = re.search(r'page[-_]?(\d+)', str(class_name))
            if page_match:
                try:
                    return int(page_match.group(1))
                except:
                    pass
    
    # Strategy 5: Sequential numbering (fallback)
    # Count position of heading in document and estimate page
    all_headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    heading_index = all_headings.index(heading_element) if heading_element in all_headings else 0
    
    # Rough estimate: assume ~3-5 headings per page
    estimated_page = max(1, (heading_index // 4) + 1)
    
    return estimated_page

def extract_section_content_from_html(html_content: str, headings_data: dict):
    """
    Extract content between headings from HTML
    
    Args:
        html_content: HTML content from PDF conversion
        headings_data: Dictionary with extracted headings information
    Returns:
        list: List of sections with content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    sections = []
    
    # Get all headings from the data
    outline = headings_data.get('outline', [])
    
    for i, heading_info in enumerate(outline):
        heading_text = heading_info['text']
        heading_level = heading_info['level']
        page_number = heading_info['page']
        
        # Find the heading element in HTML
        heading_element = None
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if tag.get_text(strip=True) == heading_text:
                heading_element = tag
                break
        
        if heading_element:
            # Extract content until next heading of same or higher level
            content_parts = []
            current_element = heading_element.next_sibling
            
            while current_element:
                if current_element.name and current_element.name.lower() in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    # Stop if we hit another heading of same or higher level
                    current_level = int(heading_level[1])
                    next_level = int(current_element.name[1])
                    if next_level <= current_level:
                        break
                
                # Extract text content
                if current_element.name:
                    text = current_element.get_text(strip=True)
                    if text and len(text) > 10:  # Only meaningful content
                        content_parts.append(text)
                
                current_element = current_element.next_sibling
            
            # Combine content
            section_content = ' '.join(content_parts)
            
            if section_content:
                sections.append({
                    'title': heading_text,
                    'level': heading_level,
                    'content': section_content,
                    'page': page_number,
                    'content_length': len(section_content)
                })
    
    return sections

def create_text_embeddings(text: str, model=None):
    """
    Create embeddings for text using SentenceTransformers
    
    Args:
        text: Text to embed
        model: SentenceTransformer model instance
    Returns:
        numpy array: Text embeddings
    """
    if model is None:
        # Use a lightweight but effective model
        model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dimensional embeddings
    
    # Create embedding
    embedding = model.encode(text, convert_to_numpy=True)
    
    return embedding.astype(np.float32)

def process_single_pdf_worker(pdf_path: str, output_directory: str, process_id: int):
    """
    Worker function to process a single PDF in a subprocess
    
    Args:
        pdf_path: Path to the PDF file
        output_directory: Base output directory
        process_id: Process identifier for logging
    Returns:
        dict: Processing result with sections data
    """
    pdf_file = Path(pdf_path)
    
    try:
        print(f"[Process {process_id}] 📄 Processing: {pdf_file.name}")
        
        # Create subdirectory for this PDF
        pdf_output_dir = Path(output_directory) / pdf_file.stem
        pdf_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract headings
        headings_data = convert_pdf_to_html_and_extract_headings(str(pdf_file), str(pdf_output_dir))
        
        if headings_data:
            # Read the generated HTML for section extraction
            html_file = pdf_output_dir / f"{pdf_file.stem}.html"
            if html_file.exists():
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Extract sections
                sections = extract_section_content_from_html(html_content, headings_data)
                
                # Save sections as JSON
                sections_file = pdf_output_dir / f"{pdf_file.stem}_sections.json"
                with open(sections_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'pdf_name': pdf_file.name,
                        'total_sections': len(sections),
                        'sections': sections
                    }, f, indent=4, ensure_ascii=False)
                
                print(f"[Process {process_id}] ✅ Successfully processed {pdf_file.name}")
                print(f"[Process {process_id}]    📋 Headings: {len(headings_data.get('outline', []))}")
                print(f"[Process {process_id}]    📝 Sections: {len(sections)}")
                
                return {
                    'pdf_file': pdf_file.name,
                    'pdf_path': str(pdf_file),
                    'status': 'success',
                    'headings_count': len(headings_data.get('outline', [])),
                    'sections_count': len(sections),
                    'output_dir': str(pdf_output_dir),
                    'sections': sections,
                    'process_id': process_id
                }
            else:
                print(f"[Process {process_id}] ❌ HTML file not generated for {pdf_file.name}")
                return {
                    'pdf_file': pdf_file.name,
                    'pdf_path': str(pdf_file),
                    'status': 'error',
                    'error': 'HTML file not generated',
                    'process_id': process_id
                }
        else:
            print(f"[Process {process_id}] ❌ Failed to extract headings from {pdf_file.name}")
            return {
                'pdf_file': pdf_file.name,
                'pdf_path': str(pdf_file),
                'status': 'error',
                'error': 'Failed to extract headings',
                'process_id': process_id
            }
    
    except Exception as e:
        print(f"[Process {process_id}] ❌ Error processing {pdf_file.name}: {e}")
        return {
            'pdf_file': pdf_file.name,
            'pdf_path': str(pdf_file),
            'status': 'error',
            'error': str(e),
            'process_id': process_id
        }

def print_progress_bar(completed: int, total: int, prefix: str = 'Progress', length: int = 40):
    """
    Print a progress bar to track processing
    """
    percent = 100 * (completed / float(total))
    filled_length = int(length * completed // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% ({completed}/{total}) Complete', end='', flush=True)
    if completed == total:
        print()  # New line when complete

def test_ollama_connection(ollama_model: str = "gemma3:1b"):
    """Test if Ollama is running and the model is available"""
    try:
        # Test if Ollama is running
        models = ollama.list()
        available_models = [model['name'] for model in models['models']]
        
        if ollama_model not in available_models:
            print(f"⚠️  Warning: Model '{ollama_model}' not found in available models: {available_models}")
            print(f"Available models: {', '.join(available_models)}")
            return False
        else:
            print(f"✅ Ollama connection successful. Using model: {ollama_model}")
            return True
            
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Make sure Ollama is running: 'ollama serve'")
        return False

def process_content_with_ollama(content: str, persona: str, job_to_be_done: str, ollama_model: str = "gemma3:1b") -> str:
    """
    Process content using Ollama Gemma model based on persona and job
    
    Args:
        content: Raw content to process
        persona: The persona/role
        job_to_be_done: The specific job/task
        ollama_model: Ollama model to use
        
    Returns:
        Refined/processed content
    """
    # Create prompt for Ollama
    prompt = f"""
You are a {persona}. Your task is to {job_to_be_done}.

Please analyze and refine the following content to make it more relevant and useful for your specific role and task. 
Focus on extracting the most important information that would help accomplish the job.

Content to analyze:
{content}

Instructions:
1. Keep only the most relevant information for a {persona} who needs to {job_to_be_done}
2. Reorganize the content to be more actionable and specific
3. Remove unnecessary details that don't directly help with the task
4. Make the content concise but comprehensive
5. Maintain all important specifics like names, locations, prices, etc.

Refined content:
"""
    
    try:
        print(f"🤖 Processing content with {ollama_model}...")
        
        # Call Ollama API
        response = ollama.generate(
            model=ollama_model,
            prompt=prompt,
            options={
                'temperature': 0.3,  # Lower temperature for more focused responses
                'top_p': 0.9,
                'max_tokens': 1000
            }
        )
        
        refined_content = response['response'].strip()
        
        # Clean up the response - remove any repeated prompt text
        if "Refined content:" in refined_content:
            refined_content = refined_content.split("Refined content:")[-1].strip()
        
        return refined_content
        
    except Exception as e:
        print(f"❌ Error processing content with Ollama: {e}")
        # Return original content if processing fails
        return content

def process_persona_job_with_chromadb(vector_db_path: str, persona: str, job_to_be_done: str, 
                                     output_file: str = None, top_k: int = 5, 
                                     ollama_model: str = "gemma3:1b") -> Dict[str, Any]:
    """
    Complete processing pipeline: extract relevant content and process with Ollama
    
    Args:
        vector_db_path: Path to ChromaDB vector database
        persona: The persona/role
        job_to_be_done: The specific job/task
        output_file: Optional output file path
        top_k: Number of sections to extract and process
        ollama_model: Ollama model to use
        
    Returns:
        Complete result dictionary matching the required format
    """
    print(f"\n{'='*60}")
    print(f"🎭 PERSONA-BASED CONTENT PROCESSING")
    print(f"{'='*60}")
    print(f"👤 Persona: {persona}")
    print(f"🎯 Job to be done: {job_to_be_done}")
    print(f"📊 Extracting top {top_k} relevant sections")
    
    # Test Ollama connection
    if not test_ollama_connection(ollama_model):
        return {"error": "Ollama connection failed"}
    
    # Initialize ChromaDB
    vector_db = ChromaVectorDatabase(vector_db_path)
    
    # Step 1: Extract relevant content
    relevant_content = vector_db.extract_relevant_content_for_persona(persona, job_to_be_done, top_k)
    
    if not relevant_content:
        return {"error": "No relevant content found"}
    
    # Step 2: Process content with Ollama
    print(f"\n🤖 Processing {len(relevant_content)} sections with {ollama_model}...")
    
    extracted_sections = []
    subsection_analysis = []
    
    for content_item in relevant_content:
        # Add to extracted sections
        extracted_sections.append({
            "document": content_item['document'],
            "section_title": content_item['section_title'],
            "importance_rank": content_item['importance_rank'],
            "page_number": content_item['page_number']
        })
        
        # Process content with Ollama
        refined_text = process_content_with_ollama(
            content_item['content'],
            persona,
            job_to_be_done,
            ollama_model
        )
        
        # Add to subsection analysis
        subsection_analysis.append({
            "document": content_item['document'],
            "refined_text": refined_text,
            "page_number": content_item['page_number']
        })
        
        print(f"  ✅ Processed: {content_item['section_title']}")
    
    # Step 3: Create final result structure
    result = {
        "metadata": {
            "input_documents": vector_db.get_input_documents_list(),
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    
    # Step 4: Save to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ PROCESSING COMPLETED")
    print(f"{'='*60}")
    print(f"📊 Total sections processed: {len(extracted_sections)}")
    print(f"📝 Input documents: {len(result['metadata']['input_documents'])}")
    print(f"💾 Output format: JSON")
    
    return result

def batch_process_pdfs_with_persona(pdf_directory: str, output_directory: str = "batch_output", 
                                   vector_db_path: str = "chroma_vector_db", max_workers: int = None,
                                   persona: str = None, job_to_be_done: str = None, 
                                   persona_output_file: str = None, ollama_model: str = "gemma3:1b"):
    """
    Process all PDF files in a directory using parallel processing, store in ChromaDB, 
    then extract and process content based on persona and job
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory for output files
        vector_db_path: Path to ChromaDB vector database directory
        max_workers: Maximum number of parallel processes (default: CPU count)
        persona: The persona/role for content extraction
        job_to_be_done: The specific job/task for content extraction
        persona_output_file: Output file for persona-processed content
        ollama_model: Ollama model to use for content processing
    Returns:
        dict: Processing results including persona processing
    """
    # First, run the parallel PDF processing
    batch_results = batch_process_pdfs_parallel(pdf_directory, output_directory, vector_db_path, max_workers)
    
    if not batch_results:
        return None
    
    # If persona and job are provided, do persona-based processing
    if persona and job_to_be_done:
        print(f"\n🎭 Starting persona-based content processing...")
        
        # Set default output file if not provided
        if not persona_output_file:
            persona_output_file = Path(output_directory) / f"persona_{persona.replace(' ', '_').lower()}_output.json"
        
        # Process content based on persona and job
        persona_results = process_persona_job_with_chromadb(
            vector_db_path=vector_db_path,
            persona=persona,
            job_to_be_done=job_to_be_done,
            output_file=str(persona_output_file),
            ollama_model=ollama_model
        )
        
        # Add persona results to batch results
        batch_results['persona_processing'] = persona_results
        batch_results['persona_output_file'] = str(persona_output_file)
    
    return batch_results

def batch_process_pdfs_parallel(pdf_directory: str, output_directory: str = "batch_output", vector_db_path: str = "chroma_vector_db", max_workers: int = None):
    """
    Process all PDF files in a directory using parallel processing and extract headings + content sections
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory for output files
        vector_db_path: Path to ChromaDB vector database directory
        max_workers: Maximum number of parallel processes (default: CPU count)
    Returns:
        dict: Processing results
    """
    pdf_dir_path = Path(pdf_directory)
    if not pdf_dir_path.exists():
        print(f"❌ Error: Directory '{pdf_directory}' not found")
        return None
    
    # Find all PDF files
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in '{pdf_directory}'")
        return None
    
    # Determine number of workers
    if max_workers is None:
        max_workers = min(len(pdf_files), mp.cpu_count())
    
    print(f"🚀 Found {len(pdf_files)} PDF files to process...")
    print(f"⚡ Using {max_workers} parallel processes...")
    
    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Process PDFs in parallel
    results = []
    successful_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all PDF processing tasks
        future_to_pdf = {
            executor.submit(process_single_pdf_worker, str(pdf_file), output_directory, i): pdf_file 
            for i, pdf_file in enumerate(pdf_files, 1)
        }
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    successful_count += 1
                
                completed_count += 1
                print_progress_bar(completed_count, len(pdf_files), 'Processing PDFs')
                    
            except Exception as e:
                print(f"❌ Exception processing {pdf_file.name}: {e}")
                results.append({
                    'pdf_file': pdf_file.name,
                    'pdf_path': str(pdf_file),
                    'status': 'error',
                    'error': f'Exception: {str(e)}'
                })
                completed_count += 1
                print_progress_bar(completed_count, len(pdf_files), 'Processing PDFs')
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Now store all successful results in ChromaDB
    print(f"\n💾 Storing results in ChromaDB Vector Database...")
    vector_db = ChromaVectorDatabase(vector_db_path)
    
    total_sections_added = 0
    for result in results:
        if result['status'] == 'success' and 'sections' in result:
            sections = result['sections']
            pdf_name = result['pdf_file']
            
            # Add sections to ChromaDB
            for section in sections:
                vector_db.add_document_section(
                    pdf_name=pdf_name,
                    section_title=section['title'],
                    content=section['content'],
                    page_number=section['page'],
                    level=section['level']
                )
                total_sections_added += 1
    
    # Save ChromaDB vector database
    vector_db.save_database()
    
    # Save batch processing results
    batch_results_file = Path(output_directory) / "batch_results.json"
    with open(batch_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processing_summary': {
                'total_files': len(pdf_files),
                'successful': successful_count,
                'failed': len(pdf_files) - successful_count,
                'processing_time_seconds': round(processing_time, 2),
                'processing_time_minutes': round(processing_time / 60, 2),
                'average_time_per_file': round(processing_time / len(pdf_files), 2),
                'max_workers': max_workers,
                'output_directory': output_directory,
                'vector_db_path': vector_db_path,
                'total_sections_stored': total_sections_added
            },
            'results': results,
            'vector_db_stats': vector_db.get_stats()
        }, f, indent=4, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🎯 PARALLEL BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"📊 Total files: {len(pdf_files)}")
    print(f"✅ Successful: {successful_count}")
    print(f"❌ Failed: {len(pdf_files) - successful_count}")
    print(f"⚡ Parallel processes: {max_workers}")
    print(f"⏱️  Processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    print(f"📈 Average time per file: {processing_time/len(pdf_files):.2f} seconds")
    print(f"📁 Output directory: {output_directory}")
    print(f"💾 ChromaDB Vector database: {vector_db_path}")
    print(f"📝 Total sections stored: {total_sections_added}")
    
    # Vector database stats
    db_stats = vector_db.get_stats()
    print(f"\n📚 CHROMADB VECTOR DATABASE STATS:")
    print(f"   Total sections: {db_stats['total_documents']}")
    print(f"   Embedding dimension: {db_stats['embedding_dimension']}")
    print(f"   Embedding model: {db_stats['embedding_model']}")
    print(f"   Average content length: {db_stats.get('average_content_length', 0):.0f} characters")
    print(f"   Database path: {db_stats['database_path']}")
    
    return {
        'results': results,
        'vector_db': vector_db,
        'stats': db_stats,
        'processing_time': processing_time,
        'max_workers': max_workers,
        'total_sections_added': total_sections_added
    }

class ChromaVectorDatabase:
    """
    ChromaDB-based vector database to store document sections with embeddings
    """
    def __init__(self, db_path: str = "chroma_vector_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_name = "pdf_sections"
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"📖 Loaded existing ChromaDB collection with {self.collection.count()} documents")
        except Exception:
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            print(f"🔧 Created new ChromaDB collection")
        
        print(f"🔧 Initialized ChromaDB Vector Database")
        print(f"   📐 Embedding dimension: {self.embedding_dim}")
        print(f"   🤖 Model: {embedding_model}")
        print(f"   📊 Current documents: {self.collection.count()}")
    
    def add_document_section(self, pdf_name: str, section_title: str, content: str, page_number: int, level: str):
        """Add a document section to the ChromaDB vector database"""
        # Create unique ID for this section
        doc_id = f"{pdf_name}_{level}_{page_number}_{hash(section_title) % 10000}"
        
        # Create embedding
        embedding = create_text_embeddings(content, self.embedding_model)
        
        # Store document info
        metadata = {
            'pdf_name': pdf_name,
            'section_title': section_title,
            'page_number': page_number,
            'level': level,
            'content_length': len(content)
        }
        
        # Add to ChromaDB collection
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding.tolist()],
            ids=[doc_id]
        )
        
        print(f"  📝 Added section: '{section_title}' from {pdf_name} (page {page_number})")
    
    def save_database(self):
        """ChromaDB automatically persists data, but we can force a checkpoint"""
        # ChromaDB automatically saves to disk, but we can print confirmation
        count = self.collection.count()
        print(f"💾 ChromaDB Vector database saved with {count} documents to: {self.db_path}")
    
    def load_database(self):
        """ChromaDB automatically loads persisted data"""
        # This is handled in __init__ when we get the collection
        pass
    
    def search_similar(self, query_text: str, top_k: int = 5):
        """Search for similar documents using ChromaDB"""
        if self.collection.count() == 0:
            return []
        
        # Create query embedding
        query_embedding = create_text_embeddings(query_text, self.embedding_model)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count())
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            document = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            
            # Convert distance to similarity (ChromaDB returns distance, not similarity)
            similarity = 1.0 - distance
            
            formatted_results.append({
                'similarity': float(similarity),
                'metadata': metadata,
                'content': document[:200] + "..." if len(document) > 200 else document
            })
        
        return formatted_results
    
    def get_stats(self):
        """Get database statistics"""
        count = self.collection.count()
        if count == 0:
            return {"total_documents": 0}
        
        # Get all documents to compute stats
        all_results = self.collection.get()
        
        pdf_counts = {}
        level_counts = {}
        total_content_length = 0
        
        for metadata in all_results['metadatas']:
            pdf_name = metadata['pdf_name']
            level = metadata['level']
            content_length = metadata.get('content_length', 0)
            
            pdf_counts[pdf_name] = pdf_counts.get(pdf_name, 0) + 1
            level_counts[level] = level_counts.get(level, 0) + 1
            total_content_length += content_length
        
        return {
            "total_documents": count,
            "embedding_dimension": self.embedding_dim,
            "embedding_model": self.embedding_model_name,
            "pdf_counts": pdf_counts,
            "level_counts": level_counts,
            "average_content_length": total_content_length / count if count > 0 else 0,
            "total_content_length": total_content_length,
            "database_path": self.db_path
        }
    
    def delete_by_pdf(self, pdf_name: str):
        """Delete all sections from a specific PDF"""
        # Get all documents and find ones from this PDF
        all_results = self.collection.get()
        ids_to_delete = []
        
        for i, metadata in enumerate(all_results['metadatas']):
            if metadata['pdf_name'] == pdf_name:
                ids_to_delete.append(all_results['ids'][i])
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            print(f"🗑️  Deleted {len(ids_to_delete)} sections from {pdf_name}")
            return len(ids_to_delete)
        return 0
    
    def extract_relevant_content_for_persona(self, persona: str, job_to_be_done: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Extract relevant content from vector database based on persona and job
        
        Args:
            persona: The persona/role (e.g., "Travel Planner")
            job_to_be_done: The specific job/task (e.g., "Plan a trip of 4 days for a group of 10 college friends")
            top_k: Number of most relevant sections to extract
        
        Returns:
            List of relevant content sections with metadata
        """
        # Create search query combining persona and job
        search_query = f"{persona} {job_to_be_done}"
        
        print(f"🔍 Searching for content relevant to: '{search_query}'")
        
        # Search in vector database
        search_results = self.search_similar(search_query, top_k=top_k)
        
        if not search_results:
            print("❌ No relevant content found in vector database")
            return []
        
        # Format results
        relevant_content = []
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            content = result['content']
            similarity = result['similarity']
            
            # Get full content from vector database (not truncated)
            full_content = self._get_full_content_for_section(metadata)
            
            relevant_content.append({
                'importance_rank': i,
                'similarity_score': similarity,
                'document': metadata['pdf_name'],
                'section_title': metadata['section_title'],
                'page_number': metadata['page_number'],
                'level': metadata['level'],
                'content': full_content or content,
                'content_length': metadata.get('content_length', len(content))
            })
            
            print(f"  {i}. {metadata['section_title']} (Similarity: {similarity:.3f})")
            print(f"     📄 {metadata['pdf_name']} | Page: {metadata['page_number']}")
        
        return relevant_content
    
    def _get_full_content_for_section(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Get full content for a section from vector database
        """
        try:
            # Query for the specific document to get full content
            results = self.collection.get(
                where={"$and": [
                    {"pdf_name": metadata['pdf_name']},
                    {"section_title": metadata['section_title']},
                    {"page_number": metadata['page_number']}
                ]}
            )
            
            if results and results['documents']:
                return results['documents'][0]
            
        except Exception as e:
            print(f"Warning: Could not retrieve full content: {e}")
        
        return None
    
    def get_input_documents_list(self) -> List[str]:
        """Get list of all input documents from vector database"""
        try:
            stats = self.get_stats()
            pdf_counts = stats.get('pdf_counts', {})
            return list(pdf_counts.keys())
        except Exception as e:
            print(f"Warning: Could not retrieve input documents: {e}")
            return []

def batch_process_pdfs(pdf_directory: str, output_directory: str = "batch_output", vector_db_path: str = "chroma_vector_db"):
    """
    Process all PDF files in a directory and extract headings + content sections
    
    Args:
        pdf_directory: Directory containing PDF files
        output_directory: Directory for output files
        vector_db_path: Path to ChromaDB vector database directory
    Returns:
        dict: Processing results
    """
    pdf_dir_path = Path(pdf_directory)
    if not pdf_dir_path.exists():
        print(f"❌ Error: Directory '{pdf_directory}' not found")
        return None
    
    # Find all PDF files
    pdf_files = list(pdf_dir_path.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in '{pdf_directory}'")
        return None
    
    print(f"🚀 Found {len(pdf_files)} PDF files to process...")
    
    # Initialize ChromaDB vector database
    vector_db = ChromaVectorDatabase(vector_db_path)
    
    # Create output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    results = []
    successful_count = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'='*60}")
        print(f"📄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")
        print(f"{'='*60}")
        
        try:
            # Create subdirectory for this PDF
            pdf_output_dir = Path(output_directory) / pdf_file.stem
            pdf_output_dir.mkdir(exist_ok=True)
            
            # Extract headings
            headings_data = convert_pdf_to_html_and_extract_headings(str(pdf_file), str(pdf_output_dir))
            
            if headings_data:
                # Read the generated HTML for section extraction
                html_file = pdf_output_dir / f"{pdf_file.stem}.html"
                if html_file.exists():
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Extract sections
                    sections = extract_section_content_from_html(html_content, headings_data)
                    
                    # Add sections to ChromaDB vector database
                    for section in sections:
                        vector_db.add_document_section(
                            pdf_name=pdf_file.name,
                            section_title=section['title'],
                            content=section['content'],
                            page_number=section['page'],
                            level=section['level']
                        )
                    
                    # Save sections as JSON
                    sections_file = pdf_output_dir / f"{pdf_file.stem}_sections.json"
                    with open(sections_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'pdf_name': pdf_file.name,
                            'total_sections': len(sections),
                            'sections': sections
                        }, f, indent=4, ensure_ascii=False)
                    
                    results.append({
                        'pdf_file': pdf_file.name,
                        'status': 'success',
                        'headings_count': len(headings_data.get('outline', [])),
                        'sections_count': len(sections),
                        'output_dir': str(pdf_output_dir)
                    })
                    successful_count += 1
                    
                    print(f"✅ Successfully processed {pdf_file.name}")
                    print(f"   📋 Headings: {len(headings_data.get('outline', []))}")
                    print(f"   📝 Sections: {len(sections)}")
                    
                else:
                    results.append({
                        'pdf_file': pdf_file.name,
                        'status': 'error',
                        'error': 'HTML file not generated'
                    })
            else:
                results.append({
                    'pdf_file': pdf_file.name,
                    'status': 'error',
                    'error': 'Failed to extract headings'
                })
        
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            results.append({
                'pdf_file': pdf_file.name,
                'status': 'error',
                'error': str(e)
            })
    
    # Save ChromaDB vector database
    vector_db.save_database()
    
    # Save batch processing results
    batch_results_file = Path(output_directory) / "batch_results.json"
    with open(batch_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'processing_summary': {
                'total_files': len(pdf_files),
                'successful': successful_count,
                'failed': len(pdf_files) - successful_count,
                'output_directory': output_directory,
                'vector_db_path': vector_db_path
            },
            'results': results,
            'vector_db_stats': vector_db.get_stats()
        }, f, indent=4, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"🎯 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"📊 Total files: {len(pdf_files)}")
    print(f"✅ Successful: {successful_count}")
    print(f"❌ Failed: {len(pdf_files) - successful_count}")
    print(f"📁 Output directory: {output_directory}")
    print(f"💾 ChromaDB Vector database: {vector_db_path}")
    
    # Vector database stats
    db_stats = vector_db.get_stats()
    print(f"\n📚 CHROMADB VECTOR DATABASE STATS:")
    print(f"   Total sections: {db_stats['total_documents']}")
    print(f"   Embedding dimension: {db_stats['embedding_dimension']}")
    print(f"   Embedding model: {db_stats['embedding_model']}")
    print(f"   Average content length: {db_stats.get('average_content_length', 0):.0f} characters")
    print(f"   Database path: {db_stats['database_path']}")
    
    return {
        'results': results,
        'vector_db': vector_db,
        'stats': db_stats
    }

def search_vector_database(query: str, vector_db_path: str = "chroma_vector_db", top_k: int = 5):
    """
    Search the ChromaDB vector database for similar content
    
    Args:
        query: Search query
        vector_db_path: Path to ChromaDB vector database directory
        top_k: Number of results to return
    Returns:
        list: Search results
    """
    vector_db = ChromaVectorDatabase(vector_db_path)
    results = vector_db.search_similar(query, top_k)
    
    print(f"\n🔍 Search Results for: '{query}'")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        print(f"{i}. {meta['section_title']} (Similarity: {result['similarity']:.3f})")
        print(f"   📄 PDF: {meta['pdf_name']} | Page: {meta['page_number']} | Level: {meta['level']}")
        print(f"   📝 Content: {result['content']}")
        print("-" * 60)
    
    return results

def display_extracted_headings(headings_data):
    """
    Display extracted headings in a nice format
    """
    if not headings_data:
        print("No headings data to display")
        return
    
    print("\n" + "="*60)
    print(f"DOCUMENT TITLE: {headings_data['title']}")
    print("="*60)
    print(f"EXTRACTED HEADINGS ({len(headings_data['outline'])} found):")
    print("-"*60)
    
    for item in headings_data['outline']:
        level = item['level']
        text = item['text']
        page = item['page']
        
        # Add indentation based on heading level
        indent = "  " * (int(level[1]) - 1) if len(level) > 1 else ""
        level_symbol = "■" if level == "H1" else "▪" if level == "H2" else "•"
        
        print(f"{indent}{level_symbol} [{level}] {text} (Page {page})")

def init_multiprocessing():
    """
    Initialize multiprocessing settings for cross-platform compatibility
    """
    # Set multiprocessing start method for Windows compatibility
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

if __name__ == "__main__":
    # Initialize multiprocessing
    init_multiprocessing()
    
    # Configuration
    batch_mode = True  # Set to False for single file processing
    use_parallel_processing = True  # Set to False for sequential processing
    max_workers = None  # None = auto-detect CPU cores, or set specific number like 4
    
    # Persona-based processing configuration
    enable_persona_processing = True  # Set to True to enable persona-based content extraction
    persona = "Travel Planner"  # The persona/role
    job_to_be_done = "Plan a trip of 4 days for a group of 10 college friends"  # The specific job/task
    ollama_model = "gemma3:1b"  # Ollama model to use
    
    if batch_mode:
        # Batch processing mode
        pdf_directory = "."  # Current directory (change to your PDF folder)
        output_directory = "batch_extracted_headings"
        vector_db_path = "pdf_sections_chroma_db"
        
        # Check if PDFs exist
        pdf_dir_path = Path(pdf_directory)
        pdf_files = list(pdf_dir_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ No PDF files found in '{pdf_directory}'")
            print("Please add PDF files to the directory or update the pdf_directory path")
            exit(1)
        
        print(f"📄 Found PDF files: {[f.name for f in pdf_files]}")
        
        if use_parallel_processing and enable_persona_processing:
            print("🚀 Starting PARALLEL BATCH PDF processing with ChromaDB Vector Database + Persona Processing...")
            print(f"📁 PDF Directory: {pdf_directory}")
            print(f"📁 Output Directory: {output_directory}")
            print(f"💾 ChromaDB Vector Database: {vector_db_path}")
            print(f"⚡ Max Workers: {max_workers if max_workers else 'Auto-detect'}")
            print(f"🖥️  Available CPU cores: {mp.cpu_count()}")
            print(f"👤 Persona: {persona}")
            print(f"🎯 Job to be done: {job_to_be_done}")
            print(f"🤖 Ollama Model: {ollama_model}")
            
            # Process all PDFs in parallel and then do persona processing
            batch_results = batch_process_pdfs_with_persona(
                pdf_directory=pdf_directory,
                output_directory=output_directory,
                vector_db_path=vector_db_path,
                max_workers=max_workers,
                persona=persona,
                job_to_be_done=job_to_be_done,
                persona_output_file=f"{output_directory}/persona_processed_output.json",
                ollama_model=ollama_model
            )
            
        elif use_parallel_processing:
            print("🚀 Starting PARALLEL BATCH PDF processing with ChromaDB Vector Database...")
            print(f"📁 PDF Directory: {pdf_directory}")
            print(f"📁 Output Directory: {output_directory}")
            print(f"💾 ChromaDB Vector Database: {vector_db_path}")
            print(f"⚡ Max Workers: {max_workers if max_workers else 'Auto-detect'}")
            print(f"🖥️  Available CPU cores: {mp.cpu_count()}")
            
            # Process all PDFs in parallel
            batch_results = batch_process_pdfs_parallel(pdf_directory, output_directory, vector_db_path, max_workers)
        else:
            print("🚀 Starting SEQUENTIAL BATCH PDF processing with ChromaDB Vector Database...")
            print(f"📁 PDF Directory: {pdf_directory}")
            print(f"📁 Output Directory: {output_directory}")
            print(f"💾 ChromaDB Vector Database: {vector_db_path}")
            
            # Process all PDFs sequentially (original function)
            batch_results = batch_process_pdfs(pdf_directory, output_directory, vector_db_path)
        
        if batch_results:
            print(f"\n✅ Batch processing completed!")
            print(f"📁 Check output files in: {output_directory}/")
            print(f"💾 ChromaDB Vector database saved to: {vector_db_path}")
            
            if 'processing_time' in batch_results:
                print(f"⏱️  Total processing time: {batch_results['processing_time']:.2f} seconds")
                if 'max_workers' in batch_results:
                    print(f"⚡ Used {batch_results['max_workers']} parallel processes")
            
            # Show persona processing results if available
            if 'persona_processing' in batch_results:
                print(f"\n🎭 Persona processing completed!")
                print(f"📄 Persona output file: {batch_results.get('persona_output_file', 'N/A')}")
                
                persona_results = batch_results['persona_processing']
                if persona_results and 'extracted_sections' in persona_results:
                    print(f"📊 Extracted {len(persona_results['extracted_sections'])} relevant sections")
                    
                    # Show first few extracted sections
                    print(f"\n📋 Top extracted sections:")
                    for i, section in enumerate(persona_results['extracted_sections'][:3], 1):
                        print(f"  {i}. {section['section_title']} (Rank: {section['importance_rank']})")
                        print(f"     📄 {section['document']} | Page: {section['page_number']}")
            
            # Example search
            print(f"\n🔍 Example search in ChromaDB vector database:")
            search_results = search_vector_database("introduction", vector_db_path, top_k=3)
            
            # Show how to use the ChromaDB vector database
            print(f"\n💡 ChromaDB Vector Database Usage Examples:")
            print(f"   # Search for content:")
            print(f"   search_vector_database('your query', '{vector_db_path}')")
            print(f"   ")
            print(f"   # Load database for custom operations:")
            print(f"   from rough import ChromaVectorDatabase")
            print(f"   db = ChromaVectorDatabase('{vector_db_path}')")
            print(f"   stats = db.get_stats()")
            print(f"   ")
            print(f"   # Persona-based processing:")
            print(f"   result = process_persona_job_with_chromadb('{vector_db_path}', 'Your Persona', 'Your Job')")
            print(f"   ")
            if use_parallel_processing and 'processing_time' in batch_results:
                estimated_sequential_time = batch_results['processing_time'] * batch_results.get('max_workers', 1)
                print(f"   # Performance comparison:")
                print(f"   Parallel processing time: {batch_results['processing_time']:.2f}s")
                print(f"   Estimated sequential time: ~{estimated_sequential_time:.2f}s")
                print(f"   Speed improvement: ~{estimated_sequential_time/batch_results['processing_time']:.1f}x faster")
        
    else:
        # Single file processing mode
        pdf_file = "file02.pdf"  # Change this to your PDF file
        output_directory = "extracted_headings"
        
        # Check if PDF exists
        if not Path(pdf_file).exists():
            print(f"Error: {pdf_file} not found")
            print("Available PDF files:")
            for pdf in Path(".").glob("*.pdf"):
                print(f"  - {pdf.name}")
            exit(1)
        
        # Convert PDF and extract headings
        print("Starting PDF to HTML conversion and heading extraction...")
        headings_data = convert_pdf_to_html_and_extract_headings(pdf_file, output_directory)
        
        # Display results
        if headings_data:
            display_extracted_headings(headings_data)
            
            # Extract sections and add to vector database
            html_file_path = Path(output_directory) / f"{Path(pdf_file).stem}.html"
            if html_file_path.exists():
                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                sections = extract_section_content_from_html(html_content, headings_data)
                
                # Create ChromaDB vector database for single file
                vector_db = ChromaVectorDatabase("single_file_chroma_db")
                for section in sections:
                    vector_db.add_document_section(
                        pdf_name=pdf_file,
                        section_title=section['title'],
                        content=section['content'],
                        page_number=section['page'],
                        level=section['level']
                    )
                vector_db.save_database()
                
                print(f"\n📝 Extracted {len(sections)} sections")
                print(f"💾 ChromaDB Vector database saved to: single_file_chroma_db")
            
            print(f"\n✅ Process completed!")
            print(f"📁 Check output files in: {output_directory}/")
            print(f"📄 HTML file: {pdf_file.replace('.pdf', '.html')}")
            print(f"📋 JSON file: {pdf_file.replace('.pdf', '.json')}")
        else:
            print("❌ Failed to extract headings")