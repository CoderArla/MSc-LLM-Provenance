import re
import requests

def parse_sonnets_text(raw_text):
    """
    The core parsing logic to extract sonnets from a block of text.
    """
    # 1. Isolate the main body of the sonnets to remove the header/footer
    try:
        start_marker = "From fairest creatures we desire increase,"
        end_marker = "FINIS" # This marks the end of the last sonnet
        start_index = raw_text.index(start_marker)
        end_index = raw_text.index(end_marker)
        sonnets_full_text = raw_text[start_index:end_index]
    except ValueError:
        print("Warning: Could not find start/end markers. The file format may have changed.")
        return []

    # 2. Split the text block by the Roman numeral pattern that separates sonnets.
    documents = re.split(r'\n\s*[IVXLCDM]+\.\s*\n', sonnets_full_text)
    
    # 3. Clean up any extra whitespace from each sonnet and remove empty entries.
    cleaned_documents = [doc.strip() for doc in documents if doc.strip()]
    
    return cleaned_documents


def parse_from_local_file(filepath):
    """
    Option B: Reads sonnets from a local text file and parses them.
    """
    print(f"Reading from local file: {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        return parse_sonnets_text(raw_text)
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# --- Main execution ---
if __name__ == "__main__":
    
    # --- OPTION A: Download directly from the web ---
    # gutenberg_url = "https://www.gutenberg.org/ebooks/1041.txt.utf-8"
    # sonnet_documents = download_and_parse_sonnets(gutenberg_url)

    # --- OPTION B: Use your local sonnets.txt file ---
    # Make sure the path to your file is correct.
    local_filepath = "sonnets.txt"
    sonnet_documents = parse_from_local_file(local_filepath)

    # --- Verification ---
    if sonnet_documents:
        print(f"\nSuccessfully parsed {len(sonnet_documents)} sonnets.")
        
        # You now have a list where each item is a sonnet.
        # From here, you would save this list into a structured file (like CSV or JSON)
        # to be used by the rest of your project's code.
        
        print("\n--- Document 0 (Sonnet I) ---")
        print(sonnet_documents[0])
        
        print("\n--- Document 153 (Sonnet CLIV) ---")
        print(sonnet_documents[-1])