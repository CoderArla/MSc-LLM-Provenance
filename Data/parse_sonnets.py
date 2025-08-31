import re
import requests
import csv

def parse_sonnets_text(raw_text):
    """
    A robust parsing logic that reads the text line-by-line.
    """
    try:
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK SHAKESPEARE'S SONNETS ***"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK SHAKESPEARE'S SONNETS ***"
        start_index = raw_text.index(start_marker) + len(start_marker)
        end_index = raw_text.index(end_marker)
        sonnets_full_text = raw_text[start_index:end_index]
    except ValueError:
        print("Error: Could not find the standard Project Gutenberg start/end markers in the file.")
        return []

    # --- NEW, MORE ROBUST LOGIC ---
    lines = sonnets_full_text.strip().split('\n')
    
    documents = []
    current_sonnet_lines = []

    for line in lines:
        line = line.strip()
        # Check if the line is a Roman numeral (and nothing else)
        if re.fullmatch(r'[IVXLCDM]+', line):
            # If it is, and we have a sonnet in progress, save it
            if current_sonnet_lines:
                documents.append("\n".join(current_sonnet_lines))
                current_sonnet_lines = [] # Reset for the next sonnet
        elif line:
            # If it's a regular line of text, add it to the current sonnet
            current_sonnet_lines.append(line)
            
    # Add the very last sonnet after the loop has finished
    if current_sonnet_lines:
        documents.append("\n".join(current_sonnet_lines))
        
    return documents
    # --- ---------------------- ---

def parse_from_local_file(filepath):
    """
    Reads sonnets from a local text file and parses them.
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
    
    local_filepath = "sonnets.txt"
    sonnet_documents = parse_from_local_file(local_filepath)

    if sonnet_documents:
        print(f"\nSuccessfully parsed {len(sonnet_documents)} sonnets.")
        
        output_filename = "sonnets_dataset.csv"
        print(f"Saving the parsed sonnets to {output_filename}...")
        
        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['source_label', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, sonnet_text in enumerate(sonnet_documents):
                writer.writerow({'source_label': i, 'text': sonnet_text})
        
        print("Done. You can now use sonnets_dataset.csv as your dataset.")
    else:
        print("Parsing failed. No documents were extracted.")
