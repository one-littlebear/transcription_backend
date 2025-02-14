import re
import os
from docx import Document
from pathlib import Path

def split_into_sentences(text):
    """
    Split text into sentences using regex.
    Handles common sentence endings (., !, ?) while accounting for common abbreviations.
    """
    # First, handle some common abbreviations to prevent false splits
    text = re.sub(r'(?<=Mr)\.', '@', text)
    text = re.sub(r'(?<=Mrs)\.', '@', text)
    text = re.sub(r'(?<=Dr)\.', '@', text)
    
    # Split sentences based on .!? followed by spaces and capital letters
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Restore the periods in abbreviations
    sentences = [s.replace('@', '.') for s in sentences]
    
    # Clean up any extra whitespace
    sentences = [s.strip() for s in sentences]
    
    return sentences

def ensure_output_directory():
    """
    Create output/cleaned directory if it doesn't exist
    """
    output_dir = Path('output/cleaned')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def process_file(input_file_path):
    """
    Process a text file to put each sentence on a new line and save as docx.
    
    Args:
        input_file_path (str): Path to the input text file
    """
    try:
        # Create output directory
        output_dir = ensure_output_directory()
        
        # Generate output filename
        input_path = Path(input_file_path)
        output_filename = f"{input_path.stem}_cleaned.docx"
        output_path = output_dir / output_filename
        
        # Read the input file
        with open(input_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split into sentences
        sentences = split_into_sentences(text)
        
        # Create a new Word document
        doc = Document()
        
        # Add each sentence as a new paragraph
        for sentence in sentences:
            doc.add_paragraph(sentence)
        
        # Save the document
        doc.save(str(output_path))
                
        print(f"Successfully processed {input_file_path}")
        print(f"Saved to {output_path}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Process all txt files in the output folder
    input_dir = Path('output')
    
    # Loop through all .txt files in the output directory
    for txt_file in input_dir.glob('*.txt'):
        process_file(txt_file)
