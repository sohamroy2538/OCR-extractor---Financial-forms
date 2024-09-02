from config import *
from utils import *
import argparse

def main(PDF_PATH):

    num_pages = OCR_preprocess(PDF_PATH, OUTPUT_DIR_IMG).extract_pages_from_pdf()
    print("PDF read successfully")
    res = OCR_mainprocess(num_pages).extract_fields()
    print("Info. extracted successfully")
    write_dicts_to_csv(res, OUTPUT_DIR_CSV)
    print("CSV written successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF and extract data.")
    parser.add_argument('PDF_PATH', nargs='?', default='', type=str, help="Path to the PDF file to be processed.")
    
    args = parser.parse_args()

    if not args.PDF_PATH:
        args.PDF_PATH = PDF_PATH  # default pdf path from config.py
    
    main(args.PDF_PATH)


