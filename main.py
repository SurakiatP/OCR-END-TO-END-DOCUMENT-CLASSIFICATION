# main.py

import os
import sys
import tempfile
from pathlib import Path
from PIL import Image
import shutil

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.split_table import detect_and_separate_tables
from src.text_detection import read_documents
from src.text_classifier import process_document_from_text

class DocumentProcessor:
    def __init__(self):
        """Initialize Document Processing Pipeline"""
        # Set up paths relative to main.py location
        self.base_dir = Path(__file__).parent.absolute()
        self.temp_dir = self.base_dir / "temp"
        self.splited_dir = self.temp_dir / "splited"
        self.detected_dir = self.temp_dir / "detected"
        self.meta_data_dir = self.base_dir / "meta_data"
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.temp_dir,
            self.splited_dir,
            self.detected_dir,
            self.meta_data_dir,
            self.base_dir / "csv_classifier"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_file(self, uploaded_file, filename=None):
        """
        Save uploaded file to temp directory
        
        Args:
            uploaded_file: File object from streamlit uploader
            filename: Optional custom filename
            
        Returns:
            str: Path to saved file
        """
        try:
            if filename is None:
                filename = uploaded_file.name
            
            # Generate unique filename to avoid conflicts
            import time
            timestamp = str(int(time.time()))
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{timestamp}{ext}"
            
            file_path = self.temp_dir / unique_filename
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            print(f"File saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            print(f"Error saving file: {e}")
            raise
    
    def process_document(self, input_file_path):
        """
        Complete document processing pipeline
        
        Args:
            input_file_path (str): Path to input document image
            
        Returns:
            dict: Processing results
        """
        print(f"=== Starting Document Processing Pipeline ===")
        print(f"Input file: {input_file_path}")
        
        results = {
            'success': False,
            'steps_completed': [],
            'errors': [],
            'output_files': {},
            'classification_results': None
        }
        
        try:
            # Step 1: Table Detection and Separation
            print(f"\n--- Step 1: Table Detection and Separation ---")
            split_results = detect_and_separate_tables(
                image_path=input_file_path,
                output_dir=str(self.splited_dir),
                min_table_area=2000
            )
            
            if not split_results['success']:
                results['errors'].append(f"Step 1 failed: {split_results.get('error', 'Unknown error')}")
                return results
            
            results['steps_completed'].append('table_separation')
            results['output_files']['split_results'] = split_results
            
            # Check if we have both text and table documents
            text_doc_path = self.splited_dir / "doc_masked.jpg"
            table_doc_path = self.splited_dir / "table_only.jpg"
            
            if not text_doc_path.exists() or not table_doc_path.exists():
                results['errors'].append("Required split files not found")
                return results
            
            # Step 2: Text Detection (OCR)
            print(f"\n--- Step 2: Text Detection and OCR ---")
            ocr_results = read_documents(
                text_document_path=str(text_doc_path),
                table_document_path=str(table_doc_path),
                confidence_threshold=0.25
            )
            
            if not ocr_results.get('success', True):  # read_documents doesn't return success flag
                results['errors'].append(f"Step 2 failed: {ocr_results.get('error', 'OCR error')}")
                return results
            
            results['steps_completed'].append('text_detection')
            results['output_files']['ocr_results'] = ocr_results
            
            # Check if text content file exists
            text_content_path = self.detected_dir / "doc_content.txt"
            if not text_content_path.exists():
                results['errors'].append("Text content file not generated")
                return results
            
            # Step 3: Text Classification and Data Extraction
            print(f"\n--- Step 3: Text Classification and Data Extraction ---")
            from src.text_classifier import TextDocumentProcessor
            
            classifier = TextDocumentProcessor()
            classifier.process_text_file(str(text_content_path))
            
            results['steps_completed'].append('text_classification')
            results['classification_results'] = "Processing completed"
            
            # Step 4: Verify output files
            print(f"\n--- Step 4: Verifying Output Files ---")
            meta_json_path = self.meta_data_dir / "meta.json"
            
            if meta_json_path.exists():
                results['success'] = True
                results['output_files']['meta_json'] = str(meta_json_path)
                print(f"✓ Meta data file created: {meta_json_path}")
            else:
                results['errors'].append("Meta data file not created")
            
            # Display summary
            print(f"\n=== Processing Summary ===")
            print(f"Steps completed: {', '.join(results['steps_completed'])}")
            print(f"Success: {results['success']}")
            
            if results['errors']:
                print(f"Errors: {', '.join(results['errors'])}")
            
            return results
            
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            import traceback
            traceback.print_exc()
            results['errors'].append(f"Pipeline error: {str(e)}")
            return results
    
    def get_processing_status(self):
        """Get current processing status"""
        status = {
            'directories_exist': True,
            'meta_file_exists': (self.meta_data_dir / "meta.json").exists(),
            'temp_files': []
        }
        
        # List temp files
        if self.temp_dir.exists():
            for file in self.temp_dir.rglob("*"):
                if file.is_file():
                    relative_path = file.relative_to(self.temp_dir)
                    status['temp_files'].append(str(relative_path))
        
        return status
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if self.temp_dir.exists():
                for item in self.temp_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                print("Temporary files cleaned up")
                return True
        except Exception as e:
            print(f"Error cleaning temp files: {e}")
            return False


# Convenience functions for direct usage
def process_single_document(file_path):
    """
    Process a single document file
    
    Args:
        file_path (str): Path to document image
        
    Returns:
        dict: Processing results
    """
    processor = DocumentProcessor()
    return processor.process_document(file_path)


def process_uploaded_file(uploaded_file):
    """
    Process an uploaded file from Streamlit
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        dict: Processing results
    """
    processor = DocumentProcessor()
    
    try:
        # Save uploaded file
        saved_path = processor.save_uploaded_file(uploaded_file)
        
        # Process document
        results = processor.process_document(saved_path)
        
        # Add file info to results
        results['input_file'] = {
            'original_name': uploaded_file.name,
            'saved_path': saved_path,
            'size': uploaded_file.size if hasattr(uploaded_file, 'size') else None
        }
        
        return results
        
    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        return {
            'success': False,
            'errors': [f"Upload processing error: {str(e)}"],
            'steps_completed': []
        }


if __name__ == "__main__":
    # Example usage for testing
    print("Document Processing Pipeline")
    print("=" * 50)
    
    # # Test with sample file
    # sample_file = Path(__file__).parent / "sample" / "1.jpg"
    
    # if sample_file.exists():
    #     print(f"Testing with sample file: {sample_file}")
    #     results = process_single_document(str(sample_file))
        
    #     print(f"\nResults:")
    #     print(f"Success: {results['success']}")
    #     print(f"Steps completed: {results['steps_completed']}")
        
    #     if results['errors']:
    #         print(f"Errors: {results['errors']}")
    # else:
    #     print(f"Sample file not found: {sample_file}")
    #     print("Please place a test image in the sample/ directory")
    
    # Show processor status
    processor = DocumentProcessor()
    status = processor.get_processing_status()
    print(f"\nProcessor Status:")
    print(f"Directories exist: {status['directories_exist']}")
    print(f"Meta file exists: {status['meta_file_exists']}")
    print(f"Temp files count: {len(status['temp_files'])}")