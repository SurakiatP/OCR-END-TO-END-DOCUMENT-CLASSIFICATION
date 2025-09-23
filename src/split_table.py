# src/split_table.py

import cv2
import numpy as np
import os
import easyocr
from pathlib import Path

class TableDetectorSeparator:
    def __init__(self, min_table_area=5000, line_thickness=2):
        """
        Initialize Table Detector and Separator (Image Processing Only)
        
        Args:
            min_table_area (int): Minimum area for table detection
            line_thickness (int): Thickness for morphological operations
        """
        self.reader = easyocr.Reader(['th', 'en'])
        self.min_table_area = min_table_area
        self.line_thickness = line_thickness
        
    def preprocess_for_table_detection(self, image):
        """Preprocess image for table detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return gray, binary
    
    def detect_table_lines(self, binary_image):
        """Detect horizontal and vertical lines of tables"""
        # Create kernel for horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Create kernel for vertical lines  
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine all lines
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        return horizontal_lines, vertical_lines, table_mask
    
    def find_table_contours(self, table_mask):
        """Find contours of tables"""
        # Improve mask
        kernel = np.ones((3, 3), np.uint8)
        table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        table_mask = cv2.morphologyEx(table_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter tables
        table_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_table_area:
                # Check shape (should be rectangular)
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) >= 4:  # At least 4 points
                    table_contours.append(contour)
        
        return table_contours
    
    def has_table(self, image_path):
        """Check if image contains tables"""
        image = cv2.imread(image_path)
        if image is None:
            return False, "Cannot read image"
        
        _, binary = self.preprocess_for_table_detection(image)
        _, _, table_mask = self.detect_table_lines(binary)
        table_contours = self.find_table_contours(table_mask)
        
        has_table = len(table_contours) > 0
        message = f"Found {len(table_contours)} tables" if has_table else "No tables found"
        
        return has_table, message
    
    def mask_baht_in_table(self, table_image):
        """Mask the word '(บาท)' in table images"""
        try:
            # Read text to find positions of "บาท"
            results = self.reader.readtext(table_image)
            masked_image = table_image.copy()
            
            for result in results:
                bbox, text, confidence = result
                # Check if text contains "บาท" 
                if "บาท" in text.lower() and confidence > 0.3:
                    # Convert bbox to points
                    points = np.array(bbox, dtype=np.int32)
                    
                    # Fill the area with white color
                    cv2.fillPoly(masked_image, [points], (255, 255, 255))
                    print(f"Masked '(บาท)' at text: '{text}'")
            
            return masked_image
        except Exception as e:
            print(f"Error masking บาท: {e}")
            return table_image
    
    def extract_table_regions(self, image, table_contours):
        """Extract table images from main image and mask บาท"""
        table_images = []
        table_bboxes = []
        
        for i, contour in enumerate(table_contours):
            # Find bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add padding around table
            padding = 10
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            # Crop table image
            table_img = image[y_start:y_end, x_start:x_end]
            
            # Mask บาท in the table image
            masked_table_img = self.mask_baht_in_table(table_img)
            
            table_images.append({
                'image': masked_table_img,  # Use masked version
                'bbox': (x_start, y_start, x_end - x_start, y_end - y_start),
                'contour': contour,
                'index': i
            })
            
            table_bboxes.append((x_start, y_start, x_end - x_start, y_end - y_start))
        
        return table_images, table_bboxes
    
    def create_masked_image(self, original_image, table_bboxes):
        """Create image with tables masked as white rectangles"""
        masked_image = original_image.copy()
        
        for bbox in table_bboxes:
            x, y, w, h = bbox
            # Fill table area with white color
            cv2.rectangle(masked_image, (x, y), (x + w, y + h), (255, 255, 255), -1)
        
        return masked_image
    
    def save_results(self, image_path, table_images, masked_image, output_dir="output"):
        """Save processing results"""
        try:
            # Create output directory
            Path(output_dir).mkdir(exist_ok=True)
            
            # Get base filename
            base_name = Path(image_path).stem
            
            saved_files = []
            
            # Save individual table images
            for table_info in table_images:
                table_img = table_info['image']
                table_index = table_info['index']
                
                table_filename = f"{output_dir}/table_only.jpg"
                cv2.imwrite(table_filename, table_img)
                saved_files.append(table_filename)
                print(f"Saved table {table_index + 1}: {table_filename}")
            
            # Save masked image
            masked_filename = f"{output_dir}/doc_masked.jpg"
            cv2.imwrite(masked_filename, masked_image)
            saved_files.append(masked_filename)
            print(f"Saved masked image: {masked_filename}")
            
            return saved_files
            
        except Exception as e:
            print(f"Error saving files: {e}")
            return []
    
    def process_image(self, image_path, output_dir="output"):
        """Complete image processing pipeline (detection and separation only)"""
        print(f"=== Processing image: {image_path} ===")
        
        try:
            # Step 1: Check if image contains tables
            print("Step 1: Checking for tables")
            has_table, message = self.has_table(image_path)
            print(f"Detection result: {message}")
            
            if not has_table:
                return None, None, []
            
            # Step 2: Read image
            print("Step 2: Reading image")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
            print(f"Image loaded successfully, size: {image.shape}")
            
            # Step 3: Preprocess and detect tables
            print("Step 3: Preprocessing and detecting tables")
            _, binary = self.preprocess_for_table_detection(image)
            h_lines, v_lines, table_mask = self.detect_table_lines(binary)
            table_contours = self.find_table_contours(table_mask)
            print(f"Table detection completed, found contours: {len(table_contours)}")
            
            # Step 4: Extract table regions
            print("Step 4: Extracting table regions")
            table_images, table_bboxes = self.extract_table_regions(image, table_contours)
            print(f"Extracted {len(table_images)} tables")
            
            if not table_images:
                print("No tables could be extracted")
                return None, None, []
            
            # Step 5: Create masked image
            print("Step 5: Creating masked image")
            masked_image = self.create_masked_image(image, table_bboxes)
            print("Masked image created successfully")
            
            # Step 6: Save results
            print("Step 6: Saving results")
            saved_files = self.save_results(image_path, table_images, masked_image, output_dir)
            print(f"Files saved successfully, count: {len(saved_files)} files")
            
            return table_images, masked_image, saved_files
            
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return None, None, []


# Main function for easy usage (Updated paths for production)
def detect_and_separate_tables(image_path, output_dir=None, min_table_area=5000):
    """
    Main function for table detection and separation (image processing only)
    Updated for production use with proper path handling
    
    Args:
        image_path (str): Path to image file
        output_dir (str): Directory for saving results (if None, uses default temp/splited)
        min_table_area (int): Minimum table area
    
    Returns:
        dict: Processing results
    """
    # Set default output directory if not provided
    if output_dir is None:
        # Get the project root directory (where main.py is located)
        current_file = Path(__file__).absolute()
        project_root = current_file.parent.parent  # Go up from src/ to project root
        output_dir = project_root / "temp" / "splited"
        
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = str(output_dir)  # Convert Path to string
    
    detector = TableDetectorSeparator(min_table_area=min_table_area)
    
    try:
        print(f"Processing image: {image_path}")
        print(f"Output directory: {output_dir}")
        
        table_images, masked_image, saved_files = detector.process_image(
            image_path, output_dir
        )
        
        # Check results before creating result dict
        has_table = table_images is not None and len(table_images) > 0
        table_count = len(table_images) if table_images else 0
        safe_saved_files = saved_files if saved_files is not None else []
        
        result = {
            'success': True,
            'has_table': has_table,
            'table_count': table_count,
            'saved_files': safe_saved_files,
            'table_images': table_images,
            'masked_image': masked_image,
            'output_directory': output_dir
        }
        
        print(f"\n=== Table Processing Summary ===")
        print(f"Tables found: {result['table_count']}")
        print(f"Files saved: {len(result['saved_files'])}")
        print(f"Output directory: {result['output_directory']}")
        for file in result['saved_files']:
            print(f"  - {file}")
        
        return result
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        print("Error details:")
        traceback.print_exc()
        
        return {
            'success': False,
            'error': str(e),
            'has_table': False,
            'table_count': 0,
            'saved_files': [],
            'table_images': None,
            'masked_image': None,
            'output_directory': output_dir
        }

# Usage example for testing
if __name__ == "__main__":
    # Example usage - Updated paths for project structure
    current_file = Path(__file__).absolute()
    project_root = current_file.parent.parent  # Go up to project root
    
    # Test with sample file
    sample_file = project_root / "sample" / "1.jpg"
    output_directory = project_root / "temp" / "splited"
    
    if sample_file.exists():
        print("Starting table processing (6 Steps - detection and separation only)...")
        print("Features:")
        print("  - Step 1: Automatic table detection")
        print("  - Step 2: Image reading")
        print("  - Step 3: Image preprocessing and table detection")
        print("  - Step 4: Extract tables as separate files") 
        print("  - Step 5: Mask tables in original image")
        print("  - Step 6: Save results")
        print("  *** Includes masking '(บาท)' text in table images ***")
        print("-" * 50)
        
        # Process image
        result = detect_and_separate_tables(
            image_path=str(sample_file),
            output_dir=str(output_directory),
            min_table_area=2000  # Lower value to detect smaller tables
        )
        
        # Display results
        if result['success']:
            print(f"\nProcessing completed successfully!")
            print(f"Number of tables: {result['table_count']}")
            print(f"Files saved: {len(result['saved_files'])}")
            
            print(f"\nGenerated files:")
            for i, file in enumerate(result['saved_files'], 1):
                file_type = "Table image" if "table_" in file else "Masked image"
                print(f"  {i}. {file_type}: {file}")
                    
        else:
            print(f"Error occurred: {result['error']}")
            print("\nPlease check:")
            print("1. Install dependencies: pip install opencv-python easyocr")
            print("2. Image file exists in correct location")
            print("3. Supported file format (jpg, png, bmp, etc.)")
            print("4. Image contains clear tables")
            print("5. Table size is larger than min_table_area")
    else:
        print(f"Sample file not found: {sample_file}")
        print("Please place a test image in the sample/ directory")