# evaluation.py
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
import Levenshtein
from main import DocumentProcessor
import matplotlib
matplotlib.use('Agg')  # Disable interactive display

class OCREvaluator:
    """
    Evaluate OCR Document Processing System
    3 Main Metrics: CER, Field Accuracy, Speed
    """
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def load_test_data(self, test_file='test_data.json'):
        """Load test dataset from JSON file"""
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # ============ Metric 1: Character Error Rate ============
    def calculate_cer(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate Character Error Rate (CER)
        CER = Edit Distance / Length of Ground Truth
        Lower is better (0 = perfect match)
        """
        if not ground_truth:
            return 0.0
        distance = Levenshtein.distance(predicted, ground_truth)
        return distance / len(ground_truth)
    
    def evaluate_ocr_accuracy(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate OCR accuracy using Character Error Rate
        Processes each test case and compares extracted text with ground truth
        
        Returns:
            dict: OCR accuracy metrics including mean CER and character accuracy
        """
        cer_scores = []
        
        for test_case in test_data:
            # Process document through the pipeline
            result = self.processor.process_document(test_case['image_path'])
            
            if result['success']:
                # Read extracted text from output file
                text_file = self.processor.detected_dir / "doc_content.txt"
                with open(text_file, 'r', encoding='utf-8') as f:
                    predicted_text = f.read()
                
                # Compare with ground truth
                ground_truth = test_case['ground_truth']['full_text']
                cer = self.calculate_cer(predicted_text, ground_truth)
                cer_scores.append(cer)
        
        return {
            'mean_cer': np.mean(cer_scores),
            'char_accuracy': 1 - np.mean(cer_scores),
            'min_cer': np.min(cer_scores),
            'max_cer': np.max(cer_scores)
        }
    
    # ============ Metric 2: Field-level Accuracy ============
    def evaluate_field_accuracy(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate accuracy for each extracted field separately
        Compares extracted fields with ground truth values
        
        Returns:
            dict: Accuracy statistics for each field type
        """
        fields = ['invoice_no', 'date', 'seller_name', 'total_amount', 'vat']
        field_results = {field: {'correct': 0, 'total': 0} for field in fields}
        
        for test_case in test_data:
            # Process document
            result = self.processor.process_document(test_case['image_path'])
            
            if result['success']:
                # Load output JSON containing extracted data
                json_path = self.processor.meta_data_dir / "meta.json"
                with open(json_path, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                
                # Get the most recently processed document
                doc_type = test_case['ground_truth']['doc_type']
                if doc_type == 'invoice' and output_data['invoices']:
                    extracted = output_data['invoices'][-1]
                elif doc_type == 'receipt' and output_data['receipts']:
                    extracted = output_data['receipts'][-1]
                else:
                    continue
                
                # Compare each field with ground truth
                ground_truth = test_case['ground_truth']
                for field in fields:
                    if field in ground_truth:
                        field_results[field]['total'] += 1
                        
                        # Map generic field name to document-specific field name
                        output_field = self._map_field_name(field, doc_type)
                        
                        # Check if extracted value matches ground truth
                        if self._is_match(extracted.get(output_field), 
                                        ground_truth[field], 
                                        field):
                            field_results[field]['correct'] += 1
        
        # Calculate accuracy percentages
        accuracy_results = {}
        for field, stats in field_results.items():
            if stats['total'] > 0:
                accuracy_results[field] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'correct': stats['correct'],
                    'total': stats['total']
                }
        
        return accuracy_results
    
    def _map_field_name(self, field: str, doc_type: str) -> str:
        """
        Map generic field names to document-specific field names
        Invoice and receipt use different field names for similar data
        """
        mapping = {
            'invoice_no': 'invoice_no' if doc_type == 'invoice' else 'receipt_no',
            'date': 'invoice_date' if doc_type == 'invoice' else 'receipt_date',
            'seller_name': 'seller_name',
            'total_amount': 'total_amount' if doc_type == 'invoice' else 'total_paid',
            'vat': 'vat_amount'
        }
        return mapping.get(field, field)
    
    def _is_match(self, predicted, ground_truth, field_type: str) -> bool:
        """
        Check if predicted value matches ground truth
        Uses different matching strategies based on field type:
        - Numeric fields: tolerance of 0.01
        - Text fields: fuzzy string matching with 90% threshold
        """
        if predicted is None or ground_truth is None:
            return False
        
        # Numeric fields: allow small tolerance for floating point comparison
        if field_type in ['total_amount', 'vat']:
            try:
                return abs(float(predicted) - float(ground_truth)) < 0.01
            except:
                return False
        
        # Text fields: use fuzzy string matching
        from fuzzywuzzy import fuzz
        similarity = fuzz.ratio(str(predicted), str(ground_truth))
        return similarity >= 90
    
    # ============ Metric 3: Processing Speed ============
    def evaluate_speed(self, test_data: List[Dict]) -> Dict:
        """
        Measure processing speed for each document
        Calculates mean, median, and throughput statistics
        
        Returns:
            dict: Processing time statistics and throughput (docs/second)
        """
        processing_times = []
        
        for test_case in test_data:
            start_time = time.time()
            
            # Process document
            result = self.processor.process_document(test_case['image_path'])
            
            elapsed_time = time.time() - start_time
            
            if result['success']:
                processing_times.append(elapsed_time)
        
        return {
            'mean_time': np.mean(processing_times),
            'median_time': np.median(processing_times),
            'min_time': np.min(processing_times),
            'max_time': np.max(processing_times),
            'std_time': np.std(processing_times),
            'throughput': 1 / np.mean(processing_times)  # documents per second
        }
    
    # ============ Run Complete Evaluation ============
    def run_full_evaluation(self, test_file='test_data.json'):
        """
        Run all three evaluation metrics and generate report
        1. OCR Accuracy (Character Error Rate)
        2. Field-level Accuracy
        3. Processing Speed
        
        Saves results to JSON file for further analysis
        """
        print("=" * 60)
        print("OCR DOCUMENT PROCESSING EVALUATION")
        print("=" * 60)
        
        # Load test dataset
        test_data = self.load_test_data(test_file)
        print(f"\nLoaded {len(test_data)} test cases")
        
        # Metric 1: OCR Accuracy using Character Error Rate
        print("\n--- Metric 1: Character Error Rate ---")
        ocr_results = self.evaluate_ocr_accuracy(test_data)
        print(f"Character Accuracy: {ocr_results['char_accuracy']*100:.2f}%")
        print(f"Mean CER: {ocr_results['mean_cer']*100:.2f}%")
        
        # Metric 2: Field-level Accuracy
        print("\n--- Metric 2: Field-level Accuracy ---")
        field_results = self.evaluate_field_accuracy(test_data)
        for field, stats in field_results.items():
            print(f"{field:15s}: {stats['accuracy']*100:.1f}% "
                  f"({stats['correct']}/{stats['total']})")
        
        # Metric 3: Processing Speed
        print("\n--- Metric 3: Processing Speed ---")
        speed_results = self.evaluate_speed(test_data)
        print(f"Mean time:   {speed_results['mean_time']:.2f}s")
        print(f"Median time: {speed_results['median_time']:.2f}s")
        print(f"Throughput:  {speed_results['throughput']:.2f} docs/sec")
        
        # Combine all results
        full_results = {
            'ocr_accuracy': ocr_results,
            'field_accuracy': field_results,
            'processing_speed': speed_results,
            'test_cases': len(test_data)
        }
        
        # Save results to JSON file
        output_file = self.results_dir / 'evaluation_report.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        return full_results


# ============ Main Execution ============
if __name__ == "__main__":
    evaluator = OCREvaluator()
    results = evaluator.run_full_evaluation('test_data.json')