# src/text_classifier.py

import re
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class TextDocumentProcessor:
    """
    Process text files to classify documents and extract key information
    Updated for production use with proper path handling
    """
    
    def __init__(self):
        # Get project root from current file location
        current_file = Path(__file__).absolute()
        project_root = current_file.parent.parent  # Go up from src/ to project root
        
        # Set paths relative to project root
        self.json_storage_path = project_root / "meta_data" / "meta.json"
        self.invoice_csv_path = project_root / "csv_classifier" / "invoice_ocr.csv" 
        self.receipt_csv_path = project_root / "csv_classifier" / "receipt_ocr.csv"
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize JSON and CSV storage files"""
        # Create directories
        self.json_storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.invoice_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize JSON file (always recreate if corrupted)
        try:
            if self.json_storage_path.exists():
                with open(self.json_storage_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # Test if file is valid JSON
        except (json.JSONDecodeError, FileNotFoundError):
            # Create/recreate if corrupted or missing
            initial_data = {
                "invoices": [],
                "receipts": [],
                "processing_history": []
            }
            with open(self.json_storage_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
            print(f"Initialized JSON file: {self.json_storage_path}")
        
        # Initialize CSV files with proper headers
        invoice_columns = [
            'invoice_no', 'invoice_date', 'due_date', 'seller_name', 'seller_tax_id',
            'buyer_name', 'buyer_tax_id', 'item_description', 'item_quantity', 
            'item_unit_price', 'subtotal', 'vat_amount', 'total_amount', 'currency',
            'notes', 'processed_timestamp'
        ]
        
        receipt_columns = [
            'receipt_no', 'receipt_date', 'seller_name', 'seller_tax_id', 'buyer_name',
            'payment_method', 'item_description', 'item_quantity', 'item_unit_price',
            'subtotal', 'vat_amount', 'total_paid', 'currency', 'acknowledgement',
            'processed_timestamp'
        ]
        
        # Force recreate CSV files to ensure proper headers
        pd.DataFrame(columns=invoice_columns).to_csv(self.invoice_csv_path, index=False, encoding='utf-8')
        pd.DataFrame(columns=receipt_columns).to_csv(self.receipt_csv_path, index=False, encoding='utf-8')
        print(f"Initialized CSV files: {self.invoice_csv_path}, {self.receipt_csv_path}")
    
    def classify_document(self, text: str) -> str:
        """
        Classify document as invoice or receipt based on text content
        
        Args:
            text (str): Full text content of document
            
        Returns:
            str: 'invoice' or 'receipt'
        """
        text_lower = text.lower()
        
        invoice_score = 0
        receipt_score = 0
        
        # Enhanced keyword scoring
        invoice_keywords = [
            'ใบแจ้งหนี้', 'invoice', 'วันครบกำหนด', 'due date', 
            'ผู้ซื้อ', 'ผู้รับในแจ้งหนี้', 'bl', 'inv', 'quotation',
            'ใบเสนอราคา', 'proposal', 'estimate'
        ]
        receipt_keywords = [
            'ใบเสร็จ', 'receipt', 'pos', 'ขอบคุณ', 'thank you',
            'ได้รับเงิน', 'ชำระเงิน', 'paid', 'payment received',
            'cash', 'credit card', 'transfer'
        ]
        
        # Count keyword occurrences
        for keyword in invoice_keywords:
            if keyword in text_lower:
                invoice_score += 2
                
        for keyword in receipt_keywords:
            if keyword in text_lower:
                receipt_score += 2
        
        # Check for document number patterns
        if re.search(r'(?:inv|bl|qt)[\-\s]*\d+', text_lower):
            invoice_score += 3
        if re.search(r'(?:rc|pos|rec)[\-\s]*\d+', text_lower):
            receipt_score += 3
        
        # Check for payment confirmation phrases
        payment_phrases = ['ได้รับเงินแล้ว', 'payment completed', 'paid in full']
        for phrase in payment_phrases:
            if phrase in text_lower:
                receipt_score += 3
        
        # Check for due date (common in invoices)
        if 'due' in text_lower or 'ครบกำหนด' in text_lower:
            invoice_score += 2
        
        print(f"Classification scores - Invoice: {invoice_score}, Receipt: {receipt_score}")
        
        return 'invoice' if invoice_score > receipt_score else 'receipt'
    
    def extract_document_numbers(self, text: str) -> str:
        """Extract document numbers from text"""
        patterns = [
            r'เลขที่\s*:?\s*([A-Z0-9\-]+)',
            r'(?:invoice|inv|bl|qt|rc|pos|receipt)\s*(?:no\.?|#)?\s*:?\s*([A-Z0-9\-]+)',
            r'(?:BL|INV|QT|RC|POS)(\d+)',
            r'(?:R-|RC)(\d{8}-\d{3})',
            r'หมายเลข\s*:?\s*([A-Z0-9\-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def extract_dates(self, text: str) -> Dict[str, str]:
        """Extract dates from text"""
        result = {'document_date': None, 'due_date': None}
        
        thai_months = {
            'มกราคม': '01', 'ม.ค.': '01', 'jan': '01', 'january': '01',
            'กุมภาพันธ์': '02', 'ก.พ.': '02', 'feb': '02', 'february': '02',
            'มีนาคม': '03', 'มี.ค.': '03', 'mar': '03', 'march': '03',
            'เมษายน': '04', 'เม.ย.': '04', 'apr': '04', 'april': '04',
            'พฤษภาคม': '05', 'พ.ค.': '05', 'may': '05',
            'มิถุนายน': '06', 'มิ.ย.': '06', 'jun': '06', 'june': '06',
            'กรกฎาคม': '07', 'ก.ค.': '07', 'jul': '07', 'july': '07',
            'สิงหาคม': '08', 'ส.ค.': '08', 'aug': '08', 'august': '08',
            'กันยายน': '09', 'ก.ย.': '09', 'sep': '09', 'september': '09',
            'ตุลาคม': '10', 'ต.ค.': '10', 'oct': '10', 'october': '10',
            'พฤศจิกายน': '11', 'พ.ย.': '11', 'nov': '11', 'november': '11',
            'ธันวาคม': '12', 'ธ.ค.': '12', 'dec': '12', 'december': '12'
        }
        
        date_patterns = [
            r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})',
            r'(\d{1,2})\s+(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)\s+(\d{4})',
            r'(\d{1,2})\s+(มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\.)(\d{4})',
            r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
            r'(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)(\d{4})',
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    groups = match.groups()
                    day = groups[0]
                    month = groups[1]
                    year = groups[2]
                    
                    # Convert Thai month if needed
                    if month.lower() in thai_months:
                        month = thai_months[month.lower()]
                    elif not month.isdigit():
                        # Try to convert English month names
                        month_lower = month.lower()
                        if month_lower in thai_months:
                            month = thai_months[month_lower]
                        else:
                            continue
                    
                    # Convert Buddhist year to Christian year
                    year = int(year)
                    if year > 2500:
                        year -= 543
                    
                    formatted_date = f"{int(day):02d}/{int(month):02d}/{year}"
                    
                    # Determine context
                    start_pos = max(0, match.start() - 50)
                    end_pos = min(len(text), match.end() + 50)
                    context = text[start_pos:end_pos].lower()
                    
                    if any(word in context for word in ['ครบกำหนด', 'due', 'payment due']):
                        result['due_date'] = formatted_date
                    else:
                        if not result['document_date']:
                            result['document_date'] = formatted_date
                        elif not result['due_date']:
                            result['due_date'] = formatted_date
                    
                except (ValueError, IndexError):
                    continue
        
        return result
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """Extract company names and tax IDs"""
        result = {
            'seller_name': None, 
            'seller_tax_id': None,
            'buyer_name': None,
            'buyer_tax_id': None
        }
        
        # Extract tax IDs (13 digits)
        tax_ids = re.findall(r'(\d{13})', text)
        
        # Extract company names
        company_patterns = [
            r'บริษัท\s+([^จำกัด\n]+(?:จำกัด)?(?:\s*\(มหาชน\))?)',
            r'ห้างหุ้นส่วน\s+([^\n]+)',
            r'([A-Z][a-zA-Z\s&]+(?:Co\.|Ltd\.|Inc\.|Corp\.|Company|Limited))',
            r'ร้าน\s*([^\n]+)',
            r'([ก-๙\s]+(?:จำกัด|มหาชน))'
        ]
        
        company_names = []
        for pattern in company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_name = match.strip()
                if len(clean_name) > 3:  # Filter out very short matches
                    company_names.append(clean_name)
        
        # Assign first found values
        if tax_ids:
            result['seller_tax_id'] = tax_ids[0]
            if len(tax_ids) > 1:
                result['buyer_tax_id'] = tax_ids[1]
                
        if company_names:
            result['seller_name'] = company_names[0]
            if len(company_names) > 1:
                result['buyer_name'] = company_names[1]
        
        return result
    
    def extract_amounts(self, text: str) -> Dict[str, float]:
        """Extract monetary amounts with improved parsing"""
        result = {'subtotal': None, 'vat_amount': None, 'total_amount': None}
        
        # Specific amount patterns with context - more precise patterns
        patterns = [
            (r'รวมเป็นเงิน\s*\(ไม่รวม\s*vat\)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?:\s*บาท)?', 'subtotal'),
            (r'ภาษีมูลค่าเพิ่ม.*?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?:\s*บาท)?', 'vat_amount'),
            (r'ภาษีมูลค่าเพิ่ม\s*(?:7%|7\s*%)?[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', 'vat'),
            (r'ยอดรวมสุทธิ\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)(?:\s*บาท)?', 'total_amount'),
        ]
        
        for pattern, amount_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    amount_str = match.group(1).replace(',', '')
                    amount = float(amount_str)
                    result[amount_type] = amount
                    break  # Take first match for each type
                except ValueError:
                    continue
        
        # Calculate missing VAT if we have subtotal
        if result['subtotal'] and (not result['vat_amount'] or result['vat_amount'] < 100):
            result['vat_amount'] = round(result['subtotal'] * 0.07, 2)
        
        return result
    
    def extract_items_info(self, text: str) -> Dict[str, any]:
        """Extract item information from text - improved version with better price parsing"""
        result = {
            'item_description': None,
            'item_quantity': None,
            'item_unit_price': None
        }
        
        # Look for column format data (extracted from tables)
        column_pattern = r'([^:]+):\s*([^:\n]+)'
        columns = re.findall(column_pattern, text)
        
        descriptions = []
        quantities = []
        prices = []
        
        for column_name, column_values in columns:
            column_name = column_name.strip()
            
            # Check if this is a description column
            if any(keyword in column_name.lower() for keyword in ['รายการ', 'item', 'description', 'service', 'บริการ', 'รายละเอียด']):
                # Clean up the values
                items = [item.strip() for item in column_values.split(',') if item.strip()]
                # Filter out non-descriptive items
                valid_items = []
                for item in items:
                    # Skip items that are mostly numbers or too short - FIXED SYNTAX ERROR
                    if len(item) > 5 and not re.match(r'^\d+$', item) and 'เลข' not in item:
                        valid_items.append(item)
                descriptions.extend(valid_items)
            
            # Check if this is a quantity column
            elif any(keyword in column_name.lower() for keyword in ['จำนวน', 'quantity', 'qty']):
                items = column_values.split(',')
                for item in items:
                    item = item.strip()
                    # Extract numbers that look like quantities (typically small integers)
                    numbers = re.findall(r'\b(\d+)\b', item)
                    for num in numbers:
                        if 1 <= int(num) <= 1000:  # Reasonable quantity range
                            quantities.append(int(num))
            
            # Check if this is a unit price column - more specific pattern
            elif any(keyword in column_name.lower() for keyword in ['ราคาต่อหน่วย', 'unit price', 'หน่วยละ']):
                items = column_values.split(',')
                for item in items:
                    item = item.strip()
                    # Extract price numbers with proper formatting
                    numbers = re.findall(r'(\d+(?:\.\d{2})?)', item)
                    for num in numbers:
                        try:
                            price = float(num)
                            if price > 0:
                                prices.append(price)
                        except ValueError:
                            pass
            
            # Check if this is a total price column
            elif any(keyword in column_name.lower() for keyword in ['ราคารวม', 'total', 'amount']):
                # Don't use total prices for unit price calculation
                pass
        
        # Assign results
        if descriptions:
            # Take only meaningful descriptions (not system text)
            meaningful_descriptions = []
            for desc in descriptions:
                # Filter out common OCR artifacts and system text
                if not any(skip_word in desc.lower() for skip_word in [
                    'เลข', 'วันที่', 'ครบกำหนด', 'อ้างอิง', 'เบอร์โทร', 'บาท', 
                    'vat', 'ภาษี', 'รวม', 'ยอด', 'เอกสาร'
                ]) and len(desc.strip()) > 8:
                    meaningful_descriptions.append(desc.strip())
            
            if meaningful_descriptions:
                result['item_description'] = ', '.join(meaningful_descriptions[:3])  # Limit to 3 items
        
        if quantities:
            result['item_quantity'] = sum(quantities)
        
        if prices:
            if len(prices) == 1:
                # only one price
                result['item_unit_price'] = prices[0]
            else:
                # multiprices - not define unit price
                result['item_unit_price'] = None
        
        return result
    
    def extract_comprehensive_fields(self, text: str) -> Dict:
        """Extract all document fields from text"""
        fields = {
            'document_no': None,
            'document_date': None,
            'due_date': None,
            'seller_name': None,
            'seller_tax_id': None,
            'buyer_name': None,
            'buyer_tax_id': None,
            'subtotal': None,
            'vat_amount': None,
            'total_amount': None,
            'item_description': None,
            'item_quantity': None,
            'item_unit_price': None
        }
        
        # Extract document number
        fields['document_no'] = self.extract_document_numbers(text)
        
        # Extract dates
        dates = self.extract_dates(text)
        fields['document_date'] = dates['document_date']
        fields['due_date'] = dates['due_date']
        
        # Extract entities
        entities = self.extract_entities(text)
        fields.update(entities)
        
        # Extract amounts
        amounts = self.extract_amounts(text)
        fields.update(amounts)
        
        # Extract items
        items = self.extract_items_info(text)
        fields.update(items)
        
        return fields
    
    def save_to_json(self, doc_data: Dict, doc_type: str):
        """Save document data to JSON file with error handling - OVERWRITE mode"""
        try:
            # Create new data structure with only the current document
            data = {
                "invoices": [],
                "receipts": [],
                "processing_history": []
            }
            
            # Add timestamp and ID
            doc_data['processed_timestamp'] = datetime.now().isoformat()
            doc_data['processing_id'] = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Add to appropriate section
            if doc_type == 'invoice':
                data['invoices'].append(doc_data)
            else:
                data['receipts'].append(doc_data)
            
            # Add to processing history
            data['processing_history'].append({
                'timestamp': doc_data['processed_timestamp'],
                'type': doc_type,
                'processing_id': doc_data['processing_id']
            })
            
            # OVERWRITE the JSON file completely
            with open(self.json_storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            print(f"Overwritten JSON: {doc_type} - {doc_data['processing_id']}")
            
        except Exception as e:
            print(f"Error saving to JSON: {e}")
            # Try to create a backup
            try:
                backup_data = {doc_type: [doc_data], "processing_history": []}
                backup_path = str(self.json_storage_path).replace('.json', '_backup.json')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2)
                print(f"Created backup at: {backup_path}")
            except:
                print("Could not create backup file")
    
    def append_to_csv(self, doc_data: Dict, doc_type: str):
        """Append document data to CSV file"""
        try:
            csv_path = self.invoice_csv_path if doc_type == 'invoice' else self.receipt_csv_path
            
            # Read existing CSV
            if csv_path.exists():
                df = pd.read_csv(csv_path, encoding='utf-8')
            else:
                df = pd.DataFrame()
            
            # Create new row
            new_row = pd.DataFrame([doc_data])
            
            # Append to existing data
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save back to CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            print(f"Appended to CSV: {csv_path}")
            
        except Exception as e:
            print(f"Error appending to CSV: {e}")
    
    def process_text_file(self, text_file_path: str):
        """
        Process a text file to extract document information
        
        Args:
            text_file_path (str): Path to the text file
        """
        print(f"Processing text file: {text_file_path}")
        
        try:
            # Read text file
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            if not text_content.strip():
                print("Text file is empty")
                return
            
            # Classify document type
            doc_type = self.classify_document(text_content)
            print(f"Document classified as: {doc_type.upper()}")
            
            # Extract comprehensive fields
            extracted_fields = self.extract_comprehensive_fields(text_content)
            
            # Prepare structured data
            structured_data = {
                'source_file': text_file_path,
                'document_type': doc_type,
                'processed_timestamp': datetime.now().isoformat(),
                'raw_text': text_content
            }
            
            # Add document-specific fields
            if doc_type == 'invoice':
                structured_data.update({
                    'invoice_no': extracted_fields.get('document_no'),
                    'invoice_date': extracted_fields.get('document_date'),
                    'due_date': extracted_fields.get('due_date'),
                    'seller_name': extracted_fields.get('seller_name'),
                    'seller_tax_id': extracted_fields.get('seller_tax_id'),
                    'buyer_name': extracted_fields.get('buyer_name'),
                    'buyer_tax_id': extracted_fields.get('buyer_tax_id'),
                    'item_description': extracted_fields.get('item_description'),
                    'item_quantity': extracted_fields.get('item_quantity'),
                    'item_unit_price': extracted_fields.get('item_unit_price'),
                    'subtotal': extracted_fields.get('subtotal'),
                    'vat_amount': extracted_fields.get('vat_amount'),
                    'total_amount': extracted_fields.get('total_amount'),
                    'currency': 'THB',
                    'notes': None
                })
            else:  # receipt
                structured_data.update({
                    'receipt_no': extracted_fields.get('document_no'),
                    'receipt_date': extracted_fields.get('document_date'),
                    'seller_name': extracted_fields.get('seller_name'),
                    'seller_tax_id': extracted_fields.get('seller_tax_id'),
                    'buyer_name': extracted_fields.get('buyer_name'),
                    'payment_method': 'cash',  # Default
                    'item_description': extracted_fields.get('item_description'),
                    'item_quantity': extracted_fields.get('item_quantity'),
                    'item_unit_price': extracted_fields.get('item_unit_price'),
                    'subtotal': extracted_fields.get('subtotal'),
                    'vat_amount': extracted_fields.get('vat_amount'),
                    'total_paid': extracted_fields.get('total_amount'),
                    'currency': 'THB',
                    'acknowledgement': None
                })
            
            # Display extracted information
            print(f"\nExtracted Information:")
            print(f"   Document No: {structured_data.get('invoice_no' if doc_type == 'invoice' else 'receipt_no')}")
            print(f"   Date: {structured_data.get('invoice_date' if doc_type == 'invoice' else 'receipt_date')}")
            print(f"   Seller: {structured_data.get('seller_name')}")
            print(f"   Total: {structured_data.get('total_amount' if doc_type == 'invoice' else 'total_paid')}")
            print(f"   Items: {structured_data.get('item_description', 'None')}")
            
            # Save to JSON
            self.save_to_json(structured_data, doc_type)
            
            # Save to CSV
            self.append_to_csv(structured_data, doc_type)
            
            print(f"Document processed successfully")
            
        except Exception as e:
            print(f"Error processing text file: {e}")
            import traceback
            traceback.print_exc()
    
    def display_summary(self):
        """Display processing summary with error handling"""
        try:
            # Try to load JSON data
            try:
                with open(self.json_storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print("JSON file corrupted or missing, cannot display summary")
                return
            
            invoices = data.get('invoices', [])
            receipts = data.get('receipts', [])
            
            print(f"\n{'=' * 60}")
            print("TEXT DOCUMENT PROCESSING SUMMARY")
            print(f"{'=' * 60}")
            print(f"Total Documents Processed: {len(invoices) + len(receipts)}")
            print(f"Invoices: {len(invoices)}")
            print(f"Receipts: {len(receipts)}")
            
            if invoices:
                print(f"\nLatest Invoice:")
                latest = invoices[-1]
                for key, value in latest.items():
                    if value and key not in ['raw_text', 'processing_id']:
                        print(f"  {key}: {value}")
            
            if receipts:
                print(f"\nLatest Receipt:")
                latest = receipts[-1]
                for key, value in latest.items():
                    if value and key not in ['raw_text', 'processing_id']:
                        print(f"  {key}: {value}")
            
            print(f"\nOutput Files:")
            print(f"  JSON: {self.json_storage_path}")
            print(f"  Invoice CSV: {self.invoice_csv_path}")
            print(f"  Receipt CSV: {self.receipt_csv_path}")
            
        except Exception as e:
            print(f"Error displaying summary: {e}")
            print("Attempting to show file status...")
            for path in [self.json_storage_path, self.invoice_csv_path, self.receipt_csv_path]:
                if path.exists():
                    size = path.stat().st_size
                    print(f"  {path}: {size} bytes")
                else:
                    print(f"  {path}: Not found")


def process_document_from_text(text_file_path: str = None):
    """
    Main function to process document from text file
    Updated for production use with proper path handling
    
    Args:
        text_file_path (str): Path to text file (if None, uses default)
    """
    if text_file_path is None:
        # Get default path from project structure
        current_file = Path(__file__).absolute()
        project_root = current_file.parent.parent  # Go up from src/ to project root
        text_file_path = project_root / "temp" / "detected" / "doc_content.txt"
    
    print("TEXT DOCUMENT PROCESSOR")
    print("=" * 50)
    
    if not Path(text_file_path).exists():
        print(f"Text file not found: {text_file_path}")
        return
    
    processor = TextDocumentProcessor()
    processor.process_text_file(str(text_file_path))
    processor.display_summary()


# Usage example
if __name__ == "__main__":
    # Process the default text file
    process_document_from_text()
    
    # Or process a specific file
    # process_document_from_text("path/to/your/text/file.txt")