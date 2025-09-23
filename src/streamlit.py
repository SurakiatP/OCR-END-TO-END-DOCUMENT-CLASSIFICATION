# src/streamlit.py - Facebook-inspired Design

import streamlit as st
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image
import time

# Add parent directory to path to import main.py
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Import our main processing pipeline
from main import process_uploaded_file, DocumentProcessor

# Facebook-inspired color palette
COLORS = {
    'primary': '#1877F2',      # Facebook Blue
    'secondary': '#42B883',    # Success Green
    'danger': '#E74C3C',       # Error Red
    'warning': '#F39C12',      # Warning Orange
    'dark': '#1C1E21',         # Facebook Dark
    'light_gray': '#F0F2F5',   # Facebook Light Gray
    'medium_gray': '#E4E6EA',  # Facebook Medium Gray
    'text_primary': '#1C1E21', # Primary Text
    'text_secondary': '#65676B', # Secondary Text
    'white': '#FFFFFF',
    'hover': '#166FE5'         # Hover Blue
}

def load_custom_css():
    """Load Facebook-inspired CSS styling"""
    st.markdown(f"""
    <style>
    /* Import Facebook font */
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {{
        background-color: {COLORS['light_gray']};
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    /* Hide default Streamlit elements */
    #MainMenu, footer, .stDeployButton {{
        display: none;
    }}
    
    /* Main container */
    .main .block-container {{
        padding: 1rem 2rem;
        max-width: 1200px;
    }}
    
    /* Header */
    .fb-header {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['hover']});
        padding: 2rem 0;
        margin: -1rem -2rem 2rem -2rem;
        text-align: center;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .fb-header h1 {{
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
    }}
    
    .fb-header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
        font-weight: 400;
    }}
    
    /* Cards */
    .fb-card {{
        background: {COLORS['white']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid {COLORS['medium_gray']};
        transition: all 0.2s ease;
    }}
    
    .fb-card:hover {{
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }}
    
    /* Upload area */
    .upload-zone {{
        background: {COLORS['white']};
        border: 2px dashed {COLORS['primary']};
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }}
    
    .upload-zone:hover {{
        border-color: {COLORS['hover']};
        background: #f8f9ff;
    }}
    
    .upload-icon {{
        font-size: 3rem;
        color: {COLORS['primary']};
        margin-bottom: 1rem;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['hover']});
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(24,119,242,0.3);
        width: 100%;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(135deg, {COLORS['hover']}, {COLORS['primary']});
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(24,119,242,0.4);
    }}
    
    /* File uploader */
    .stFileUploader > div > div {{
        background: {COLORS['white']};
        border: 2px dashed {COLORS['medium_gray']};
        border-radius: 12px;
        padding: 2rem;
    }}
    
    /* Progress bars */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']});
    }}
    
    /* Metrics */
    .fb-metric {{
        background: {COLORS['white']};
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid {COLORS['primary']};
        margin: 0.5rem 0;
    }}
    
    .fb-metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {COLORS['primary']};
        margin: 0;
    }}
    
    .fb-metric-label {{
        font-size: 0.9rem;
        color: {COLORS['text_secondary']};
        font-weight: 500;
        margin: 0.5rem 0 0 0;
    }}
    
    /* Document cards */
    .doc-card {{
        background: {COLORS['white']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid {COLORS['medium_gray']};
        position: relative;
        overflow: hidden;
    }}
    
    .doc-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['secondary']});
    }}
    
    .doc-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid {COLORS['medium_gray']};
    }}
    
    .doc-title {{
        font-size: 1.3rem;
        font-weight: 600;
        color: {COLORS['text_primary']};
        margin: 0;
    }}
    
    .doc-badge {{
        background: {COLORS['primary']};
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }}
    
    .doc-badge.receipt {{
        background: {COLORS['secondary']};
    }}
    
    .doc-info-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}
    
    .doc-info-item {{
        background: {COLORS['light_gray']};
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid {COLORS['primary']};
    }}
    
    .doc-info-label {{
        font-size: 0.8rem;
        color: {COLORS['text_secondary']};
        font-weight: 500;
        margin-bottom: 0.3rem;
    }}
    
    .doc-info-value {{
        font-size: 1rem;
        color: {COLORS['text_primary']};
        font-weight: 600;
    }}
    
    .amount-highlight {{
        font-size: 1.2rem;
        color: {COLORS['secondary']};
        font-weight: 700;
    }}
    
    /* Processing steps */
    .process-step {{
        background: {COLORS['white']};
        border: 1px solid {COLORS['medium_gray']};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }}
    
    .process-step.completed {{
        border-color: {COLORS['secondary']};
        background: #f0fff4;
    }}
    
    .process-step.error {{
        border-color: {COLORS['danger']};
        background: #fff5f5;
    }}
    
    .process-step.running {{
        border-color: {COLORS['warning']};
        background: #fffaf0;
    }}
    
    .step-icon {{
        margin-right: 1rem;
        font-size: 1.2rem;
    }}
    
    .step-text {{
        font-weight: 500;
        color: {COLORS['text_primary']};
    }}
    
    /* Success/Error messages */
    .stSuccess {{
        background: #f0fff4;
        border: 1px solid {COLORS['secondary']};
        border-radius: 8px;
    }}
    
    .stError {{
        background: #fff5f5;
        border: 1px solid {COLORS['danger']};
        border-radius: 8px;
    }}
    
    .stInfo {{
        background: #f0f9ff;
        border: 1px solid {COLORS['primary']};
        border-radius: 8px;
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {COLORS['light_gray']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['primary']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS['hover']};
    }}
    
    /* Animation */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .fade-in {{
        animation: fadeIn 0.5s ease-out;
    }}
    
    /* Responsive */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem;
        }}
        
        .fb-header {{
            margin: -1rem -1rem 2rem -1rem;
        }}
        
        .doc-info-grid {{
            grid-template-columns: 1fr;
        }}
    }}

    .documents-section {{
        border: 5px solid {COLORS['primary']};
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem 0;
    }}
    </style>
    """, unsafe_allow_html=True)

def load_json_data():
    """Load data from JSON file"""
    try:
        current_dir = Path(__file__).parent.parent.absolute()
        json_path = current_dir / "meta_data" / "meta.json"
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"invoices": [], "receipts": [], "processing_history": []}
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {"invoices": [], "receipts": [], "processing_history": []}

def display_processing_step(step_number, step_name, status="running"):
    """Display processing step with status"""
    icons = {
        "running": "⏳",
        "completed": "✅", 
        "error": "❌"
    }
    
    st.markdown(f"""
    <div class="process-step {status}">
        <div class="step-icon">{icons.get(status, "⏳")}</div>
        <div class="step-text">Step {step_number}: {step_name}</div>
    </div>
    """, unsafe_allow_html=True)

def display_summary_metrics(data):
    """Display summary metrics in Facebook style"""
    invoices = data.get('invoices', [])
    receipts = data.get('receipts', [])
    
    total_docs = len(invoices) + len(receipts)
    total_amount = sum(float(inv.get('total_amount', 0) or 0) for inv in invoices) + \
                   sum(float(rec.get('total_paid', 0) or 0) for rec in receipts)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="fb-metric">
            <div class="fb-metric-value">{total_docs}</div>
            <div class="fb-metric-label">Total Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="fb-metric">
            <div class="fb-metric-value">{len(invoices)}</div>
            <div class="fb-metric-label">Invoices</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="fb-metric">
            <div class="fb-metric-value">{len(receipts)}</div>
            <div class="fb-metric-label">Receipts</div>
        </div>
        """, unsafe_allow_html=True)
    
    # if total_amount > 0:
    #     st.markdown(f"""
    #     <div class="fb-card">
    #         <h3 style="margin: 0 0 1rem 0; color: {COLORS['text_primary']};">💰 Total Value</h3>
    #         <div style="font-size: 2rem; font-weight: 700; color: {COLORS['secondary']};">
    #             ฿{total_amount:,.2f} THB
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)

def display_document_card(doc_data, doc_type, index):
    """Display document in Facebook-style card"""
    # Get document fields based on type
    if doc_type == "invoice":
        doc_no = doc_data.get('invoice_no', 'N/A')
        doc_date = doc_data.get('invoice_date', 'N/A')
        total_amount = doc_data.get('total_amount', 0) or 0
        amount_label = "Total Amount"
    else:
        doc_no = doc_data.get('receipt_no', 'N/A')  
        doc_date = doc_data.get('receipt_date', 'N/A')
        total_amount = doc_data.get('total_paid', 0) or 0
        amount_label = "Total Paid"
    
    badge_class = "doc-badge" if doc_type == "invoice" else "doc-badge receipt"
    
    # Create a more descriptive title using document number
    document_title = f"{doc_type.title()} {doc_no}" if doc_no != 'N/A' else f"{doc_type.title()}"
    
    # st.markdown(f"""
    # <div class="doc-card fade-in">
    #     <div class="doc-header">
    #         <div class="doc-title">{document_title}</div>
    #         <div class="{badge_class}">{doc_type.upper()}</div>
    #     </div>
        
    #     <div class="doc-info-grid">
    #         <div class="doc-info-item">
    #             <div class="doc-info-label">Document No.</div>
    #             <div class="doc-info-value">{doc_no}</div>
    #         </div>
    #         <div class="doc-info-item">
    #             <div class="doc-info-label">Date</div>
    #             <div class="doc-info-value">{doc_date if doc_date != 'N/A' else 'Not specified'}</div>
    #         </div>
    #         <div class="doc-info-item">
    #             <div class="doc-info-label">Seller</div>
    #             <div class="doc-info-value">{doc_data.get('seller_name', 'N/A')}</div>
    #         </div>
    #         <div class="doc-info-item">
    #             <div class="doc-info-label">{amount_label}</div>
    #             <div class="doc-info-value amount-highlight">฿{total_amount:,.2f}</div>
    #         </div>
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)
    
    # Additional details in expander - Enhanced with comprehensive data in single view
    with st.expander("📋 View Full Details", expanded=False):
        if doc_type == "invoice":
            # Invoice comprehensive view
            st.markdown("### 📄 Invoice Details")
            
            # Financial section
            st.markdown("#### 💰 Financial Information")
            fin_col1, fin_col2, fin_col3, fin_col4 = st.columns(4)
            with fin_col1:
                st.metric("Subtotal", f"฿{doc_data.get('subtotal', 0):,.2f}" if doc_data.get('subtotal') else "N/A")
            with fin_col2:
                st.metric("VAT (7%)", f"฿{doc_data.get('vat_amount', 0):,.2f}" if doc_data.get('vat_amount') else "N/A")
            with fin_col3:
                st.metric("Total Amount", f"฿{total_amount:,.2f}")
            with fin_col4:
                st.metric("Currency", doc_data.get('currency', 'THB'))
            
            if doc_data.get('due_date'):
                st.info(f"⏰ Due Date: {doc_data.get('due_date')}")
            
            st.divider()
            
            # Company section
            st.markdown("#### 🏢 Company Information")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown("**Seller (From):**")
                st.write(f"• **Name:** {doc_data.get('seller_name', 'N/A')}")
                st.write(f"• **Tax ID:** {doc_data.get('seller_tax_id', 'N/A')}")
            with comp_col2:
                st.markdown("**Buyer (To):**")
                st.write(f"• **Name:** {doc_data.get('buyer_name', 'N/A')}")
                st.write(f"• **Tax ID:** {doc_data.get('buyer_tax_id', 'N/A')}")
            
            st.divider()
            
            # Items section
            st.markdown("#### 📦 Item Details")
            if any([doc_data.get('item_description'), doc_data.get('item_quantity'), doc_data.get('item_unit_price')]):
                item_col1, item_col2, item_col3 = st.columns(3)
                with item_col1:
                    if doc_data.get('item_description'):
                        st.markdown("**Description:**")
                        st.write(doc_data.get('item_description'))
                with item_col2:
                    if doc_data.get('item_quantity'):
                        st.metric("Quantity", doc_data.get('item_quantity'))
                with item_col3:
                    if doc_data.get('item_unit_price'):
                        st.metric("Unit Price", f"฿{doc_data.get('item_unit_price'):,.2f}")
            else:
                st.info("No item details available")
            
            st.divider()
            
            # Document info section
            st.markdown("#### 📄 Document Information")
            doc_col1, doc_col2 = st.columns(2)
            with doc_col1:
                st.write(f"• **Document Type:** {doc_data.get('document_type', 'N/A').upper()}")
                st.write(f"• **Processing ID:** {doc_data.get('processing_id', 'N/A')}")
            with doc_col2:
                if doc_data.get('processed_timestamp'):
                    formatted_time = datetime.fromisoformat(doc_data.get('processed_timestamp')).strftime('%d/%m/%Y %H:%M:%S')
                    st.write(f"• **Processed:** {formatted_time}")
                if doc_data.get('notes'):
                    st.write(f"• **Notes:** {doc_data.get('notes')}")
        
        else:  # Receipt comprehensive view
            st.markdown("### 🧾 Receipt Details")
            
            # Payment section
            st.markdown("#### 💰 Payment Information")
            pay_col1, pay_col2, pay_col3, pay_col4 = st.columns(4)
            with pay_col1:
                st.metric("Subtotal", f"฿{doc_data.get('subtotal', 0):,.2f}" if doc_data.get('subtotal') else "N/A")
            with pay_col2:
                st.metric("VAT (7%)", f"฿{doc_data.get('vat_amount', 0):,.2f}" if doc_data.get('vat_amount') else "N/A")
            with pay_col3:
                st.metric("Total Paid", f"฿{total_amount:,.2f}")
            with pay_col4:
                st.metric("Currency", doc_data.get('currency', 'THB'))
            
            st.info(f"💳 Payment Method: {doc_data.get('payment_method', 'Cash').title()}")
            
            st.divider()
            
            # Company section
            st.markdown("#### 🏢 Company Information")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown("**Seller (Store):**")
                st.write(f"• **Name:** {doc_data.get('seller_name', 'N/A')}")
                st.write(f"• **Tax ID:** {doc_data.get('seller_tax_id', 'N/A')}")
            with comp_col2:
                st.markdown("**Buyer (Customer):**")
                st.write(f"• **Name:** {doc_data.get('buyer_name', 'N/A')}")
                if doc_data.get('buyer_tax_id'):
                    st.write(f"• **Tax ID:** {doc_data.get('buyer_tax_id')}")
            
            st.divider()
            
            # Items section
            st.markdown("#### 📦 Purchase Details")
            if any([doc_data.get('item_description'), doc_data.get('item_quantity'), doc_data.get('item_unit_price')]):
                item_col1, item_col2, item_col3 = st.columns(3)
                with item_col1:
                    if doc_data.get('item_description'):
                        st.markdown("**Items:**")
                        st.write(doc_data.get('item_description'))
                with item_col2:
                    if doc_data.get('item_quantity'):
                        st.metric("Total Quantity", doc_data.get('item_quantity'))
                with item_col3:
                    if doc_data.get('item_unit_price'):
                        st.metric("Unit Price", f"฿{doc_data.get('item_unit_price'):,.2f}")
            else:
                st.info("No purchase details available")
            
            st.divider()
            
            # Document info section
            st.markdown("#### 📄 Receipt Information")
            doc_col1, doc_col2 = st.columns(2)
            with doc_col1:
                st.write(f"• **Document Type:** {doc_data.get('document_type', 'N/A').upper()}")
                st.write(f"• **Processing ID:** {doc_data.get('processing_id', 'N/A')}")
            with doc_col2:
                if doc_data.get('processed_timestamp'):
                    formatted_time = datetime.fromisoformat(doc_data.get('processed_timestamp')).strftime('%d/%m/%Y %H:%M:%S')
                    st.write(f"• **Processed:** {formatted_time}")
                if doc_data.get('acknowledgement'):
                    st.write(f"• **Acknowledgement:** {doc_data.get('acknowledgement')}")
        
        # Raw text section (at the bottom)
        if doc_data.get('raw_text'):
            st.divider()
            st.markdown("#### 📝 Raw OCR Text")
            st.text_area("Extracted text from document:", 
                       value=doc_data.get('raw_text', ''), 
                       height=150, 
                       disabled=True)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Document Processor",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'last_processed_file' not in st.session_state:
        st.session_state.last_processed_file = None
    
    # Header
    st.markdown("""
    <div class="fb-header">
        <h1>📄 Document Processing System</h1>
        <p>AI-Powered Invoice & Receipt Analysis with OCR Technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content in two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Upload section
        st.markdown("""
        <div class="fb-card">
            <h3 style="margin: 0 0 1rem 0; color: #1C1E21;">📁 Upload Document</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an invoice or receipt image",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="Supports JPG, PNG, PDF formats"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Document", use_column_width=True)
            
            # File info
            st.markdown(f"""
            <div class="fb-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #1C1E21;">📋 File Information</h4>
                <p><strong>Name:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {uploaded_file.size:,} bytes</p>
                <p><strong>Type:</strong> {uploaded_file.type}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Show processing steps
                    progress_container = st.container()
                    with progress_container:
                        display_processing_step(1, "Table Detection", "running")
                        display_processing_step(2, "OCR Text Extraction", "running") 
                        display_processing_step(3, "Data Classification", "running")
                        display_processing_step(4, "Generate Results", "running")
                    
                    # Process the document
                    results = process_uploaded_file(uploaded_file)
                    
                    # Store results in session state
                    st.session_state.processing_results = results
                    st.session_state.last_processed_file = uploaded_file.name
                    
                    # Update progress display
                    with progress_container:
                        if results['success']:
                            display_processing_step(1, "Table Detection", "completed")
                            display_processing_step(2, "OCR Text Extraction", "completed")
                            display_processing_step(3, "Data Classification", "completed") 
                            display_processing_step(4, "Generate Results", "completed")
                        else:
                            steps = results['steps_completed']
                            all_steps = ["table_separation", "text_detection", "text_classification", "generate_results"]
                            step_names = ["Table Detection", "OCR Text Extraction", "Data Classification", "Generate Results"]
                            
                            for i, (step, name) in enumerate(zip(all_steps, step_names), 1):
                                if step in steps:
                                    display_processing_step(i, name, "completed")
                                else:
                                    display_processing_step(i, name, "error")
                                    break
                    
                    # Show results
                    if results['success']:
                        st.success("Document processed successfully!")
                        st.balloons()
                        st.rerun()  # Refresh to show new data
                    else:
                        st.error("Processing failed: " + ", ".join(results['errors']))
        
        # Last processing result
        if st.session_state.processing_results:
            results = st.session_state.processing_results
            status_icon = "✅" if results['success'] else "❌"
            status_text = "Success" if results['success'] else "Failed"
            
            st.markdown(f"""
            <div class="fb-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #1C1E21;">📊 Last Processing</h4>
                <p><strong>File:</strong> {st.session_state.last_processed_file}</p>
                <p><strong>Status:</strong> {status_icon} {status_text}</p>
                <p><strong>Steps:</strong> {len(results['steps_completed'])}/4</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Load and display data
        data = load_json_data()
        
        # Display summary metrics
        display_summary_metrics(data)
        
        # Display documents
        st.markdown("""
        <div class="documents-section">
            <h3 style="margin: 0 0 1rem 0; color: #1C1E21;">📑 Processed Documents</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Combine all documents
        all_docs = []
        for inv in data.get('invoices', []):
            all_docs.append((inv, 'invoice', inv.get('processed_timestamp', '')))
        for rec in data.get('receipts', []):
            all_docs.append((rec, 'receipt', rec.get('processed_timestamp', '')))
        
        if all_docs:
            # Sort by timestamp (newest first)
            all_docs.sort(key=lambda x: x[2], reverse=True)
            
            # Display documents
            for i, (doc_data, doc_type, timestamp) in enumerate(all_docs, 1):
                display_document_card(doc_data, doc_type, i)

        # if all_docs:
        #     st.info(f"Found {len(all_docs)} processed documents")

        else:
            st.markdown("""
            <div class="fb-card" style="text-align: center; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
                <h3 style="color: #65676B; margin: 0 0 1rem 0;">No documents processed yet</h3>
                <p style="color: #65676B; margin: 0;">Upload an invoice or receipt to get started!</p>
            </div>
            """, unsafe_allow_html=True)
    
    # System information footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="fb-card" style="text-align: center; margin-top: 2rem;">
        <p style="color: #65676B; margin: 0; font-size: 0.9rem;">
            Document Processing System | Powered by EasyOCR + Computer Vision | Made with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()