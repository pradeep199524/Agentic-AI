import os
import json
import logging
import re
import pandas as pd
import pdfplumber

from backend.database import engine, SessionLocal
from backend.models import Base, Document, Page

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
Base.metadata.create_all(engine)

def clean_table(table):
    if not table: return []
    cleaned = []
    for row in table:
        clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(cell for cell in clean_row):
            cleaned.append(clean_row)
    return cleaned

def filter_table_chars(obj, table_bboxes):
    if obj.get("object_type") != "char": return True
    x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
    margin = 3.0 
    for b_x0, b_top, b_x1, b_bottom in table_bboxes:
        if not (x1 < (b_x0 - margin) or x0 > (b_x1 + margin) or bottom < (b_top - margin) or top > (b_bottom + margin)):
            return False 
    return True 

# FIXED: Added doc_id=None to the function signature
def process_pdf(file_path, session, doc_id=None):
    filename = os.path.basename(file_path)
    clean_filename = re.sub(r'^\d+_', '', filename)
    
    try:
        if doc_id is None:
            existing_doc = session.query(Document).filter_by(filename=clean_filename).first()
            if existing_doc:
                session.query(Page).filter_by(document_id=existing_doc.id).delete()
                session.delete(existing_doc)
                session.flush()
            
            doc = Document(filename=clean_filename, file_type='pdf', status='completed')
            session.add(doc)
            session.flush()
            current_doc_id = doc.id
        else:
            current_doc_id = doc_id

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.find_tables()
                raw_tables = [t.extract() for t in tables]
                table_bboxes = [t.bbox for t in tables] 
                cleaned_tables = [clean_table(t) for t in raw_tables if t]
                
                filtered_page = page.filter(lambda obj: filter_table_chars(obj, table_bboxes))
                text = filtered_page.extract_text()
                
                if not text and not cleaned_tables: continue
                
                final_text = ""
                if text:
                    text = text.replace("(cid:127)", "•")
                    lines = [line for line in text.split('\n') if not re.match(r'^Page \d+$', line.strip(), re.IGNORECASE)]
                    final_text = '\n'.join(lines).strip()
                
                page_json = {
                    "text": final_text,
                    "tables": cleaned_tables,
                    "character_count": len(final_text)
                }
                
                # Link to the correct current_doc_id
                page_record = Page(document_id=current_doc_id, page_number=page_num, content=page_json)
                session.add(page_record)
                
        session.commit()
        logging.info(f"Successfully processed PDF: {filename}")
        
    except Exception as e:
        session.rollback() 
        logging.error(f"Failed to process PDF {filename}: {e}")
        raise e

# FIXED: Added filename and doc_id=None to the function signature
def process_csv(file_path, filename, session, db_engine, doc_id=None):
    clean_filename = re.sub(r'^\d+_', '', filename)
    table_name = f"csv_data_{clean_filename.split('.')[0].lower()}" 
    
    try:
        if doc_id is None:
            existing_doc = session.query(Document).filter_by(filename=clean_filename).first()
            if existing_doc:
                session.query(Page).filter_by(document_id=existing_doc.id).delete()
                session.delete(existing_doc)
                session.flush()
            
            doc = Document(filename=clean_filename, file_type='csv', status='completed')
            session.add(doc)
            session.flush()
            current_doc_id = doc.id
        else:
            current_doc_id = doc_id

        df = pd.read_csv(file_path)
        df.dropna(how='all', inplace=True)
        df.fillna("UNKNOWN", inplace=True) 
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df.to_sql(table_name, db_engine, if_exists='replace', index=False)
        
        session.commit()
        logging.info(f"Successfully processed CSV: {filename}")
    except Exception as e:
        session.rollback()
        logging.error(f"Failed to process CSV {filename}: {e}")
        raise e

if __name__ == "__main__":
    db_session = SessionLocal()
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if not os.path.exists(data_folder):
        data_folder = 'data'
        
    if os.path.exists(data_folder):
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            if os.path.isfile(file_path):
                if filename.lower().endswith('.pdf'):
                    process_pdf(file_path, db_session)
                elif filename.lower().endswith('.csv'):
                    process_csv(file_path, filename, db_session, engine)
    db_session.close()