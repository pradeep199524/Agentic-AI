from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Document(Base):
    """Tracks every file (PDF or CSV) uploaded to the system."""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False, unique=True)
    file_type = Column(String, nullable=False)  # 'pdf' or 'csv'
    status = Column(String, default="processing", nullable=False)  # 'processing', 'completed', 'failed'
    upload_date = Column(DateTime, default=datetime.utcnow)
    
    # Links to the Page table. If a document is deleted, its pages are deleted too.
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")


class Page(Base):
    """Stores the content of each PDF page in a structured JSON format."""
    __tablename__ = 'pages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    page_number = Column(Integer, nullable=False)
    
    # Storing paragraphs and metadata as a structured JSON object
    content = Column(JSON, nullable=False) 
    
    # Link back to the parent document
    document = relationship("Document", back_populates="pages")


class GlossaryTerm(Base):
    """Stores dynamic acronyms and expansions without hardcoding."""
    __tablename__ = "glossary_terms"
    
    id = Column(Integer, primary_key=True, index=True)
    term = Column(String, unique=True, index=True)
    expansion = Column(String)