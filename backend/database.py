from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# TODO: Replace with your actual PostgreSQL credentials
# Format: postgresql://username:password@localhost:5432/database_name
DATABASE_URI = 'postgresql://postgres:Admin@localhost:5432/ai_data_db'

# Initialize the database engine and session
engine = create_engine(DATABASE_URI)
SessionLocal = sessionmaker(bind=engine)