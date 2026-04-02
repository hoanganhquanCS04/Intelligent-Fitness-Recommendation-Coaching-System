from sqlalchemy import create_engine, delcarative_base
from sqlalchemy.orm import sessionmaker
import os

engine = create_engine(os.getenv("db_connection_string"))

Session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Model = delcarative_base()