import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import ast # Use ast for safe evaluation of string literals

# --- Configuration ---
load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

DB_NAME = "clinical_trials_db"
COLLECTION_NAME = "trials_collection"
INDEX_NAME = "vector_index"
# --- UPDATED FILE PATH AND FORMAT ---
DATA_FILE_PATH = "oncology_trials.csv"

if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in the environment variables.")

# --- Helper function to safely parse list-like strings ---
def safe_literal_eval(val):
    """Safely evaluates a string that looks like a list."""
    if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return [] # Return empty list on parsing error
    return [] # Return empty list if not a string list

# --- Helper function to create text chunks ---
def create_document_summary(row):
    """Creates a human-readable summary for a single trial."""
    indications = safe_literal_eval(row['indications'])
    indications_str = ", ".join(indications) if indications else "Not specified"
    
    drugs = safe_literal_eval(row['interventions_drugs'])
    drugs_str = ", ".join(drugs) if drugs else "Not specified"

    return (
        f"Clinical trial with NCT ID {row['nct_id']} is a {row['phase']} study titled '{row['brief_title']}'.\n"
        f"Status: The trial's recruitment status is '{row['recruitment_status']}'.\n"
        f"Details: It plans to enroll {row['enrollment']} patients, starting around {row['start_date']}.\n"
        f"Indication(s): This trial is for patients with {indications_str}.\n"
        f"Intervention(s): The drugs being investigated are: {drugs_str}."
    )

def ingest_data():
    """Loads data from a CSV, creates summaries, generates embeddings, and stores in MongoDB."""
    print("Loading data from CSV file...")
    try:
        # --- UPDATED DATA LOADING ---
        # Reads a comma-separated file by default.
        df = pd.read_csv(DATA_FILE_PATH) 
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print(f"Error: The data file '{DATA_FILE_PATH}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    df.fillna("Not specified", inplace=True)
    
    print(f"Loaded {len(df)} trials. Creating document summaries...")
    documents = df.apply(create_document_summary, axis=1).tolist()
    
    metadatas = df.to_dict('records')

    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Connecting to MongoDB Atlas and ingesting data...")
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    
    collection.delete_many({})
    print(f"Cleared existing documents in '{COLLECTION_NAME}'.")
    
    vector_store = MongoDBAtlasVectorSearch.from_texts(
        texts=documents,
        embedding=embeddings,
        collection=collection,
        metadatas=metadatas,
        index_name=INDEX_NAME
    )
    
    print(f"Successfully ingested {len(documents)} documents into MongoDB Atlas.")
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()