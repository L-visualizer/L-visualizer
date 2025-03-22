# NLP-Based Credential Search & Verification System

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import sqlite3
from datetime import datetime
import json

class CredentialDatabase:
    """Database handler for credentials"""
    
    def __init__(self, db_path: str = "credentials.db"):
        """Initialize the credential database"""
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
        
    def _create_tables(self):
        """Create the necessary tables if they don't exist"""
        # Credentials table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS credentials (
            id INTEGER PRIMARY KEY,
            holder_id INTEGER,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            issuer TEXT NOT NULL,
            issue_date TEXT NOT NULL,
            expiry_date TEXT,
            verification_level INTEGER DEFAULT 0,
            verification_source TEXT,
            verification_date TEXT,
            embedding_id INTEGER,
            FOREIGN KEY (holder_id) REFERENCES users(id),
            FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
        )
        ''')
        
        # Users table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            profile TEXT
        )
        ''')
        
        # Embeddings table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        ''')
        
        self.conn.commit()
    
    def add_credential(self, holder_id: int, title: str, description: str, 
                       issuer: str, issue_date: str, expiry_date: Optional[str] = None,
                       verification_level: int = 0, verification_source: Optional[str] = None,
                       verification_date: Optional[str] = None) -> int:
        """Add a new credential to the database"""
        self.cursor.execute('''
        INSERT INTO credentials 
        (holder_id, title, description, issuer, issue_date, expiry_date, 
         verification_level, verification_source, verification_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (holder_id, title, description, issuer, issue_date, expiry_date,
              verification_level, verification_source, verification_date))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def update_embedding(self, credential_id: int, embedding_vector: np.ndarray) -> int:
        """Store embedding vector for a credential"""
        vector_bytes = embedding_vector.tobytes()
        created_at = datetime.now().isoformat()
        
        self.cursor.execute('''
        INSERT INTO embeddings (vector, created_at)
        VALUES (?, ?)
        ''', (vector_bytes, created_at))
        
        embedding_id = self.cursor.lastrowid
        
        self.cursor.execute('''
        UPDATE credentials
        SET embedding_id = ?
        WHERE id = ?
        ''', (embedding_id, credential_id))
        
        self.conn.commit()
        return embedding_id
    
    def get_credential(self, credential_id: int) -> Dict:
        """Retrieve a credential by ID"""
        self.cursor.execute('''
        SELECT c.*, u.name as holder_name
        FROM credentials c
        JOIN users u ON c.holder_id = u.id
        WHERE c.id = ?
        ''', (credential_id,))
        
        columns = [col[0] for col in self.cursor.description]
        result = self.cursor.fetchone()
        
        if result:
            return dict(zip(columns, result))
        return None
    
    def get_embedding(self, embedding_id: int) -> np.ndarray:
        """Retrieve embedding vector by ID"""
        self.cursor.execute('''
        SELECT vector FROM embeddings
        WHERE id = ?
        ''', (embedding_id,))
        
        result = self.cursor.fetchone()
        if result:
            return np.frombuffer(result[0], dtype=np.float32)
        return None
    
    def get_all_credential_embeddings(self) -> List[Tuple[int, np.ndarray]]:
        """Retrieve all credential IDs and their embeddings"""
        self.cursor.execute('''
        SELECT c.id, e.vector
        FROM credentials c
        JOIN embeddings e ON c.embedding_id = e.id
        ''')
        
        results = []
        for row in self.cursor.fetchall():
            credential_id = row[0]
            embedding = np.frombuffer(row[1], dtype=np.float32)
            results.append((credential_id, embedding))
        
        return results
    
    def verify_credential(self, credential_id: int, verification_level: int,
                         verification_source: str) -> bool:
        """Update verification status of a credential"""
        verification_date = datetime.now().isoformat()
        
        self.cursor.execute('''
        UPDATE credentials
        SET verification_level = ?,
            verification_source = ?,
            verification_date = ?
        WHERE id = ?
        ''', (verification_level, verification_source, verification_date, credential_id))
        
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def close(self):
        """Close the database connection"""
        self.conn.close()


class NLPCredentialSearch:
    """NLP-based credential search engine using sentence embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the search engine with a sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.db = CredentialDatabase()
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the sentence transformer"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def process_new_credential(self, credential_id: int) -> bool:
        """Process a newly added credential to generate and store its embedding"""
        credential = self.db.get_credential(credential_id)
        if not credential:
            return False
        
        # Combine title and description for better semantic representation
        text = f"{credential['title']}. {credential['description']}"
        embedding = self.generate_embedding(text)
        
        # Store the embedding
        self.db.update_embedding(credential_id, embedding)
        return True
    
    def search_credentials(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for credentials using natural language query"""
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Get all credential embeddings
        all_credentials = self.db.get_all_credential_embeddings()
        
        if not all_credentials:
            return []
        
        # Calculate similarities
        credential_ids = [item[0] for item in all_credentials]
        credential_embeddings = np.vstack([item[1] for item in all_credentials])
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            credential_embeddings
        )[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            credential_id = credential_ids[idx]
            similarity_score = similarities[idx]
            credential_data = self.db.get_credential(credential_id)
            
            if credential_data:
                credential_data['relevance_score'] = float(similarity_score)
                results.append(credential_data)
        
        return results
    
    def batch_index_credentials(self, credential_ids: List[int]) -> int:
        """Process a batch of credentials to generate and store embeddings"""
        success_count = 0
        for cred_id in credential_ids:
            if self.process_new_credential(cred_id):
                success_count += 1
        return success_count


class CredentialVerifier:
    """Handles verification of credentials"""
    
    def __init__(self, db: Optional[CredentialDatabase] = None):
        """Initialize the credential verifier"""
        self.db = db if db else CredentialDatabase()
        
    def verify_manual(self, credential_id: int, verifier_notes: str) -> bool:
        """Manually verify a credential"""
        return self.db.verify_credential(
            credential_id=credential_id,
            verification_level=1,
            verification_source=f"Manual verification: {verifier_notes}"
        )
    
    def verify_institutional(self, credential_id: int, institution_id: str,
                           verification_token: str) -> bool:
        """Verify a credential through an institutional source"""
        # In a real system, this would validate the token with the institution's API
        # Here we're just simulating the process
        if self._validate_institution_token(institution_id, verification_token):
            return self.db.verify_credential(
                credential_id=credential_id,
                verification_level=2,
                verification_source=f"Institution verified: {institution_id}"
            )
        return False
    
    def verify_blockchain(self, credential_id: int, blockchain_proof: str) -> bool:
        """Verify a credential using blockchain proof"""
        # In a real system, this would validate the proof on the blockchain
        # Here we're just simulating the process
        if self._validate_blockchain_proof(blockchain_proof):
            return self.db.verify_credential(
                credential_id=credential_id,
                verification_level=3,
                verification_source=f"Blockchain verified: {blockchain_proof[:10]}..."
            )
        return False
    
    def _validate_institution_token(self, institution_id: str, token: str) -> bool:
        """Validate an institutional verification token"""
        # Mock implementation - would connect to institution's verification API
        return len(token) > 10 and institution_id.isalnum()
    
    def _validate_blockchain_proof(self, proof: str) -> bool:
        """Validate a blockchain verification proof"""
        # Mock implementation - would validate against actual blockchain
        try:
            proof_data = json.loads(proof)
            return "txHash" in proof_data and "timestamp" in proof_data
        except:
            return False


# Example usage code

def demo_system():
    """Demonstration of the credential search and verification system"""
    
    # Initialize the database
    db = CredentialDatabase(":memory:")  # Using in-memory DB for demo
    
    # Add a test user
    db.cursor.execute(
        "INSERT INTO users (name, email, profile) VALUES (?, ?, ?)",
        ("John Doe", "john@example.com", "Software Engineer with 5 years experience")
    )
    user_id = db.cursor.lastrowid
    db.conn.commit()
    
    # Add some sample credentials
    credentials = [
        {
            "title": "Machine Learning Specialist",
            "description": "Certified in developing and deploying machine learning models including neural networks and decision trees. Proficient in TensorFlow and PyTorch.",
            "issuer": "TechCert Institute",
            "issue_date": "2023-05-15"
        },
        {
            "title": "AWS Certified Solutions Architect",
            "description": "Professional certification validating expertise in designing distributed systems on AWS. Covers compute, networking, storage, and database AWS services.",
            "issuer": "Amazon Web Services",
            "issue_date": "2022-10-22",
            "expiry_date": "2025-10-22"
        },
        {
            "title": "Python Developer Certification",
            "description": "Certified in Python programming, including OOP concepts, data structures, algorithms, and web development using Django and Flask.",
            "issuer": "Python Software Foundation",
            "issue_date": "2021-08-30"
        }
    ]
    
    credential_ids = []
    for cred in credentials:
        cred_id = db.add_credential(
            holder_id=user_id,
            title=cred["title"],
            description=cred["description"],
            issuer=cred["issuer"],
            issue_date=cred["issue_date"],
            expiry_date=cred.get("expiry_date")
        )
        credential_ids.append(cred_id)
    
    # Initialize the search engine
    search_engine = NLPCredentialSearch()
    
    # Index the credentials
    for cred_id in credential_ids:
        search_engine.process_new_credential(cred_id)
    
    # Perform some example searches
    search_queries = [
        "machine learning expert who knows neural networks",
        "someone who can build cloud infrastructure on AWS",
        "Python programmer with web development skills",
        "data science and big data analytics professional"
    ]
    
    print("=== SEARCH RESULTS ===")
    for query in search_queries:
        print(f"\nSearch query: '{query}'")
        results = search_engine.search_credentials(query, top_k=2)
        
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} ({result['issuer']}) - Relevance: {result['relevance_score']:.2f}")
            print(f"   {result['description'][:100]}...")
    
    # Demonstrate verification
    verifier = CredentialVerifier(db)
    
    # Manual verification
    verifier.verify_manual(
        credential_ids[0], 
        "Verified through phone call with issuing institution"
    )
    
    # Institutional verification
    verifier.verify_institutional(
        credential_ids[1],
        "aws-certification",
        "valid-token-12345"
    )
    
    # Blockchain verification
    blockchain_proof = json.dumps({
        "txHash": "0x1234567890abcdef",
        "timestamp": "2023-12-15T14:22:10Z",
        "issuerDID": "did:example:123456789abcdefghi"
    })
    verifier.verify_blockchain(
        credential_ids[2],
        blockchain_proof
    )
    
    # Check verification status
    print("\n=== VERIFICATION STATUS ===")
    for cred_id in credential_ids:
        cred = db.get_credential(cred_id)
        print(f"{cred['title']}: Level {cred['verification_level']} - {cred['verification_source']}")
    
    db.close()

if __name__ == "__main__":
    demo_system()