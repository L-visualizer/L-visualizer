import firebase_admin
from firebase_admin import credentials, firestore
cred = credentials.Certificate("firebase-admin-sdk.json")  # Get from Firebase Console
firebase_admin.initialize_app(cred)

db = firestore.client()

def add_user_to_firestore(username: str):
    doc_ref = db.collection("users").document(username)
    doc_ref.set({"username": username, "created_at": firestore.SERVER_TIMESTAMP})
