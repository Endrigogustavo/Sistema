import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import spacy

# Carregar o modelo spaCy
nlp = spacy.load("en_core_web_md")  # ou uma versão mais leve como 'en_core_web_sm'

app = Flask(__name__)
CORS(app)

# Configurar o Firebase
cred_path = os.path.join(os.path.dirname(__file__), '../Database/FirebaseDaeLink.json')
cred = credentials.Certificate(cred_path) if os.path.exists(cred_path) else None

if cred:
    firebase_admin.initialize_app(cred)
else:
    print("Arquivo de credenciais não encontrado. Verifique o caminho.")

db = firestore.client()
collection_name = 'PCD'

def get_jobs_from_firestore():
    jobs_ref = db.collection(collection_name)
    return [{**doc.to_dict(), 'id': doc.id} for doc in jobs_ref.stream()]

jobs = get_jobs_from_firestore()

@app.route('/recommend', methods=['POST'])
def recommend():
    job_title = request.json.get('descrição')
    job_index = find_job_index_by_similar_description(job_title)

    if job_index is None:
        return jsonify({"error": "Esta vaga não existe"}), 404

    recommendations = [jobs[job_index]] + [jobs[i] for i in range(len(jobs)) if i != job_index][:19]
    return jsonify(recommendations)

def find_job_index_by_similar_description(description):
    job_descriptions = [job.get('descrição', '') for job in jobs]
    doc1 = nlp(description)
    similarities = []

    for job in job_descriptions:
        doc2 = nlp(job)
        similarities.append(doc1.similarity(doc2))

    most_similar_index = similarities.index(max(similarities))

    return most_similar_index if max(similarities) > 0.1 else None

if __name__ == '__main__':
    app.run(debug=True)
