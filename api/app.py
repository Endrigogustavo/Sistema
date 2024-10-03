import os
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

# Configurar o Firebase
cred_path = os.path.join(os.path.dirname(__file__), '../Database/FirebaseDaeLink.json')

if not os.path.exists(cred_path):
    print("Arquivo de credenciais não encontrado. Verifique o caminho.")
else:
    print("Arquivo de credenciais encontrado.")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

collection_name = 'PCD'

def get_jobs_from_firestore():
    jobs_ref = db.collection(collection_name)
    docs = jobs_ref.stream()
    jobs = []
    for doc in docs:
        job = doc.to_dict()
        job['id'] = doc.id  
        jobs.append(job)
    return jobs

jobs = get_jobs_from_firestore()
descriptions = [job['descrição'] for job in jobs]
tfidf = TfidfVectorizer().fit_transform(descriptions)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    job_title = data.get('descrição')

    job_index = find_job_index_by_similar_description(job_title)

    if job_index is None:
        return jsonify({"error": "Esta vaga não existe"}), 404

    cosine_similarities = linear_kernel(tfidf[job_index:job_index + 1], tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-20:-1]

    recommendations = [jobs[i] for i in related_docs_indices if i != job_index]
    recommendations.insert(0, jobs[job_index])

    return jsonify(recommendations)

def find_job_index_by_similar_description(description):
    job_descriptions = [job.get('descrição', '') for job in jobs]
    job_descriptions.append(description)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_descriptions)

    cosine_similarities = linear_kernel(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    most_similar_job_index = cosine_similarities.argmax()

    if cosine_similarities[most_similar_job_index] > 0.1:
        return most_similar_job_index

    return None

if __name__ == '__main__':
    app.run(debug=True)
