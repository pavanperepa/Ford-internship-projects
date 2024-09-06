import numpy as np
from sentence_transformers import SentenceTransformer
from db_connect import MetapixDb

# Initialize the database connection
db_instance = MetapixDb.instance()
db = db_instance.get_db()
collection = db['car_images']

# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def query_images(query_text):
    """Queries the database for images matching the given query text."""
    query_embedding = model.encode(query_text).tolist()
    
    # Find the most similar images using the embeddings
    results = collection.find({})
    similarities = []

    for result in results:
        image_embedding = result['embeddings']
        similarity = np.dot(query_embedding, image_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(image_embedding))
        similarities.append((result['path'], similarity))

    # Sort the results by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the most similar image paths
    return [path for path, similarity in similarities[:5]]  # Adjust the number of results as needed

# Example query
query = "red ford mustang"
matching_images = query_images(query)
print("Matching images:", matching_images)
