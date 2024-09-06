# import os
# import json
# import torch
# import clip
# import numpy as np
# from PIL import Image
# from flask import Flask, request, jsonify, send_from_directory, render_template
# from sklearn.metrics.pairwise import cosine_similarity

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image_folder = "car_images"  # Updated folder path
# metadata_path = "image_labels.json"

# app = Flask(__name__)

# # Load images and metadata
# def load_images_and_metadata(image_folder, metadata_path):
#     with open(metadata_path, 'r') as f:
#         metadata = json.load(f)

#     images = []
#     descriptions = []
#     image_files = []

#     # Load labeled images
#     for image_file, data in metadata.items():
#         image_path = os.path.join(image_folder, image_file)
#         if os.path.exists(image_path):
#             images.append(Image.open(image_path))
#             descriptions.append(data['description'])
#             image_files.append(image_file)

#     # Load new images
#     for image_file in os.listdir(image_folder):
#         if image_file not in metadata:
#             image_path = os.path.join(image_folder, image_file)
#             if os.path.exists(image_path):
#                 images.append(Image.open(image_path))
#                 descriptions.append("unknown image")
#                 image_files.append(image_file)

#     return images, descriptions, image_files

# # Generate embeddings
# def generate_embeddings(model, preprocess, images, descriptions):
#     image_embeddings = []
#     for image in images:
#         image_input = preprocess(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             image_embedding = model.encode_image(image_input)
#         image_embeddings.append(image_embedding.cpu().numpy())

#     text_tokens = clip.tokenize(descriptions).to(device)
#     with torch.no_grad():
#         text_embeddings = model.encode_text(text_tokens)

#     return image_embeddings, text_embeddings.cpu().numpy()

# # Query function
# def query_images(query, image_embeddings, image_files, model):
#     text_tokens = clip.tokenize([query]).to(device)
#     with torch.no_grad():
#         query_embedding = model.encode_text(text_tokens).cpu().numpy()

#     similarities = cosine_similarity(query_embedding, np.vstack(image_embeddings)).flatten()
#     sorted_indices = np.argsort(similarities)[::-1]

#     results = [(image_files[idx], similarities[idx]) for idx in sorted_indices]
#     return results

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/search', methods=['POST'])
# def search():
#     query = request.form['query']
#     results = query_images(query, image_embeddings, image_files, model)
#     response = []
#     for image_file, similarity in results[:5]:
#         response.append({
#             'file_name': image_file,
#             'similarity': float(similarity),
#             'description': descriptions[image_files.index(image_file)],
#             'image_url': f'/car_images/{image_file}'
#         })
#     return jsonify(response)

# @app.route('/car_images/<path:filename>')
# def send_image(filename):
#     return send_from_directory(image_folder, filename)

# if __name__ == "__main__":
#     images, descriptions, image_files = load_images_and_metadata(image_folder, metadata_path)
#     image_embeddings, text_embeddings = generate_embeddings(model, preprocess, images, descriptions)
#     app.run(debug=True)


def solution(n, m, figures):
    # Initialize the grid with zeros
    grid = [[0] * m for _ in range(n)]
    
    # Define the shapes with their relative positions
    shapes = {
        'A': [(0, 0), (0, 1), (1, 0), (1, 1)],       # 2x2 square
        'B': [(0, 0), (1, 0), (2, 0), (2, 1)],       # L-shape
        'C': [(0, 1), (1, 0), (1, 1), (2, 0)],       # Z-shape
        'D': [(0, 0), (0, 1), (0, 2), (1, 1)],       # T-shape
        'E': [(0, 0), (1, 0), (1, 1), (1, 2)]        # mirrored L-shape
    }
    
    def can_place(shape, x, y):
        for dx, dy in shape:
            if x + dx >= n or y + dy >= m or grid[x + dx][y + dy] != 0:
                return False
        return True
    
    def place_shape(shape, x, y, index):
        for dx, dy in shape:
            grid[x + dx][y + dy] = index
    
    # Place each figure on the grid
    for idx, figure in enumerate(figures, 1):
        shape = shapes[figure]
        placed = False
        for i in range(n):
            for j in range(m):
                if can_place(shape, i, j):
                    place_shape(shape, i, j, idx)
                    placed = True
                    break
            if placed:
                break

    return grid

# Test case
n = 4
m = 4
figures = ['D', 'B', 'A', 'C']
output = solution(n, m, figures)
for row in output:
    print(row)
