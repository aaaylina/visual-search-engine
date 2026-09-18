import pickle
import io
from fastapi import FastAPI, UploadFile, File
from extractor import FeatureExtractor

app = FastAPI(title="Visual Search Engine")

extractor = FeatureExtractor()

with open('knn_model.pkl', 'rb') as f:
    data = pickle.load(f) 
    knn_model = data['knn']
    image_paths = data['paths']

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_stream = io.BytesIO(image_bytes)

    query_vector = extractor.extract(image_stream)

    distances, indices = knn_model.kneighbors([query_vector])

    results = []
    for i in range(len(indices[0])):
        results.append({
            "image_path": image_paths[indices[0][i]],
            "distance": float(distances[0][i])
        })

    return {"status": "success", "similar_items": results}