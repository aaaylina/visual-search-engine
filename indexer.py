import os
import pickle
from extractor import FeatureExtractor
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def build_index():
    extractor = FeatureExtractor()
    image_paths = []
    embeddings = []

    dataset_dir = "dataset"

    for filename in os.listdir(dataset_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(dataset_dir, filename)
            image_paths.append(path)

            vector = extractor.extract(path)
            embeddings.append(vector)

    print(f"Обработано {len(image_paths)} картинок. Обучаю k-NN...")

    knn = NearestNeighbors(n_neighbors=3, metric='cosine')
    knn.fit(embeddings)

    with open('knn_model.pkl', 'wb') as f:
        pickle.dump({'knn': knn, 'paths': image_paths}, f)

    df = pd.DataFrame({
        'image_id': range(len(image_paths)), 
        'file_path': image_paths
    })
    df.to_csv('dataset_meta.csv', index=False)

    print("индекс сохранен в knn_model.pkl")

if __name__ == "__main__":
    build_index()
