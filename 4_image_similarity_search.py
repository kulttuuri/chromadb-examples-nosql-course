import os
import time
import uuid
import json
import chromadb
from embedding_helper_class import ImageEmbeddingGenerator

client = chromadb.Client()
print("\n### EXERCISE 4 - FINDING SIMILAR IMAGE EMBEDDINGS WITH CHROMADB")

###
# 1. MODEL VECTOR EMBEDDING GENERATOR CONFIGURATIONS
#    Specify your configurations here for the image embedder.
###

embedder = ImageEmbeddingGenerator()

# Normalize generated vector embeddings?
embedder.normalize = False

# Only works when normalize above is enabled.
# Can be "L2" (Euclidean) or "min-max".
# Note that L2 normalization is not ideal for L2 metric.
embedder.normalization_method = "L2"

###
# 2. CREATE COLLECTION & CONFIGURE DISTANCE METRIC
#    Configure the distance metric below.
###

# Create a collection to hold our data
collection_name = "image_embeddings"
collection = client.create_collection(
    name = collection_name,
    # Configure the distance / similarity metric below.
    # Can be: "l2" (euclidean), "cosine", or "ip".
    # Note that the cosine and ip is also transferred to distance metric, thus lower number is always better match.
    metadata = { "hnsw:space": "l2" }
)

print("Distance / similarity metric used: " + str(collection.metadata))
if embedder.normalize: print("Vector normalization is enabled with method: " + embedder.normalization_method)
else: print("Vector normalization is disabled")

###
# 3. ADD DOCUMENTS TO THE COLLECTION
#    All .jpg files are automatically loaded from folder find_images and vectorized to a file called
#    image_vector_embeddings.json. They are then dumbed to collection from that json file.
###
vector_file = "image_vector_embeddings.json"
embedder.images_to_vector_json_file("db_images", "image_vector_embeddings.json")

# Read image vector embeddings from the JSON file
with open('image_vector_embeddings.json', 'r') as f:
    image_embeddings = json.load(f)

# Loop through each object in the JSON and add it to the collection
start_time = time.time()

for image_data in image_embeddings:
    # Generate a random UUID for each image
    image_id = str(uuid.uuid4())
    filename = image_data["filename"]
    embedding = image_data["vector_embedding"]
    
    # Add the image data to the collection
    collection.add(
        embeddings=[embedding],
        metadatas=[{"filename": filename}],
        ids=[image_id]
    )

print(f"Added {len(collection.get(ids=[])['ids'])} documents to the collection in {time.time() - start_time:.2f}s")

###
# 4. FIND THE MOST SIMILAR / CLOSEST RESULTS TO A GIVEN TEXT.
# While passing text as a query text, it will again use the model "MiniLM-L6-v2" to create the vector embedding.
# 
# For the actual search it uses ANN and the HNSW method in the background.
# This means that the result is not always the EXACTLY correct one, but one that is very close to it.
# > Trading speed for accuracy!
###

what_to_search_for = "find_images/cat2.jpg"
how_many_results = 3

print("\nSEARCHING FOR: " + what_to_search_for)
print(f"\nRESULTS (top {how_many_results} results):")

results = collection.query(
    query_embeddings=[embedder.get_image_vector_embedding(what_to_search_for)],
    n_results=how_many_results, # How many most approximate similar results to get back
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
)

# Iterate over the results and print them in the desired format
ids = results.get('ids', [[]])[0]
distances = results.get('distances', [[]])[0]
metadatas = results.get('metadatas', [[]])[0]
for result_id, distance, metadata in zip(ids, distances, metadatas):
    filename = metadata.get('filename', 'No filename available')
    print(f"id: {result_id}, distance: {distance:.6f}, filename: {filename}")