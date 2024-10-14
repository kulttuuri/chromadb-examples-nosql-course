import os
import time
import chromadb
from embedding_helper_class import TextEmbeddingGenerator

client = chromadb.Client()
print("\n### EXERCISE 3 - FINDING WEBSITE TEXT EMBEDDINGS WITH CHROMADB")

###
# 1. MODEL VECTOR EMBEDDING GENERATOR CONFIGURATIONS
#    Specify your configurations here for the text embedder.
###

embedder = TextEmbeddingGenerator()

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
collection_name = "text_embeddings"
collection = client.create_collection(
    name = collection_name,
    # Configure the distance / similarity metric below.
    # Can be: "l2" (euclidean), "cosine", or "ip".
    # Note that the cosine and ip is also transferred to distance metric, thus lower number is always better match.
    metadata = { "hnsw:space": "cosine" }
)

print("Distance / similarity metric used: " + str(collection.metadata))
if embedder.normalize: print("Vector normalization is enabled with method: " + embedder.normalization_method)
else: print("Vector normalization is disabled")

###
# 3. ADD DOCUMENTS TO THE COLLECTION
#    Add your website texts to the collection.
#    Model "bert-base-uncased" is used to generate embeddings for the text.
###
start_time = time.time()

collection.add(
    embeddings=[embedder.get_website_text_as_embedding("https://www.prusa3d.com/")],
    metadatas=[{ "url": "https://www.prusa3d.com/" }],
    ids=["1"]
)

collection.add(
    embeddings=[embedder.get_website_text_as_embedding("https://www.apple.com/")],
    metadatas=[{ "url": "https://www.apple.com/" }],
    ids=["2"]
)

collection.add(
    embeddings=[embedder.get_website_text_as_embedding("https://www.trychroma.com/")],
    metadatas=[{ "url": "https://www.trychroma.com/" }],
    ids=["3"]
)

collection.add(
    embeddings=[embedder.get_website_text_as_embedding("https://www.samk.fi/en/samk/sustainable-development/")],
    metadatas=[{ "url": "https://www.samk.fi/en/samk/sustainable-development/" }],
    ids=["4"]
)

print(f"Added {len(collection.get(ids=[])['ids'])} documents to the collection in {time.time() - start_time:.2f}s")
###
# 4. FIND THE MOST SIMILAR / CLOSEST RESULTS TO A GIVEN TEXT.
#    Model "bert-base-uncased" is used to generate embeddings for the text.
# 
# For the actual search it uses ANN and the HNSW method in the background.
# This means that the result is not always the EXACTLY correct one, but one that is very close to it.
# > Trading speed for accuracy!
###

what_to_search_for = "SAMK sustainability and sustainable development"
#what_to_search_for = "Chroma database system"
#what_to_search_for = "Apple iphone 16"
#what_to_search_for = "Prusa 3d printer"
how_many_results = 4

print("\nSEARCHING FOR: " + what_to_search_for)
print(f"\nRESULTS (top {how_many_results} results):")

results = collection.query(
    query_embeddings=[embedder.text_to_vector(what_to_search_for)],
    n_results=how_many_results, # How many most approximate similar results to get back
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
)

# Iterate over the results and print them in the desired format
ids = results.get('ids', [[]])[0]
distances = results.get('distances', [[]])[0]
metadatas = results.get('metadatas', [[]])[0]
for result_id, distance, metadata in zip(ids, distances, metadatas):
    url = metadata.get('url', 'No url available')
    print(f"id: {result_id}, distance: {distance:.6f}, url: {url}")