import time
import chromadb

client = chromadb.Client()
print("\n### EXERCISE 1 - DEFAULT CHROMADB EMBEDDING TEXT SEARCH")

###
# 1. CREATE COLLECTION & CONFIGURE DISTANCE METRIC
#    Configure the distance metric below.
###

# Create a collection to hold our data
collection_name = "default_text_embeddings"
collection = client.create_collection(
    name = collection_name,
    # Configure the distance / similarity metric below.
    # Can be: "l2" (euclidean), "cosine", or "ip".
    # Note that the cosine and ip is also transferred to distance metric, thus lower number is always better match.
    metadata = { "hnsw:space": "cosine" }
)
print("Distance / similarity metric used: " + str(collection.metadata))

###
# 2. ADD DOCUMENTS TO THE COLLECTION
#    Add your website texts to the collection.
#    If we add strings / plain text in the document, like below, then chromadb will automatically use
#    model "MiniLM-L6-v2" to create a vector embedding of the given text.
#    Note: this uses CPU, which is SLOW compared to GPU!
###
start_time = time.time()

collection.add(
    documents=["print('Hello world!')"],
    metadatas=[{ "language": "Python", "source": "own" }],
    ids=["1"]
)

collection.add(
    documents=["std::cout('Hello world!')"],
    metadatas=[{ "language": "c++", "source": "own" }],
    ids=["2"]
)

collection.add(
    documents=["for i in items: print(i)"],
    metadatas=[{ "language": "Python", "source": "own" }],
    ids=["3"]
)

print(f"Added {len(collection.get(ids=[])['ids'])} documents to the collection in {time.time() - start_time:.2f}s")

###
# 3. FIND THE MOST SIMILAR / CLOSEST RESULTS TO A GIVEN TEXT.
# While passing text as a query text, it will again use the model "MiniLM-L6-v2" to create the vector embedding.
# 
# For the actual search it uses ANN and the HNSW method in the background.
# This means that the result is not always the EXACTLY correct one, but one that is very close to it.
# > Trading speed for accuracy!
###

what_to_search_for = "Write python print hello world"
#what_to_search_for = "what was cout"
#what_to_search_for = "loop with python, for"
how_many_results = 3

print("\nSEARCHING FOR: " + what_to_search_for)
print(f"\nRESULTS (top {how_many_results} results):")

results = collection.query(
    query_texts=what_to_search_for,
    n_results=how_many_results, # How many most approximate similar results to get back
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

# Iterate over the results and print them in the desired format
ids = results.get('ids', [[]])[0]
distances = results.get('distances', [[]])[0]
metadatas = results.get('metadatas', [[]])[0]
documents = results.get('documents', [[]])[0]
for result_id, distance, metadata, document in zip(ids, distances, metadatas, documents):
    language = metadata.get('language', 'No language available')
    source = metadata.get('source', 'No source available')
    print(f"id: {result_id}, distance: {distance:.6f}, language: {language}, source: {source}, document: {document}")