from towhee import pipeline
embedding_pipeline = pipeline('towhee/audio-embedding-vggish')

import numpy as np
import os
from pathlib import Path

dataset_path = './music_dataset/dataset'
music_list = [f for f in Path(dataset_path).glob('*')]
vec_sets = []

for audio_path in music_list:
    vecs = embedding_pipeline(str(audio_path))
    norm_vecs = [vec / np.linalg.norm(vec) for vec in vecs[0][0]]
    vec_sets.append(norm_vecs)
import pymilvus as milvus

# Gather vectors in a list
vectors = []
for i in range(len(vec_sets)):
    for vec in vec_sets[i]:
        vectors.append(vec)

collection_name = 'music_recognition'
vec_dim = len(vectors[0])

# connect to local Milvus service
milvus.connections.connect(host='localhost', port=19530)

# create collection
id_field = milvus.FieldSchema(name="id", dtype=milvus.DataType.INT64, descrition="int64", is_primary=True, auto_id=True)
vec_field = milvus.FieldSchema(name="vec", dtype=milvus.DataType.FLOAT_VECTOR, dim=vec_dim)
schema = milvus.CollectionSchema(fields=[id_field, vec_field])
collection = milvus.Collection(name=collection_name, schema=schema)

# insert data to Milvus
res = collection.insert([vectors])

# maintain mappings between primary keys of music clips and the original music for retrieval
full_music_list = []
music_dict = dict()
for i in range(len(vec_sets)):
    for _ in range(len(vec_sets[i])):
        full_music_list.append(music_list[i])        
for i, pk in enumerate(res.primary_keys):
    music_dict[pk] = full_music_list[i]

query_audio_path = './music_dataset/query/blues_clip.wav'
query_vecs = embedding_pipeline(query_audio_path) # Get vectors of the given audio
norm_query_vecs = [vec / np.linalg.norm(vec) for vec in query_vecs[0][0]] # Normalize vectors

collection.load()
results = collection.search(data=norm_query_vecs, anns_field="vec", param={"metric_type": 'L2'}, limit=1)

import IPython

votes = [music_dict[x.ids[0]] for x in results]
pred = max(set(votes), key = votes.count)

print(str(pred))
IPython.display.Audio(Path(pred))

print(str(query_audio_path))
IPython.display.Audio(Path(query_audio_path))



