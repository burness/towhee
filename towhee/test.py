from towhee import pipeline

embedding_pipeline = pipeline('image-embedding')
embedding = embedding_pipeline('https://docs.towhee.io/img/logo.png')
print("embedding shape: {0}".format(embedding.shape))