from tensorflow.keras import layers
from Models.Architectures.Layers.Embeddings import Glove


def choose_embeddings(max_features, embedding_dim, sequence_length, vectorization_layer, inputs, embedding_type):
    if embedding_type == "glove":
        return Glove.glove(max_features, embedding_dim, sequence_length, vectorization_layer, inputs)
    elif embedding_type == "standard":
        return tf_standard(max_features, embedding_dim, sequence_length, inputs)


def tf_standard(max_features, embedding_dim, sequence_length, inputs):
    x = layers.Embedding(input_dim=max_features,
                         output_dim=embedding_dim,
                         input_length=sequence_length)(inputs)

    return x
