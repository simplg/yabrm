from argparse import ArgumentParser
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
import math


def create_model_inputs(sequence_length=4):
    return {
        "user_id": layers.Input(name="user_id", shape=(1,), dtype=tf.int32),
        "sequence_book_ids": layers.Input(
            name="sequence_book_ids", shape=(sequence_length - 1,), dtype=tf.int32
        ),
        "target_book_id": layers.Input(
            name="target_book_id", shape=(1,), dtype=tf.int32
        ),
        "sequence_ratings": layers.Input(
            name="sequence_ratings", shape=(sequence_length - 1,), dtype=tf.float32
        ),
    }


def encode_input_features(inputs, vocab: dict, sequence_length=4, include_user_id=False):
    encoded_transformer_features = []
    encoded_other_features = []

    other_feature_names = []
    if include_user_id:
        other_feature_names.append("user_id")

    ## Encode user features
    for feature_name in other_feature_names:
        if not feature_name in vocab:
            raise ValueError(f"{feature_name} is not in vocab")
        # Convert the string input values into integer indices.
        vocabulary = vocab[feature_name]
        lookup_type = layers.IntegerLookup
        if isinstance(vocab[feature_name], dict):
            vocabulary = vocab[feature_name]["vocabulary"]
            lookup_type = vocab[feature_name]["lookup_type"]
        idx = lookup_type(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)(
            inputs[feature_name]
        )
        # Compute embedding dimensions
        embedding_dims = int(math.sqrt(len(vocabulary)))
        # Create an embedding layer with the specified dimensions.
        embedding_encoder = layers.Embedding(
            input_dim=len(vocabulary),
            output_dim=embedding_dims,
            name=f"{feature_name}_embedding",
        )
        # Convert the index values to embedding representations.
        encoded_other_features.append(embedding_encoder(idx))

    ## Create a single embedding vector for the user features
    if len(encoded_other_features) > 1:
        encoded_other_features = layers.concatenate(encoded_other_features)
    elif len(encoded_other_features) == 1:
        encoded_other_features = encoded_other_features[0]
    else:
        encoded_other_features = None

    ## Create a books embedding encoder
    book_vocabulary = vocab["book_id"]
    book_embedding_dims = int(math.sqrt(len(book_vocabulary)))
    # Create a lookup to convert string values to integer indices.
    book_index_lookup = layers.IntegerLookup(
        vocabulary=book_vocabulary,
        mask_token=None,
        num_oov_indices=0,
        name="book_index_lookup",
    )
    # Create an embedding layer with the specified dimensions.
    book_embedding_encoder = layers.Embedding(
        input_dim=len(book_vocabulary),
        output_dim=book_embedding_dims,
        name=f"book_embedding",
    )

    ## Define a function to encode a given book id.
    def encode_book(book_id):
        # Convert the string input values into integer indices.
        book_idx = book_index_lookup(book_id)
        book_embedding = book_embedding_encoder(book_idx)
        encoded_book = book_embedding
        return encoded_book

    ## Encoding target_book_id
    target_book_id = inputs["target_book_id"]
    encoded_target_book = encode_book(target_book_id)

    ## Encoding sequence book_ids.
    sequence_books_ids = inputs["sequence_book_ids"]
    encoded_sequence_books = encode_book(sequence_books_ids)
    # Create positional embedding.
    position_embedding_encoder = layers.Embedding(
        input_dim=sequence_length,
        output_dim=book_embedding_dims,
        name="position_embedding",
    )
    positions = tf.range(start=0, limit=sequence_length - 1, delta=1)
    encodded_positions = position_embedding_encoder(positions)
    # Retrieve sequence ratings to incorporate them into the encoding of the book.
    sequence_ratings = tf.expand_dims(inputs["sequence_ratings"], -1)
    # Add the positional encoding to the book encodings and multiply them by rating.
    encoded_sequence_books_with_position_and_rating = layers.Multiply()(
        [(encoded_sequence_books + encodded_positions), sequence_ratings]
    )

    # Construct the transformer inputs.
    for encoded_book in tf.unstack(
        encoded_sequence_books_with_position_and_rating, axis=1
    ):
        encoded_transformer_features.append(tf.expand_dims(encoded_book, 1))
    encoded_transformer_features.append(encoded_target_book)

    encoded_transformer_features = layers.concatenate(
        encoded_transformer_features, axis=1
    )

    return encoded_transformer_features, encoded_other_features

def create_model(vocab, sequence_length=4, include_user_id=False, dropout_rate=.2, hidden_units=[256, 128], num_heads=3):
    inputs = create_model_inputs(sequence_length)
    transformer_features, other_features = encode_input_features(
        inputs, vocab, sequence_length=sequence_length, include_user_id=include_user_id
    )

    # Create a multi-headed attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=transformer_features.shape[2], dropout=dropout_rate
    )(transformer_features, transformer_features)

    # Transformer block.
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x1 = layers.Add()([transformer_features, attention_output])
    x1 = layers.LayerNormalization()(x1)
    x2 = layers.LeakyReLU()(x1)
    x2 = layers.Dense(units=x2.shape[-1])(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    transformer_features = layers.Add()([x1, x2])
    transformer_features = layers.LayerNormalization()(transformer_features)
    features = layers.Flatten()(transformer_features)

    # Included the other features.
    if other_features is not None:
        features = layers.concatenate(
            [features, layers.Reshape([other_features.shape[-1]])(other_features)]
        )

    # Fully-connected layers.
    for num_units in hidden_units:
        features = layers.Dense(num_units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.LeakyReLU()(features)
        features = layers.Dropout(dropout_rate)(features)

    outputs = layers.Dense(units=1)(features)
    model = Model(inputs=inputs, outputs=outputs)
    return model