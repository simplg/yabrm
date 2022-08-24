import pandas as pd
import tensorflow as tf


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):
    def process(features):
        sequence_book_ids = tf.strings.to_number(tf.strings.split(features["sequence_book_ids"], ","), tf.int32).to_tensor()

        # The last book id in the sequence is the target book.
        features["target_book_id"] = sequence_book_ids[:, -1]
        features["sequence_book_ids"] = sequence_book_ids[:, :-1]

        sequence_ratings = tf.strings.to_number(tf.strings.split(features["sequence_ratings"], ","), tf.float32).to_tensor()

        # The last rating in the sequence is the target for the model to predict.
        target = sequence_ratings[:, -1]
        features["sequence_ratings"] = sequence_ratings[:, :-1]

        return features, target

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        num_epochs=1,
        header=False,
        field_delim="|",
        column_names=["sequence_ratings","sequence_book_ids","user_id"],
        shuffle=shuffle,
    ).map(process)

    return dataset
