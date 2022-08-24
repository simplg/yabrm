from datetime import datetime
import os
from pickle import dump
import pandas as pd
import numpy as np
from itertools import chain
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from ml.helpers import get_engine
from ml.recommandation_models.bst_model import create_model
from ml.recommandation_models.book_dataset import get_dataset_from_csv
from ml.recommandation_models.popularity_model import PopularityBasedModel


def train_popularity_model():
    pop_model = PopularityBasedModel()
    engine = get_engine()
    with engine.connect() as conn:
        ratings_df = pd.read_sql('SELECT r.book_id, AVG(r."value") as avg_rating, COUNT(r.user_id) as vote_count FROM ml_rating r GROUP BY r.book_id', con=conn)
    pop_model.fit(ratings_df)
    dump(pop_model, open('data/output/pop_model.pkl', 'wb'))


def train_bst_model(batch_size=32):
    engine = get_engine()
    with engine.connect() as conn:
        ratings = pd.read_sql('SELECT r.user_id, r.book_id, r."value" as rating FROM ml_rating r WHERE r.book_id < 500', con=conn)
    bst_model = create_model(vocab={"book_id": ratings["book_id"].drop_duplicates().tolist(), "user_id": ratings["user_id"].drop_duplicates().tolist()}, include_user_id=True)
    del ratings
    bst_model.compile(
        optimizer=Adagrad(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()],
    )
    train_dataset = get_dataset_from_csv("data/intermediate/train_data.csv", batch_size=batch_size)
    test_dataset = get_dataset_from_csv("data/intermediate/test_data.csv", batch_size=batch_size)
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = "data/model_cpt/cp-{epoch:04d}.ckpt"

    # Create a callback that saves the model's weights
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    bst_model.fit(train_dataset, epochs=5, validation_data=test_dataset, callbacks=[tensorboard_callback, cp_callback])
    bst_model.save('data/output/bst_model')


def prepare_csv_bst_model():
    engine = get_engine()
    with engine.connect() as conn:
        ratings = pd.read_sql("SELECT r.user_id, r.book_id, r.\"value\" as rating FROM ml_rating r WHERE r.book_id < 500", con=conn)
        ratings["book_id"] = ratings["book_id"].astype(int)
        ratings["user_id"] = ratings["user_id"].astype(int)
        ratings["rating"] = ratings["rating"].astype(int)
        ratings_data = ratings.groupby("user_id").agg({"rating": list, "book_id": list, "user_id": "first"})
        ratings_data["book_id"] = ratings_data["book_id"].apply(create_sequences)

        ratings_data["rating"] = ratings_data["rating"].apply(create_sequences)
        ratings_data = ratings_data.explode(["book_id", "rating"], ignore_index=True).dropna()

        ratings_data.book_id = ratings_data.book_id.apply(lambda x: ",".join([str(v) for v in x]))
        ratings_data.rating = ratings_data.rating.apply(lambda x: ",".join([str(v) for v in x]))

        ratings_data.rename(
            columns={"book_id": "sequence_book_ids", "rating": "sequence_ratings"},
            inplace=True,
        )
        random_selection = np.random.rand(len(ratings_data.index)) <= 0.85
        train_data = ratings_data[random_selection]
        test_data = ratings_data[~random_selection]

        train_data.to_csv("data/intermediate/train_data.csv", index=False, sep="|", header=False)
        test_data.to_csv("data/intermediate/test_data.csv", index=False, sep="|", header=False)


def create_sequences(values, window_size = 4, step_size = 2):
    sequences = []
    start_index = 0
    while True:
        end_index = start_index + window_size
        seq = values[start_index:end_index]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start_index += step_size
    return sequences

