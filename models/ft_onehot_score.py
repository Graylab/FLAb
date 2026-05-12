import pandas as pd
import os
os.environ["MPLCONFIGDIR"] = os.path.expanduser("~/.config/matplotlib")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface/transformers")
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

import uuid
import tempfile
import argparse
import random
import numpy as np
import scipy
from tensorflow import keras

from embedding_help_onehot import create_embedding
from embedding_train import emb_predictor, set_seed


def train_predictor(train_file, valid_file, test_file,
                    epochs=240, lr=1e-4, epsilon=1e-07,
                    batch=8, dropout=0.2, dense=32, seed=99,
                    num_labels=1):
    """
    Train a prediction head using precomputed embeddings from train/valid/test PKL files
    """

    set_seed(seed)

    # Load embeddings
    df_train = pd.read_pickle(train_file)
    df_valid = pd.read_pickle(valid_file)
    df_test  = pd.read_pickle(test_file)

    # Normalizer for input features
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(df_train.iloc[:, :-3])) # changed from -2 to -3 since have an extra str column now (light)

    model = emb_predictor(lr, dropout, dense, normalizer, epsilon, num_labels)
    model.summary()

    spearman = []

    class Spearman_Callback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            predictions = self.model.predict(df_valid.iloc[:, :-3])
            spearman.append(
                scipy.stats.spearmanr(a=np.array(df_valid.iloc[:, -1]), b=predictions).correlation
            )

    history = model.fit(
        np.array(df_train.iloc[:, :-3]),
        np.array(df_train.iloc[:, -1]),
        batch_size=batch,
        validation_data=[np.array(df_valid.iloc[:, :-3]), np.array(df_valid.iloc[:, -1])],
        epochs=epochs,
        callbacks=[Spearman_Callback()]
    )

    return model, history, spearman


def ft_onehot_score(train_df, valid_df, test_df):
    """
    Input: DataFrames for train, valid, test
    Output: Predicted values for test
    """

    # Create a temporary directory for PKL files
    with tempfile.TemporaryDirectory() as temp_dir:
        unique_id = str(uuid.uuid4())[:8]  # short unique identifier

        # Save temporary PKL files
        train_file = os.path.join(temp_dir, f"train_{unique_id}.pkl")
        valid_file = os.path.join(temp_dir, f"valid_{unique_id}.pkl")
        test_file  = os.path.join(temp_dir, f"test_{unique_id}.pkl")

        train_df.to_pickle(train_file)
        valid_df.to_pickle(valid_file)
        test_df.to_pickle(test_file)

        # Generate embeddings and overwrite PKL files
        for data_name, df_path in zip(["train", "valid", "test"], [train_file, valid_file, test_file]):
            df = pd.read_pickle(df_path)[['heavy', 'light', 'label']]
            # Replace unusual amino acids with "X"
            df["heavy"] = df["heavy"].str.replace('|'.join(["O","B","U","Z","J"]), "X", regex=True)
            df["light"] = df["light"].str.replace('|'.join(["O","B","U","Z","J"]), "X", regex=True)
            emb = create_embedding(df)
            emb.to_pickle(df_path)  # overwrite with embeddings

        # Train model using embeddings
        model, history, spearman = train_predictor(train_file, valid_file, test_file, seed=42)

        # Predict on test set
        test_emb = pd.read_pickle(test_file)
        predictions = model.predict(test_emb.iloc[:, :-3])

    # All temporary files deleted automatically after exiting context
    return predictions.tolist()
