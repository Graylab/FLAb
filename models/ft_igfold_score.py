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

#from embedding_help_antiberty import create_embedding
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


def ft_igfold_gt_score(train_df, valid_df, test_df):
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
            emb = create_embedding(df, emb_type="per_prot")
            emb.to_pickle(df_path)  # overwrite with embeddings

        # Train model using embeddings
        model, history, spearman = train_predictor(train_file, valid_file, test_file, seed=42)

        # Predict on test set
        test_emb = pd.read_pickle(test_file)
        predictions = model.predict(test_emb.iloc[:, :-3])

    # All temporary files deleted automatically after exiting context
    return predictions.tolist()



# ---------------- PATCH FOR TORCH ≥2.6 ----------------
import torch
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.tokenization_bert import BertTokenizer

torch.serialization.add_safe_globals([BertConfig])
torch.serialization.add_safe_globals([BertTokenizer])
# ------------------------------------------------------

from transformers.tokenization_utils import Trie
from transformers.models.bert.tokenization_bert import BasicTokenizer, WordpieceTokenizer
torch.serialization.add_safe_globals([Trie, BasicTokenizer, WordpieceTokenizer])

import pytorch_lightning as pl
from pytorch_lightning.core.saving import load as pl_load

orig_load = pl_load

def patched_load(*args, **kwargs):
    checkpoint = orig_load(*args, **kwargs)
    if "pytorch-lightning_version" not in checkpoint:
        checkpoint["pytorch-lightning_version"] = "1.6.5"  # or whatever IgFold was trained with
    return checkpoint

pl.core.saving.load = patched_load

from igfold import IgFoldRunner

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
}

igfold = IgFoldRunner()
emb = igfold.embed(
    sequences=sequences,  # Antibody sequences
)

print(emb.bert_embs)       # Embeddings from AntiBERTy final hidden layer (dim: 1, L, 512)
print(emb.gt_embs)         # Embeddings after graph transformer layers (dim: 1, L, 64)
print(emb.structure_embs)  # Embeddings after template incorporation IPA (dim: 1, L, 64)



