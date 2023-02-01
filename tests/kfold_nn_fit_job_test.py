import os
import sys
import shutil
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(".."))
from ringer.jobs import KFoldNNFitJob


def build_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(30,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[],
    )
    return model


dataset_path = os.path.join("test_data", "breast_cancer_dataset.parquet")
dataset = pd.read_parquet(dataset_path)
job_id = "0"

model_config_path = os.path.join("test_data",
                                 "binary_classification_model_config.json")

fit_kwargs = dict(
    epochs=6,
    verbose=0
)

dataset_dict = {
    "path": os.path.join("test_data", "breast_cancer_dataset.parquet"),
    "type": "parquet",
    "feature_names": "all",
    "target_names": ["target"]
}

output_dir = os.path.join("test_data", "job_outputs", "kfold_job_test")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

fit_job = KFoldNNFitJob(
    job_id="KFoldNNFitJob",
    dataset_info=dataset_dict,
    fit_kwargs=fit_kwargs,
    preprocessing_pipeline=StandardScaler(),
    fit_pipeline=True,
    n_folds=2,
    fold_col_name="fold_id",
    n_inits=5,
    output_dir=output_dir,
    n_jobs=4,
    build_fn=build_fn,
    build_fn_kwargs={}
)
res = fit_job.run()
print("Finished")
