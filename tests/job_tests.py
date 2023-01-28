import os
import sys
import shutil
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(".."))
from ringer.jobs import NNFitJob
# from ringer.logging import set_loggers
# set_loggers()


dataset_path = os.path.join("test_data", "breast_cancer_dataset.parquet")
dataset = pd.read_parquet(dataset_path)
job_id = "0"

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(30,)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

compile_kwargs = dict(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[]
)
fit_kwargs = dict(
    epochs=6
)

output_dir = os.path.join("test_data", "job_outputs")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

fit_job = NNFitJob(
    job_id="0",
    dataset=dataset,
    target_name="target",
    feature_names="all",
    model=model,
    compile_kwargs=compile_kwargs,
    fit_kwargs=fit_kwargs,
    preprocessing_pipeline=StandardScaler(),
    fit_pipeline=True,
    n_folds=10,
    fold=0,
    fold_col_name="fold_id",
    output_dir=output_dir
)
fit_job.run()
