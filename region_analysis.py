import os
import pandas as pd
from ringer.data import NamedDatasetLoader
import ringer.regions as regions


dataset_name = 'mc16_boosted'
region_name = 'l2calo_2017'
et_eta_regions, n_ets, n_etas = regions.get_named_et_eta_regions(region_name)

data_loader = NamedDatasetLoader(dataset_name)
data_df = data_loader.load_data_df(
    columns=[et_eta_regions[0].et_key, et_eta_regions[0].eta_key]
)

region_sample_count = regions.count_region_samples(
    data_df,
    et_eta_regions
)

sample_count_data = list()
for region, count in zip(et_eta_regions, region_sample_count):
    sample_count_data.append(
        [region.et_idx, region.eta_idx, count, dataset_name]
    )

sample_count_df = pd.DataFrame(
    sample_count_data,
    columns=['et_idx', 'eta_idx', 'sample', 'dataset']
)
sample_count_df = sample_count_df.pivot(index='eta_idx',
                                        columns='et_idx',
                                        values='sample')
sample_count_df.to_json(os.path.join('data', 'region_sample_count.json'))
