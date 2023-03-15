# MODELS

# - $HOME/workspace/tunings/TrigL2_20180125_v8 \
# - $HOME/workspace/tunings/TrigL2_20220905_v8.100 \
# - $HOME/workspace/tunings/TrigL2_20210907_v8.5 \
# - $HOME/workspace/tunings/TrigL2_20220912_v8.5.100 \
# - $HOME/workspace/tunings/TrigL2_20210306_v10 \
# - $HOME/workspace/tunings/TrigL2_20220822_v10.100 \
# - $HOME/workspace/tunings/TrigL2_20220704_v20 \
# - $HOME/workspace/tunings/TrigL2_20220802_v20.100 \
# - $HOME/workspace/tunings/TrigL2_20230303_vInceptionPerLayer \

# DATASETS

# mc16_boosted


cd ~/workspace/ringer
python simulate_chains.py \
--dataset mc16_boosted \
--models \
$HOME/workspace/tunings/TrigL2_20180125_v8 \
$HOME/workspace/tunings/TrigL2_20220905_v8.100 \
$HOME/workspace/tunings/TrigL2_20210907_v8.5 \
$HOME/workspace/tunings/TrigL2_20220912_v8.5.100 \
$HOME/workspace/tunings/TrigL2_20210306_v10 \
$HOME/workspace/tunings/TrigL2_20220822_v10.100 \
$HOME/workspace/tunings/TrigL2_20220704_v20 \
$HOME/workspace/tunings/TrigL2_20220802_v20.100 \
$HOME/workspace/tunings/TrigL2_20230303_vInceptionPerLayer \
$HOME/workspace/tunings/TrigL2_20230305_vInceptionPerLayer.100 \
--log \
--cutbased