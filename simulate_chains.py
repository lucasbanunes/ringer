import os
import ROOT
ROOT.gStyle.SetOptStat(0);
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from argparse import ArgumentParser
from typing import List
from itertools import product

from kepler.pandas.menu import ElectronSequence as Chain
from kepler.pandas.decorators import RingerDecorator
from ringer.generators import ringer_generators
from ringer.utils import get_logger
from ringer.constants import DROP_COLS, L1SEEDS_PER_ENERGY, CRITERIA_CONF_NAMES, ENERGY_CHAINS, TRIG_STEPS
from ringer.data import NamedDatasetLoader

def parse_args():
    et_bins = [15, 20, 30, 40, 50, 1000000]
    eta_bins = [0.0, 0.8, 1.37, 1.54, 2.37, 2.50]
    chain_choices = list(ENERGY_CHAINS.keys())
    et_choices = list(range(len(et_bins)-1))
    eta_choices = list(range(len(eta_bins)-1))
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset directory path")
    parser.add_argument("--models", nargs="+", required=True, help="models directory path, can be more than one", dest="modelpaths")
    parser.add_argument("--cutbased", action="store_true", help="if passed, plots the cutbased results")
    parser.add_argument("--chains", nargs="+", default=chain_choices, choices=chain_choices, help="chains to be plotted, defults to all chains", type=int, dest="chain_names")
    parser.add_argument("--dev", action="store_true", help="if passed, runs the code only with the leblon region")
    parser.add_argument("--log", action="store_true", help="if passed, creates a log file with script activity")
    parser.add_argument("--ets", nargs="+", choices=et_choices, default=et_choices, type=int,
        help="et regions to simulate")
    parser.add_argument("--etas", nargs="+", choices=eta_choices, default=eta_choices, type=int,
        help="eta regions to simulate")
    args = parser.parse_args().__dict__
    args["chain_names"] = [ENERGY_CHAINS[energy] for energy in args["chain_names"]]
    return args

def build_decorators(modelpaths: List[str], cutbased: bool):
    decorators = list()
    strategy_cols = defaultdict(list)
    if cutbased:
        strategy_cols["noringer"] = list()
    for modelpath, criterion in product(modelpaths, CRITERIA_CONF_NAMES.keys()):
        conf_name = CRITERIA_CONF_NAMES[criterion]
        confpath = os.path.join(modelpath, conf_name)
        env = ROOT.TEnv(confpath)
        ringer_version = env.GetValue("__version__", "")
        if ringer_version == "should_be_filled":
            raise ValueError(f"The model from {modelpath} does not have a version. Please fill it. Version found: {ringer_version}")
        ringer_name = f"ringer_{ringer_version}"
        strat_criterion = f"{ringer_name}_{criterion}"
        simulation_logger.info(f"Building decorator for {confpath}. Version: {ringer_version}")
        decorator = RingerDecorator(strat_criterion, confpath, ringer_generators[ringer_version])
        decorators.append(decorator)
        strategy_cols[ringer_name].append(strat_criterion)
        strategy_cols[ringer_name].append(strat_criterion + "_output")
    
    return decorators, strategy_cols

def build_chains(chain_names: List[str], strategy_cols):
    chains = list()
    step_chain_names = list()
    for chain_name, strategy in product(chain_names, strategy_cols.keys()):
        spliited_chain_name = chain_name.split("_")
        criterion = spliited_chain_name[1].replace("lh", "")
        step_chain_name = f"HLT_{chain_name.format(strategy=strategy)}"
        step_chain_names.append(step_chain_name)
        strategy_cols[strategy].append(step_chain_name)
        for trigger_step in TRIG_STEPS:
            if trigger_step != "HLT":
                lower_chain = step_chain_name.replace("HLT", trigger_step)
                step_chain_names.append(lower_chain)
                strategy_cols[strategy].append(lower_chain)
        energy = int(spliited_chain_name[0][1:])
        l1seed = L1SEEDS_PER_ENERGY[energy]
        l2calo_column = f"{strategy}_{criterion}"
        simulation_logger.info(f"Building chain: {step_chain_name} model: {l2calo_column}")
        if strategy == "noringer":
            chain = Chain(step_chain_name, L1Seed=l1seed)
        else:
            chain = Chain(step_chain_name, L1Seed=l1seed, l2calo_column=l2calo_column)
        chains.append(chain)
        
    return chains


def run_simulation(dataset: str, dev: bool, ets: List[int], etas: List[int],
                   chains, decorators, strategy_cols):
    dataset_loader = NamedDatasetLoader(dataset)
    data_df_dtypes = dataset_loader.get_data_df_dtypes()
    data_cols = data_df_dtypes.index
    load_cols = [col for col in data_cols if col not in DROP_COLS]
    last_strat = None
    ibins = product([4], [4]) if dev else product(ets, etas)
    last_bin = 0 if dev else len(ets)*len(etas) - 1
    for i, (et_bin_idx, eta_bin_idx) in enumerate(ibins):
        start_msg = f"et {et_bin_idx} eta {eta_bin_idx}: "
        simulation_logger.info(start_msg + "Loading data_df")
        data = dataset_loader.load_data_df(load_cols, et_bin_idx, eta_bin_idx)
        simulation_logger.info(start_msg + "Simulating")
        for decorator in decorators:
            decorator.apply(data)
        for chain in chains:
            chain.apply(data)

        for strategy in strategy_cols.keys():
            # Saves the id for future joining if necessary
            selection_cols = ["id", "region_id"] + strategy_cols[strategy]
            selected_data = data[selection_cols]
            simulation_logger.info(start_msg + f"Dumping {strategy}")
            dataset_loader.dump_strategy_df(selected_data, strategy, et_bin_idx, eta_bin_idx)

            
def simulate(dataset: str, modelpaths: List[str], cutbased: bool, 
        chain_names: List[str], dev: bool, ets: List[int], etas: List[int], **kwargs):

    simulation_logger.info("Building decorators")
    decorators, strategy_cols = build_decorators(modelpaths, cutbased)
    simulation_logger.info("Building chains")
    chains = build_chains(chain_names, strategy_cols)
    run_simulation(dataset, dev, ets, etas, chains, decorators, strategy_cols)


if __name__ == "__main__":
    args = parse_args()
    simulation_logger = get_logger("simulate_chains", file=args["log"])
    simulation_logger.info("Script start")
    simulation_logger.info("Parsed args")
    for key, value in args.items():
        simulation_logger.info(f"{key}: {value}")
    simulate(**args)
    simulation_logger.info("Finished")
