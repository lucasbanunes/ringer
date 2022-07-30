import os
from kolmov import crossval_table, get_color_fader, fit_table
from saphyra.core import ReferenceReader
import numpy as np
import pandas as pd
import collections
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy
from argparse import ArgumentParser

def create_op_dict(op):
    d = {
              op+'_pd_ref'    : "reference/"+op+"_cutbased/pd_ref#0",
              op+'_fa_ref'    : "reference/"+op+"_cutbased/fa_ref#0",
              op+'_sp_ref'    : "reference/"+op+"_cutbased/sp_ref",
              op+'_pd_val'    : "reference/"+op+"_cutbased/pd_val#0",
              op+'_fa_val'    : "reference/"+op+"_cutbased/fa_val#0",
              op+'_sp_val'    : "reference/"+op+"_cutbased/sp_val",
              op+'_pd_op'     : "reference/"+op+"_cutbased/pd_op#0",
              op+'_fa_op'     : "reference/"+op+"_cutbased/fa_op#0",
              op+'_sp_op'     : "reference/"+op+"_cutbased/sp_op",

              # Counts
              op+'_pd_ref_passed'    : "reference/"+op+"_cutbased/pd_ref#1",
              op+'_fa_ref_passed'    : "reference/"+op+"_cutbased/fa_ref#1",
              op+'_pd_ref_total'     : "reference/"+op+"_cutbased/pd_ref#2",
              op+'_fa_ref_total'     : "reference/"+op+"_cutbased/fa_ref#2",
              op+'_pd_val_passed'    : "reference/"+op+"_cutbased/pd_val#1",
              op+'_fa_val_passed'    : "reference/"+op+"_cutbased/fa_val#1",
              op+'_pd_val_total'     : "reference/"+op+"_cutbased/pd_val#2",
              op+'_fa_val_total'     : "reference/"+op+"_cutbased/fa_val#2",
              op+'_pd_op_passed'     : "reference/"+op+"_cutbased/pd_op#1",
              op+'_fa_op_passed'     : "reference/"+op+"_cutbased/fa_op#1",
              op+'_pd_op_total'      : "reference/"+op+"_cutbased/pd_op#2",
              op+'_fa_op_total'      : "reference/"+op+"_cutbased/fa_op#2",
    }
    return d

def generator( path ):
    def norm1( data ):
        norms = np.abs( data.sum(axis=1) )
        norms[norms==0] = 1
        return data/norms[:,None]
    from Gaugi import load
    d = load(path)
    feature_names = d['features'].tolist()

    # How many events?
    n = d['data'].shape[0]
    
    # extract rings
    data_rings = norm1(d['data'][:,1:101])
    target = d['target']
    avgmu = d['data'][:,0]
    
    return [data_rings], target, avgmu


parser = ArgumentParser(
    prog='decision threshold fit',
    description='Script for fitting a decision threshold w.r.t collision pileup for the neural ringer',
)
parser.add_argument('--dataset', '-d', nargs=1, type=str, required=True)
parser.add_argument('--output', '-o', nargs=1, type=str, required=True)
parser.add_argument('--model', '-m', nargs=1, type=str, required=True)
parser.add_argument('--extra-bin', '-eb', action='store_true', dest='extra_bin')
parser.add_argument('--model-version', '-mv', nargs=1, type=str, dest='model_version', default='vX')

args = parser.parse_args()
etbins = [15, 20, 30, 40, 50, 1000000]
n_ets = len(etbins)-1
etabins = [0.0, 0.8, 1.37, 1.54, 2.37, 2.50]
n_etas = len(etabins)-1
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

tuned_info = collections.OrderedDict( {
              # validation
              "max_sp_val"      : 'summary/max_sp_val',
              "max_sp_pd_val"   : 'summary/max_sp_pd_val#0',
              "max_sp_fa_val"   : 'summary/max_sp_fa_val#0',
              # Operation
              "max_sp_op"       : 'summary/max_sp_op',
              "max_sp_pd_op"    : 'summary/max_sp_pd_op#0',
              "max_sp_fa_op"    : 'summary/max_sp_fa_op#0',
              } )

tuned_info.update(create_op_dict('tight'))
tuned_info.update(create_op_dict('medium'))
tuned_info.update(create_op_dict('loose'))
tuned_info.update(create_op_dict('vloose'))

# Reading tunings and selecting the best ones
cv  = crossval_table( tuned_info, etbins = etbins , etabins = etabins )
cv.fill(args.model_path, args.model_version)
cv.table().to_csv(os.path.join(args.output_dir, 'cross_val_table.csv'))
best_inits = cv.filter_inits("max_sp_val")  #Selects best the best modelo from each init from each fold
best_inits.to_csv(os.path.join(args.output_dir, 'best_inits_table.csv'))
best_sorts = cv.filter_sorts( best_inits , 'max_sp_op') #Selects the best init from all folds
best_sorts.to_csv(os.path.join(args.output_dir, 'best_sorts.csv'))
best_models = cv.get_best_models(best_sorts, remove_last=True)  #Loads the best models and removes the activation layer

# Loads the reference files
homepath = os.path.expanduser('~')
datapath = os.path.join(homepath, 'data', args.dataset)
refpath = os.path.join(datapath, 'references')
ref_filepath = os.path.join(refpath, args.dataset + '_et{ET}_eta{ETA}.ref.pic.gz')
ref_filepaths = [[ ref_filepath.format(ET=et,ETA=eta) for eta in range(n_etas)] for et in range(n_ets)]
ref_matrix = [[ {} for eta in range(n_etas)] for et in range(n_ets)]
references = ['tight_cutbased', 'medium_cutbased' , 'loose_cutbased', 'vloose_cutbased']
for et_bin in range(n_ets):
    for eta_bin in range(n_etas):
        for name in references:
            refObj = ReferenceReader().load(ref_filepaths[et_bin][eta_bin])
            _pd = refObj.getSgnPassed(name)/refObj.getSgnTotal(name)
            fa = refObj.getBkgPassed(name)/refObj.getBkgTotal(name)
            ref_matrix[et_bin][eta_bin][name] = {'pd':_pd, 'fa':fa, 'pd_epsilon':0}

# Fitting thresholds
fit_etbins = etbins.copy()
fit_etabins = etabins.copy()
if args.extra_bin:
    fit_etbins = fit_etbins.insert(-2, 100)
data_filepath = os.path.join(datapath, args.dataset + '_et{ET}_eta{ETA}.npz')
paths = [[ data_filepath.format(ET=et,ETA=eta) for eta in range(n_etas)] for et in range(n_ets)]
ct  = fit_table(generator, fit_etbins , fit_etabins, 0.02, 0.5, 16, 60, xmin_percentage=0.05, xmax_percentage=99.95)
fit_name = f'correction_{args.model_version}_{args.dataset}'
ct.fill(paths, best_models, ref_matrix, fit_name)
fit_table = ct.table()
fit_table.to_csv(os.path.join(args.output_dir, 'threshold_table.csv'))
ct.dump_beamer_table(ct.table(), best_models, f'{args.dataset} {args.model_version} tuning', 
    fit_name + '.pdf')

# Exporting models
model_name_format = 'data17_13TeV_EGAM1_probes_lhmedium_EGAM7_vetolhvloose.model_{args.model_version}.electron{op}.et%d_eta%d'
config_name_format = 'ElectronRinger{op}TriggerConfig.conf'
for idx, op in enumerate(['Tight','Medium','Loose','VeryLoose']):
    ct.export(best_models, 
              model_name_format.format(op=op), 
              config_name_format.format(op=op), 
              references[idx], 
              to_onnx='new')