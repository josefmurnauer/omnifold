import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib import rc
from numpy import inf
import itertools
from os import listdir
import uproot
import awkward as ak
import matplotlib as mpl
from datetime import datetime
import subprocess
import os
import tensorflow as tf
import sys
import importlib
from my_utils import *
sys.path.append('/scratch/mjosef/Unfolding/omnifold')
from omnifold import DataLoader, MLP, SetStyle, HistRoutine, net, PET, omnifold
import test_omnifold
import PET
from IPython.display import Image

GPU_number = "6"

os.environ['CUDA_VISIBLE_DEVICES']=GPU_number # GPU Server Available: 0,1,2,3,4,5,6,7
has_gpu=True

from pickle import dump

print(tf.config.list_physical_devices())

if has_gpu :
    os.environ['CUDA_VISIBLE_DEVICES']=GPU_number
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

#############################
# Config #
omnifold_name = "closure_pythia_WWbbvsHerwig_WWbb-15_iter-2M_evts_12_BS-64"
weight_folder = '/scratch/mjosef/Unfolding/omnifold/weights_pythia_WWbbvsHerwig_WWbb_iterations'
number_of_events = 2_000_000
iterations = 15
batch_size = 64

#############################


pythia_df = pd.read_pickle('/scratch/mjosef/OMNIFOLD_Tutorial/datasets/WbWb_files/bulk_region/df_pythia_ttbar_singletop_DR.pkl')
herwig_df = pd.read_pickle('/scratch/mjosef/OMNIFOLD_Tutorial/datasets/WbWb_files/bulk_region/df_herwig_ttbar_singletop_DR.pkl')
print("Loaded root files")

pythia_train, pythia_test = subset(pythia_df, number_of_events, train_test=True)
herwig_subset = subset(herwig_df, number_of_events, train_test=False)
print("Made subsets")
pythia_reco_train, pythia_truth_train = MC_data_shaper(pythia_train)
pythia_reco_test, pythia_truth_test = MC_data_shaper(pythia_test)
herwig_reco, herwig_truth = MC_data_shaper(herwig_subset)
#pythia_reco_train, pythia_truth_train = np.concatenate([MC_data_shaper(pythia_train)[0], pairwise(MC_data_shaper(pythia_train)[0])], axis=1), np.concatenate([MC_data_shaper(pythia_train)[1], pairwise(MC_data_shaper(pythia_train)[1])], axis=1)
#pythia_reco_test, pythia_truth_test = np.concatenate([MC_data_shaper(pythia_test)[0], pairwise(MC_data_shaper(pythia_test)[0])], axis=1), np.concatenate([MC_data_shaper(pythia_test)[1], pairwise(MC_data_shaper(pythia_test)[1])], axis=1)
#herwig_reco, herwig_truth = np.concatenate([MC_data_shaper(herwig_subset)[0], pairwise(MC_data_shaper(herwig_subset)[0])], axis=1), np.concatenate([MC_data_shaper(herwig_subset)[1], pairwise(MC_data_shaper(herwig_subset)[1])], axis=1)
print("Prepared arrays")

jet_scaler = JetScaler(mask_value=0.0)
jet_scaler.fit(pythia_reco_train)
X_pythia_reco_scaled = jet_scaler.transform(pythia_reco_train)
X_pythia_truth_scaled = jet_scaler.transform(pythia_truth_train)
X_herwig_reco_scaled = jet_scaler.transform(herwig_reco)
X_herwig_truth_scaled = jet_scaler.transform(herwig_truth)
Y_truth_scaled = jet_scaler.transform(pythia_truth_test)
Y_reco_scaled = jet_scaler.transform(pythia_reco_test)
print("Scaled data")

herwig_loader = DataLoader(reco = X_herwig_reco_scaled, gen = X_herwig_truth_scaled, weight = herwig_subset['eventWeight'],
                           pass_reco = herwig_subset['pass_reco'], pass_gen = herwig_subset['pass_particle'], normalize=False)

pythia_loader = DataLoader(reco = X_pythia_reco_scaled, gen = X_pythia_truth_scaled, weight = pythia_train['eventWeight'],
                           pass_reco = pythia_train['pass_reco'], pass_gen = pythia_train['pass_particle'], normalize=False)

testset_loader = DataLoader(reco = pythia_reco_test, gen = pythia_truth_test, weight = pythia_test['eventWeight'],
                           pass_reco = pythia_test['pass_reco'], pass_gen = pythia_test['pass_particle'], normalize=False)
print("Initalized Dataloader")

ndim = 4 # 4 features: pt, eta, phi, mass
npart = 12 # 14 particles: l1, b1-4, j1-6, met

model1 = PET.PET(num_feat = 4, num_part = 12, local=False)
model2 = PET.PET(num_feat = 4, num_part = 12, local=False)
print("Created model")

omnifold = test_omnifold.MultiFold(
    omnifold_name,
    model1, # model_reco
    model2, # model_gen
    herwig_loader, # data
    pythia_loader, # MC
    batch_size = batch_size,
    niter = iterations,  #Number of Iterations                                                                                                                                                                                                  
    epochs=150,     
    weights_folder = weight_folder,
    verbose = True,
    early_stop=5,
    lr = 5e-6,
)

omnifold.Unfold()
unfolded_weights  = omnifold.reweight(Y_truth_scaled,omnifold.model2,batch_size=1000)
##np.save(weight_folder + f"/{omnifold_name}-unfolded_weights.npy", unfolded_weights)
#print("Saved unfolded weights")


print("Started plotting ...")
# --- Create figure with 3x4 subplots ---
fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

for i, pname in enumerate(particles[:npart]):
    # Prepare data
    data_dict = {
        'Truth Distribution': herwig_truth[:, i, 0][herwig_truth[:, i, 0] != 0],
        'Generated MC': pythia_truth_test[:, i, 0][pythia_truth_test[:, i, 0] != 0],
        'Unfolded Data': pythia_truth_test[:, i, 0][pythia_truth_test[:, i, 0] != 0],
    }
    weight_dict = {
        'Truth Distribution': herwig_subset['eventWeight'][herwig_truth[:, i, 0] != 0],
        'Generated MC': pythia_test['eventWeight'][pythia_truth_test[:, i, 0] != 0],
        'Unfolded Data': (unfolded_weights * pythia_test['eventWeight'])[pythia_truth_test[:, i, 0] != 0],
    }
    
    # Create a small inset for ratio
    # Here we just stack main+ratio manually inside the same subplot
    # Define height ratios
    gs = axes[i].get_gridspec()
    for ax in axes[i].get_shared_x_axes().get_siblings(axes[i]):
        ax.remove()
    
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    sub_gs = GridSpecFromSubplotSpec(2, 1, height_ratios=[3,1], subplot_spec=gs[i], hspace=0.05)
    ax_main = fig.add_subplot(sub_gs[0])
    ax_ratio = fig.add_subplot(sub_gs[1], sharex=ax_main)
    
    plot_pt_subplot(ax_main, ax_ratio, data_dict, weight_dict, pt_binning[pname])
    
    ax_main.set_title(pname)

plt.tight_layout()
plt.savefig(f"{weight_folder}/{omnifold_name}_results.png", dpi=150)
plt.close()
print(f"📊 Saved 3×4 grid plot")
print("Everything done")