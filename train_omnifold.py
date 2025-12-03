import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
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
from omnifold import DataLoader, MLP, SetStyle, HistRoutine, net, PET, omnifold_slicing
import omnifold_routine
import PET

GPU_number = "7"

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
omnifold_name = "closure_pythia_ttbarvsHerwig_WWbb-10_iter-2M_evts_14"
weight_folder = '/scratch/mjosef/Unfolding/omnifold/weights_pythia_ttbarvsHerwig_WWbb_iterations'
number_of_events = 2_000_000
iterations = 10
batch_size = 128
is_DATA = False

#############################


pythia_df = pd.read_pickle('/scratch/mjosef/OMNIFOLD_Tutorial/datasets/WbWb_files/bulk_region/df_pythia_ttbar.pkl')
herwig_df = pd.read_pickle('/scratch/mjosef/OMNIFOLD_Tutorial/datasets/WbWb_files/bulk_region/df_herwig_ttbar_singletop_DR.pkl')
#data_df = pd.read_pickle('/scratch/mjosef/OMNIFOLD_Tutorial/datasets/WbWb_files/bulk_region/df_data.pkl')
#data_df.fillna(0.0, inplace=True)
TUnfold_incl_path = '/scratch/mjosef/OMNIFOLD_Tutorial/datasets/WbWb_files/bulk_region/unfolding_SR_Bulk_Final_l_4j_incl_TUnfoldStandalone_OptionA_data_nonClosureAlternative.root'
TUnfold_incl_file = uproot.open(TUnfold_incl_path)
print("Loaded root files")

pythia_train, pythia_test = subset(pythia_df, number_of_events, train_test=True)
herwig_subset = subset(herwig_df, number_of_events, train_test=False)
print("Made subsets")
#pythia_reco_train, pythia_truth_train = MC_data_shaper(pythia_train)
#pythia_reco_test, pythia_truth_test = MC_data_shaper(pythia_test)
#herwig_reco, herwig_truth = MC_data_shaper(herwig_subset)
#data_array = DATA_shaper(data_df)
pythia_reco_train, pythia_truth_train = np.concatenate([MC_data_shaper(pythia_train)[0], pairwise(MC_data_shaper(pythia_train)[0])], axis=1), np.concatenate([MC_data_shaper(pythia_train)[1], pairwise(MC_data_shaper(pythia_train)[1])], axis=1)
pythia_reco_test, pythia_truth_test = np.concatenate([MC_data_shaper(pythia_test)[0], pairwise(MC_data_shaper(pythia_test)[0])], axis=1), np.concatenate([MC_data_shaper(pythia_test)[1], pairwise(MC_data_shaper(pythia_test)[1])], axis=1)
herwig_reco, herwig_truth = np.concatenate([MC_data_shaper(herwig_subset)[0], pairwise(MC_data_shaper(herwig_subset)[0])], axis=1), np.concatenate([MC_data_shaper(herwig_subset)[1], pairwise(MC_data_shaper(herwig_subset)[1])], axis=1)
print("Prepared arrays")

jet_scaler = JetScaler(mask_value=0.0)
jet_scaler.fit(pythia_reco_train)
X_pythia_reco_scaled = jet_scaler.transform(pythia_reco_train)
X_pythia_truth_scaled = jet_scaler.transform(pythia_truth_train)
X_herwig_reco_scaled = jet_scaler.transform(herwig_reco)
X_herwig_truth_scaled = jet_scaler.transform(herwig_truth)
#X_data_scaled = jet_scaler.transform(data_array)
Y_truth_scaled = jet_scaler.transform(pythia_truth_test)
Y_reco_scaled = jet_scaler.transform(pythia_reco_test)
print("Scaled data")

herwig_loader = DataLoader(reco = X_herwig_reco_scaled, gen = X_herwig_truth_scaled, weight = herwig_subset['eventWeight'],
                           pass_reco = herwig_subset['pass_reco'], pass_gen = herwig_subset['pass_particle'], normalize=False)

pythia_loader = DataLoader(reco = X_pythia_reco_scaled, gen = X_pythia_truth_scaled, weight = pythia_train['eventWeight'],
                           pass_reco = pythia_train['pass_reco'], pass_gen = pythia_train['pass_particle'], normalize=False)

testset_loader = DataLoader(reco = pythia_reco_test, gen = pythia_truth_test, weight = pythia_test['eventWeight'],
                           pass_reco = pythia_test['pass_reco'], pass_gen = pythia_test['pass_particle'], normalize=False)

#data_loader = DataLoader(reco = X_data_scaled, weight = data_df['eventWeight'], normalize=False)
print("Initalized Dataloader")

ndim = 4 # 4 features: pt, eta, phi, mass
npart = 14 # 14 particles: l1, b1-4, j1-6, met

model1 = PET.PET(num_feat = 4, num_part = 14, local=True, K=4)
model2 = PET.PET(num_feat = 4, num_part = 14, local=True, K=4)
#model1 = build_transformer_model()
#model2 = build_transformer_model()
print("Created model")

omnifold = omnifold_routine.MultiFold(
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
    lr = 5e-5,
)

omnifold.Unfold()
unfolded_weights  = omnifold.reweight(Y_truth_scaled,omnifold.model2,batch_size=1000)
#np.save(weight_folder + f"/{omnifold_name}-unfolded_weights.npy", unfolded_weights)
#print("Saved unfolded weights")


print("Started plotting ...")

if is_DATA==False:
    if npart ==12:
        ## --- Create figure with 3x4 subplots ---
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, pname in enumerate(particles):
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

    elif npart==14:
        ## --- Create figure with 4x4 subplots ---
        fig = plt.figure(figsize=(20, 12))
        outer_gs = fig.add_gridspec(4, 4, wspace=0.3, hspace=0.3)
        
        for i, pname in enumerate(particles):
            from matplotlib.gridspec import GridSpecFromSubplotSpec
            sub_gs = GridSpecFromSubplotSpec(2, 1, height_ratios=[3,1], subplot_spec=outer_gs[i], hspace=0.05)
            ax_main = fig.add_subplot(sub_gs[0])
            ax_ratio = fig.add_subplot(sub_gs[1], sharex=ax_main)

            if i < len(particles) - 2:
                idx = (i, 0)
            elif i == len(particles) - 2:
                idx = (-2, 3)
            elif i == len(particles) - 1:
                idx = (-1, 3)

            data_dict = {
                'Truth Distribution': herwig_truth[:, idx[0], idx[1]][herwig_truth[:, idx[0], idx[1]] != 0],
                'Generated MC': pythia_truth_test[:, idx[0], idx[1]][pythia_truth_test[:, idx[0], idx[1]] != 0],
                'Unfolded Data': pythia_truth_test[:, idx[0], idx[1]][pythia_truth_test[:, idx[0], idx[1]] != 0],
            }
            weight_dict = {
                'Truth Distribution': herwig_subset['eventWeight'][herwig_truth[:, idx[0], idx[1]] != 0],
                'Generated MC': pythia_test['eventWeight'][pythia_truth_test[:, idx[0], idx[1]] != 0],
                'Unfolded Data': (unfolded_weights * pythia_test['eventWeight'])[pythia_truth_test[:, idx[0], idx[1]] != 0],
            }

            plot_pt_subplot(ax_main, ax_ratio, data_dict, weight_dict, pt_binning[pname])
            ax_main.set_xlabel(pname, fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{weight_folder}/{omnifold_name}_results.png", dpi=150)
        plt.close()
        print(f"Saved 4×4 grid plot")

else:
    fig = plot_omnifold_vs_tunfold(
    unfolded_weights=unfolded_weights,
    TUnfold_incl_file=TUnfold_incl_file,
    pythia_test=pythia_test,
    pythia_truth_test=pythia_truth_test,
    luminosity=3244.54 + 33402.2 + 44630.6 + 58791.6)

    fig.savefig(f"{weight_folder}/{omnifold_name}_results.png", dpi=150)
print("Everything done")