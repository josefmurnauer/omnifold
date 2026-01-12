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

GPU_number = "3"

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
#number_of_events_pythia = 25_000_000
#number_of_events_herwig = 17_000_000
#number_of_events = number_of_events_pythia+number_of_events_herwig
#particle_index = 1
number_of_events = 5_000_000
omnifold_name = f"14_part_{number_of_events}_pythia_WbWb_background"
weight_folder = '/scratch/mjosef/Unfolding/omnifold/data/'
iterations = 2
batch_size = 128
is_DATA = False

#############################



pythia_df = pd.read_pickle('/ptmp/mpp/mjosef/data_files/WbWb_files/bulk_region/WbWb_background_df.pkl')
#pythia_df = pd.read_pickle('/ptmp/mpp/mjosef/data_files/WbWb_files/bulk_region/df_pythia_ttbar.pkl')
#pythia_df = pd.read_pickle('/ptmp/mpp/mjosef/data_files/WbWb_files/bulk_region/df_pythia_ttbar_singletop_DR.pkl')
#herwig_df = pd.read_pickle('/ptmp/mpp/mjosef/data_files/WbWb_files/bulk_region/df_herwig_ttbar_singletop_DR.pkl')
data_df = pd.read_pickle('/ptmp/mpp/mjosef/data_files/WbWb_files/bulk_region/df_data.pkl')
data_df.fillna(0.0, inplace=True)
TUnfold_incl_path = '/ptmp/mpp/mjosef/data_files/WbWb_files/bulk_region/unfolding_SR_Bulk_Final_l_4j_incl_TUnfoldStandalone_OptionA_data_nonClosureAlternative.root'
#TUnfold_incl_file = uproot.open(TUnfold_incl_path)
print("Loaded root files")

pythia_train, pythia_test = subset(pythia_df[pythia_df['pass_matched']==1], number_of_events, train_test=True)
#data_subset = subset(data_df, 4_300_000, train_test=False)
data_df['eventWeight'] = data_df['eventWeight'] * 0.75
#herwig_subset = subset(herwig_df[herwig_df['pass_matched']==1], number_of_events, train_test=False)
print("Made subsets")
#pythia_reco_train, pythia_truth_train = MC_data_shaper(pythia_train)
#pythia_reco_test, pythia_truth_test = MC_data_shaper(pythia_test)
#herwig_reco, herwig_truth = MC_data_shaper(herwig_subset)
data_array = np.concatenate([DATA_shaper(data_df), pairwise(DATA_shaper(data_df))], axis=1)
pythia_reco_train, pythia_truth_train = np.concatenate([MC_data_shaper(pythia_train)[0], pairwise(MC_data_shaper(pythia_train)[0])], axis=1), np.concatenate([MC_data_shaper(pythia_train)[1], pairwise(MC_data_shaper(pythia_train)[1])], axis=1)
pythia_reco_test, pythia_truth_test = np.concatenate([MC_data_shaper(pythia_test)[0], pairwise(MC_data_shaper(pythia_test)[0])], axis=1), np.concatenate([MC_data_shaper(pythia_test)[1], pairwise(MC_data_shaper(pythia_test)[1])], axis=1)
#herwig_reco, herwig_truth = np.concatenate([MC_data_shaper(herwig_subset)[0], pairwise(MC_data_shaper(herwig_subset)[0])], axis=1), np.concatenate([MC_data_shaper(herwig_subset)[1], pairwise(MC_data_shaper(herwig_subset)[1])], axis=1)
print("Prepared arrays")

jet_scaler = JetScaler(mask_value=0.0)
jet_scaler.fit(pythia_reco_train)
X_pythia_reco_scaled = jet_scaler.transform(pythia_reco_train)
X_pythia_truth_scaled = jet_scaler.transform(pythia_truth_train)
#X_herwig_reco_scaled = jet_scaler.transform(herwig_reco)
#X_herwig_truth_scaled = jet_scaler.transform(herwig_truth)
X_data_scaled = jet_scaler.transform(data_array)
Y_truth_scaled = jet_scaler.transform(pythia_truth_test)
Y_reco_scaled = jet_scaler.transform(pythia_reco_test)
print("Scaled data")

#herwig_loader = DataLoader(reco = X_herwig_reco_scaled, gen = X_herwig_truth_scaled, weight = herwig_subset['eventWeight'], normalize=False)

pythia_loader = DataLoader(reco = X_pythia_reco_scaled, gen = X_pythia_truth_scaled, weight = pythia_train['eventWeight'],normalize=False)

testset_loader = DataLoader(reco = pythia_reco_test, gen = pythia_truth_test, weight = pythia_test['eventWeight'],normalize=False)

data_loader = DataLoader(reco = X_data_scaled, weight = data_df['eventWeight'], normalize=False)
print("Initalized Dataloader")

ndim = 4 # 4 features: pt, eta, phi, mass
npart = 14 # 14 particles: l1, b1-4, j1-6, met

model1 = PET.PET(num_feat = 4, num_part = 14, local=True, K=4)
model2 = PET.PET(num_feat = 4, num_part = 14, local=True, K=4)
#model1 = MLP(1)
#model2 = MLP(1)
print("Created model")

omnifold = omnifold_routine.MultiFold(
    omnifold_name,
    model1, # model_reco
    model2, # model_gen
    data_loader, # data
    pythia_loader, # MC
    batch_size = batch_size,
    niter = iterations,  #Number of Iterations                                                                                                                                                                                                  
    epochs=100,     
    weights_folder = weight_folder, 
    verbose = True,
    early_stop=5,
    lr = 5e-4,
)

omnifold.Unfold()
print("Unfolded data")
sys.exit() 
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

            if i < len(particles):
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

    elif npart==1:
        def plot_pt_subplot(ax_main, ax_ratio, data_dict, weights_dict, bins, logx=False):
            if logx==True:
                vals = np.concatenate([
                    herwig_reco[:, particle_index, 0],
                    pythia_reco_test[:, particle_index, 0],
                    pythia_truth_test[:, particle_index, 0]])
                vals = vals[vals > 0]

                xmin = np.percentile(vals, 0.1) 
                xmax = np.percentile(vals, 99.9)
                pname = particles[particle_index]
                bins = make_log_bins(xmin, xmax, n_per_decade=10)
            x_centers = 0.5 * (bins[1:] + bins[:-1])
            ref_name = next(iter(data_dict.keys()))
            
            # Reference histogram
            ref_counts, _ = np.histogram(data_dict[ref_name], bins=bins, weights=weights_dict[ref_name])
            
            # Omnifold uncertainties: sum of weights squared per bin
            sum_w2_dict = {}
            err_dict = {}
            for label, values in data_dict.items():
                w2 = weights_dict[label]**2
                sum_w2, _ = np.histogram(values, bins=bins, weights=w2)
                sum_w2_dict[label] = sum_w2
                err_dict[label] = np.sqrt(sum_w2)

            maxy = 0
            colors = ['black', '#3f90da', '#ffa90e']  # black, blue, orange
            err_colors = [None, '#3f90da', '#ffa90e']

            for i, (label, values) in enumerate(data_dict.items()):
                counts, _ = np.histogram(values, bins=bins, weights=weights_dict[label])
                maxy = max(maxy, counts.max())

                # step histogram
                ax_main.step(bins[:-1], counts, where='post', color=colors[i], label=label)

                # error band for blue/orange curves (Omnifold)
                if i > 0:
                    ratio = np.divide(counts, ref_counts, out=np.zeros_like(counts), where=ref_counts != 0)
                    ratio_err = np.zeros_like(ratio)
                    nonzero = (ref_counts != 0)
                    ratio_err[nonzero] = ratio[nonzero] * np.sqrt(
                        (err_dict[label][nonzero] / counts[nonzero])**2 +
                        (err_dict[ref_name][nonzero] / ref_counts[nonzero])**2
                    )

                    ax_ratio.step(x_centers, ratio, where='mid', color=colors[i], label=label)
                    ax_ratio.fill_between(
                        x_centers,
                        ratio - ratio_err,
                        ratio + ratio_err,
                        step='mid',
                        alpha=0.3,
                        color=err_colors[i],
                    )

            #ax_main.set_ylim(0, 1.3 * maxy)
            ax_main.set_yscale('log')
            ax_main.grid(True, ls='--', alpha=0.5)
            ax_main.legend(fontsize=9)

            ax_ratio.axhline(1.0, color='r', ls='--')
            ax_ratio.set_ylim(0.9, 1.1)
            ax_ratio.grid(True, ls='--', alpha=0.5)
            ax_ratio.set_xlabel("$p_T$ [GeV]")



        def load_updated_weights(it, stepn, n_ensemble=1):
            """
            Load the weights for a given iteration and step
            """
            model = omnifold.model1 if stepn==1 else omnifold.model2
            in_data = Y_reco_scaled[:,particle_index,0] if stepn==1 else Y_truth_scaled[:,particle_index,0]
            #particle_name = particles[particle_index]
            model_name = f"{weight_folder}/OmniFold_{omnifold_name}_iter{it}_step{stepn}.weights.h5"
            model.build(input_shape=in_data.shape)
            model.load_weights(model_name)
            
            f = omnifold_routine.expit(model.predict(in_data, batch_size=1000))
            w = f / (1 - f)
            w = np.nan_to_num(w[:,0], posinf=1)
            return w


        def plot_all_iterations_updated(n_iterations, n_ensemble=1, logx=False):

            pname = particles[particle_index]
            fig = plt.figure(figsize=(13, 3.2*n_iterations))
            gs = GridSpec(n_iterations, 2, figure=fig, wspace=0.25, hspace=0.35)

            # --- Loop over iterations ---
            for it in range(n_iterations):

                # --- Replay weights up to iteration it ---
                weights_pull = np.ones_like(testset_loader.weight)
                weights_push = np.ones_like(testset_loader.weight)

                for it_replay in range(it+1):
                    # Step 1
                    w1 = load_updated_weights(it_replay, 1, n_ensemble)
                    weights_pull = weights_push * w1
                    # Step 2
                    w2 = load_updated_weights(it_replay, 2, n_ensemble)
                    weights_push = w2

                for stepn in [1, 2]:

                    sub = GridSpecFromSubplotSpec(
                        2, 1,
                        subplot_spec=gs[it, stepn-1],
                        height_ratios=[3,1],
                        hspace=0.05
                    )
                    ax_main = fig.add_subplot(sub[0])
                    ax_ratio = fig.add_subplot(sub[1], sharex=ax_main)

                    if stepn==1:
                        data_dict = {
                            "Pseudo-Data Reco": herwig_reco[:, particle_index, 0],
                            "MC Reco": pythia_reco_test[:, particle_index, 0],
                            "Reweighted": pythia_reco_test[:, particle_index, 0]
                        }
                        weights_dict = {
                            "Pseudo-Data Reco": herwig_loader.weight,
                            "MC Reco": testset_loader.weight,
                            "Reweighted": testset_loader.weight * weights_pull
                        }
                    else:
                        data_dict = {
                            "Truth": herwig_truth[:, particle_index, 0],
                            "Generated": pythia_truth_test[:, particle_index, 0],
                            "Unfolded": pythia_truth_test[:, particle_index, 0]
                        }
                        weights_dict = {
                            "Truth": herwig_loader.weight,
                            "Generated": testset_loader.weight,
                            "Unfolded": testset_loader.weight * weights_push
                        }

                    plot_pt_subplot(ax_main, ax_ratio, data_dict, weights_dict, pt_binning[pname], logx=logx)
                    ax_main.set_title(f"Iteration {it+1}, Step {stepn}")
                    fig.suptitle(f"{particles[particle_index]}"+"_pT", fontsize=18)
                plt.tight_layout()
        plt.savefig(f"{weight_folder}/{omnifold_name}_results.png", dpi=150)
        plt.close()
        print(f"Saved 2x3 grid plot")

else:
    fig = plot_omnifold_vs_tunfold(
    unfolded_weights=unfolded_weights,
    TUnfold_incl_file=TUnfold_incl_file,
    pythia_test=pythia_test,
    pythia_truth_test=pythia_truth_test,
    luminosity=3244.54 + 33402.2 + 44630.6 + 58791.6)

    fig.savefig(f"{weight_folder}/{omnifold_name}_results.png", dpi=150)
print("Everything done")
