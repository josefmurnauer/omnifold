import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib import rc
import matplotlib as mpl

def MC_data_shaper(df):

    particle_names = ['l1', 'b1', 'b2', 'b3', 'b4', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']

    # Build reco and truth arrays
    reco_pts  = [df[f'pt{p}']   for p in particle_names]
    reco_etas = [df[f'eta{p}']  for p in particle_names]
    reco_phis = [df[f'phi{p}']  for p in particle_names]
    
    # Masses: lepton = 0, b-jets = mb1–4, jets = mj1–6
    reco_masses = [np.zeros_like(df['ptl1'])] + \
                  [df[f'mb{i}'] for i in range(1, 5)] + \
                  [df[f'mj{i}'] for i in range(1, 7)]
    # Stack into (12 particles, n_events, 4)
    reco_features = np.stack([reco_pts, reco_etas, reco_phis, reco_masses], axis=-1)
    reco_features = reco_features.transpose(1, 0, 2)  # → (n_events, 12, 4)

    # Neutrino (reco): met, eta=0, phi=metphi, mass=0
    met_pt = df['met']
    met_phi = df['metphi']
    zeros = np.zeros_like(met_pt)
    neutrino_reco = np.stack([met_pt, zeros, met_phi, zeros], axis=-1)  # (n_events, 4)

    # Append neutrino
    reco_features = np.concatenate([reco_features, neutrino_reco[:, None, :]], axis=1)  # (n_events, 12, 4)
    truth_pts  = [df[f'truth_pt{p}']   for p in particle_names]
    truth_etas = [df[f'truth_eta{p}']  for p in particle_names]
    truth_phis = [df[f'truth_phi{p}']  for p in particle_names]
    
    truth_masses = [np.zeros_like(df['truth_ptl1'])] + \
                   [df[f'truth_mb{i}'] for i in range(1, 5)] + \
                   [df[f'truth_mj{i}'] for i in range(1, 7)]

    truth_features = np.stack([truth_pts, truth_etas, truth_phis, truth_masses], axis=-1)
    truth_features = truth_features.transpose(1, 0, 2)

    # Neutrino (truth): use 'truth_met' and 'truth_met_phi'
    truth_met_pt = df['truth_met']
    truth_met_phi = df['truth_metphi']
    zeros_truth = np.zeros_like(truth_met_pt)
    neutrino_truth = np.stack([truth_met_pt, zeros_truth, truth_met_phi, zeros_truth], axis=-1)

    truth_features = np.concatenate([truth_features, neutrino_truth[:, None, :]], axis=1)
    
    return reco_features, truth_features

def subset(df, n_evts, train_test=False):
    
    random_subset = df.sample(n_evts, random_state=42)
    subset_fraction = n_evts / len(df)
    random_subset["eventWeight"] = random_subset['eventWeight'] / subset_fraction # Normalize

    if train_test:
        remaining_df = df.drop(random_subset.index)
        second_subset = remaining_df.sample(n_evts, random_state = 99)
        second_subset['eventWeight'] = second_subset['eventWeight'] / subset_fraction
        
        return random_subset, second_subset

    return random_subset

def convert_to4vector(array):
    """
    Convert an array of shape (n_events, n_particles, 4) with (pt, eta, phi, m)
    to an array of shape (n_events, n_particles, 4) with (E, px, py, pz).
    """
    pt = array[:, 0]
    eta = array[:, 1]
    phi = array[:, 2]
    m = array[:, 3]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    return np.stack((E, px, py, pz), axis=-1)

def calculate_mass(array):
    """
    Calculate the invariant mass of the system represented by the input array.
    The input array should have shape (n_events, n_particles, 4) with (E, px, py, pz).
    Returns an array of shape (n_events,) with the invariant mass for each event.
    """
    E = array[:, 0]
    px = array[:, 1]
    py = array[:, 2]
    pz = array[:, 3]

    mass_squared = E**2 - (px**2 + py**2 + pz**2)
    mass_squared = np.maximum(mass_squared, 0)  # Avoid negative values due to numerical issues
    return np.sqrt(mass_squared)

def W_candidate(array):
    # Hard coded W mass
    W_mass = 80.379

    # leading and subleading jets (indices 5 and 6)
    leading_jet = array[:, 5, :]
    subleading_jet = array[:, 6, :]
    leading_jet_4vec = convert_to4vector(leading_jet)
    subleading_jet_4vec = convert_to4vector(subleading_jet)
    combined_4vec = leading_jet_4vec + subleading_jet_4vec

    # Calculate masses
    mass_leading = calculate_mass(leading_jet_4vec)
    mass_subleading = calculate_mass(subleading_jet_4vec)
    mass_combined = calculate_mass(combined_4vec)

    # Stack all masses and 4-vectors: shape (n_events, 3)
    all_masses = np.stack([mass_leading, mass_subleading, mass_combined], axis=1)
    all_4vecs = np.stack([leading_jet_4vec, subleading_jet_4vec, combined_4vec], axis=1)

    # Find index of mass closest to W_mass for each event
    idx_closest = np.argmin(np.abs(all_masses - W_mass), axis=1)

    # Select 4-vector for each event
    selected_4vec = all_4vecs[np.arange(len(array)), idx_closest, :]

    return selected_4vec

def min_mbl(array):
    # b-jet indices: 1, 2
    b_jets = [array[:, i, :] for i in range(1,3)]
    lepton = array[:, 0, :]

    b_jet_4vecs = [convert_to4vector(b_jet) for b_jet in b_jets]
    lepton_4vec = convert_to4vector(lepton)

    mbl_values = []
    for b_jet_4vec in b_jet_4vecs:
        combined_4vec = b_jet_4vec + lepton_4vec
        mbl = calculate_mass(combined_4vec)
        mbl_values.append(mbl)

    mbl_stack = np.stack(mbl_values, axis=1)  # shape (n_events, 2)
    return np.min(mbl_stack, axis=1)  # shape (n_events,)

def max_mbl(array):
    # b-jet indices: 1, 2
    b_jets = [array[:, i, :] for i in range(1,3)]
    lepton = array[:, 0, :]

    b_jet_4vecs = [convert_to4vector(b_jet) for b_jet in b_jets]
    lepton_4vec = convert_to4vector(lepton)

    mbl_values = []
    for b_jet_4vec in b_jet_4vecs:
        combined_4vec = b_jet_4vec + lepton_4vec
        mbl = calculate_mass(combined_4vec)
        mbl_values.append(mbl)

    mbl_stack = np.stack(mbl_values, axis=1)  # shape (n_events, 2)
    return np.max(mbl_stack, axis=1)  # shape (n_events,)

def min_mbW(array):
    # b-jet indices: 1, 2
    b_jet1 = array[:, 1, :]
    b_jet2 = array[:, 2, :]

    b_jet1_4vec = convert_to4vector(b_jet1)
    b_jet2_4vec = convert_to4vector(b_jet2)

    W_4vec = W_candidate(array)

    combined1_4vec = b_jet1_4vec + W_4vec
    combined2_4vec = b_jet2_4vec + W_4vec

    mbW1 = calculate_mass(combined1_4vec)
    mbW2 = calculate_mass(combined2_4vec)

    mbW_stack = np.stack([mbW1, mbW2], axis=1)  # shape (n_events, 2)
    return np.min(mbW_stack, axis=1)  # shape (n_events,)


def mbb(array):
    # b-jet indices: 1, 2
    b_jet1 = array[:, 1, :]
    b_jet2 = array[:, 2, :]

    b_jet1_4vec = convert_to4vector(b_jet1)
    b_jet2_4vec = convert_to4vector(b_jet2)

    combined_4vec = b_jet1_4vec + b_jet2_4vec
    return calculate_mass(combined_4vec)

def mjj(array):
    # jet indices: 5, 6
    jet1 = array[:, 5, :]
    jet2 = array[:, 6, :]

    jet1_4vec = convert_to4vector(jet1)
    jet2_4vec = convert_to4vector(jet2)

    combined_4vec = jet1_4vec + jet2_4vec
    return calculate_mass(combined_4vec)


def dR2b(array):
    # b-jet indices: 1, 2
    b_jet1 = array[:, 1, :]
    b_jet2 = array[:, 2, :]

    eta1 = b_jet1[:, 1]
    phi1 = b_jet1[:, 2]
    eta2 = b_jet2[:, 1]
    phi2 = b_jet2[:, 2]

    delta_eta = eta1 - eta2
    delta_phi = np.abs(phi1 - phi2)
    delta_phi = np.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)

    return np.sqrt(delta_eta**2 + delta_phi**2)

def dR2j(array):
    # b-jet indices: 5, 6
    jet1 = array[:, 5, :]
    jet2 = array[:, 6, :]

    eta1 = jet1[:, 1]
    phi1 = jet1[:, 2]
    eta2 = jet2[:, 1]
    phi2 = jet2[:, 2]

    delta_eta = eta1 - eta2
    delta_phi = np.abs(phi1 - phi2)
    delta_phi = np.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)

    return np.sqrt(delta_eta**2 + delta_phi**2)

def min_dr_bl(array):
    lepton = array[:, 0, :]            # shape (n_events, n_features)
    b_jets = [array[:, i, :] for i in range(1,3)]  # b-jet indices 1 and 2

    dr_values = []

    for b_jet in b_jets:
        delta_eta = lepton[:, 1] - b_jet[:, 1]  # assuming index 1 = eta
        delta_phi = lepton[:, 2] - b_jet[:, 2]  # assuming index 2 = phi
        # wrap-around phi
        delta_phi = np.mod(delta_phi + np.pi, 2*np.pi) - np.pi
        dr = np.sqrt(delta_eta**2 + delta_phi**2)
        dr_values.append(dr)

    dr_stack = np.stack(dr_values, axis=1)  # shape (n_events, 2)
    return np.min(dr_stack, axis=1)         # shape (n_events,)

def max_dr_bl(array):
    lepton = array[:, 0, :]            # shape (n_events, n_features)
    b_jets = [array[:, i, :] for i in range(1,3)]  # b-jet indices 1 and 2

    dr_values = []

    for b_jet in b_jets:
        delta_eta = lepton[:, 1] - b_jet[:, 1]  # assuming index 1 = eta
        delta_phi = lepton[:, 2] - b_jet[:, 2]  # assuming index 2 = phi
        # wrap-around phi
        delta_phi = np.mod(delta_phi + np.pi, 2*np.pi) - np.pi
        dr = np.sqrt(delta_eta**2 + delta_phi**2)
        dr_values.append(dr)

    dr_stack = np.stack(dr_values, axis=1)  # shape (n_events, 2)
    return np.max(dr_stack, axis=1)

def pairwise(array):
    new_entry1 = np.stack([min_mbl(array), min_dr_bl(array), max_dr_bl(array), max_mbl(array)],axis=-1)
    new_entry1_extended = new_entry1[:,np.newaxis,:]
    new_entry2 = np.stack([mbb(array), dR2b(array), dR2j(array), mjj(array)],axis=-1)
    new_entry2_extended = new_entry2[:,np.newaxis,:]
    return np.concatenate([new_entry1_extended, new_entry2_extended], axis=1)


class JetScaler:
    def __init__(self, mask_value=0.0):
        self.mask_value = mask_value
        self.scaler = StandardScaler()
    
    def _valid_mask(self, X):
        """
        Returns a boolean mask of jets that are *not* padding.
        Padding jets are those where all 4 features == mask_value.
        """
        return ~np.all(X == self.mask_value, axis=-1)
    
    def fit(self, X):
        """
        Fit the scaler only on valid (non-padded) jets.
        """
        X = np.array(X, dtype=float)
        valid_mask = self._valid_mask(X)
        valid_jets = X[valid_mask]
        self.scaler.fit(valid_jets)
    
    def transform(self, X):
        """
        Transform valid jets, leave padded jets at mask_value.
        """
        X = np.array(X, dtype=float)
        orig_shape = X.shape
        valid_mask = self._valid_mask(X)
        
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = np.zeros_like(X_flat)
        
        # Only transform valid jets
        X_scaled[valid_mask.reshape(-1)] = self.scaler.transform(X_flat[valid_mask.reshape(-1)])
        
        # Keep masked jets as mask_value (usually 0)
        X_scaled[~valid_mask.reshape(-1)] = self.mask_value
        
        return X_scaled.reshape(orig_shape)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
particles = ['l1', 'b1', 'b2', 'b3', 'b4', 
             'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'met'
             #,'min_mbl', 'mbb', 'max_mbl', 'mjj'
             ]

pt_binning = {
    'l1': np.linspace(0, 700, 16),
    'b1': np.linspace(0, 400, 21),
    'b2': np.linspace(0, 300, 21),
    'b3': np.linspace(0, 200, 16),
    'b4': np.linspace(0, 150, 12),
    'j1': np.linspace(0, 500, 26),
    'j2': np.linspace(0, 400, 21),
    'j3': np.linspace(0, 300, 16),
    'j4': np.linspace(0, 250, 16),
    'j5': np.linspace(0, 200, 16),
    'j6': np.linspace(0, 150, 12),
    'met': np.linspace(0, 300, 31),
    #'min_mbl': np.linspace(0, 600, 21),
    #'mbb': np.linspace(0,900, 21),
    #'max_mbl': np.linspace(0, 600, 21),
    #'mjj': np.linspace(0,900, 21),
}
    
def plot_pt_subplot(ax_main, ax_ratio, data_dict, weights_dict, bins):
    x_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Reference histogram: first entry
    ref_name = list(data_dict.keys())[0]
    ref_counts, _ = np.histogram(data_dict[ref_name], bins=bins, weights=weights_dict[ref_name])
    
    maxy = 0
    colors = ['black', '#3f90da', '#ffa90e', '#bd1f01']  # adjust as needed
    
    for i, (label, values) in enumerate(data_dict.items()):
        counts, _ = np.histogram(values, bins=bins, weights=weights_dict[label])
        ax_main.step(bins[:-1], counts, where='post', color=colors[i], label=label, linewidth=2)
        maxy = max(maxy, counts.max())
        
        if ax_ratio is not None and i > 0:  # skip reference
            ratio = np.divide(counts, ref_counts, out=np.zeros_like(counts), where=ref_counts!=0)
            ax_ratio.step(x_centers, ratio, where='mid', color=colors[i], label=f"{label}/{ref_name}")
    
    ax_main.set_ylim(0, 1.3*maxy)
    ax_main.set_xlim(25,)
    ax_main.grid(True, linestyle='--', alpha=0.5)
    ax_main.legend(fontsize=10)
    if ax_ratio is not None:
        ax_ratio.axhline(1.0, color='r', linestyle='--')
        ax_ratio.set_ylim(0.8, 1.2)
        ax_ratio.set_xlabel(r"$p_T$ [GeV]")
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.grid(True, linestyle='--', alpha=0.5)


import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class MaskedAveragePooling(layers.Layer):
    def call(self, x, mask):
        # x: (batch, seq_len, features)
        # mask: (batch, seq_len), dtype bool
        mask = tf.cast(mask, tf.float32)[..., tf.newaxis]  # (batch, seq_len, 1)
        x = x * mask
        return tf.reduce_sum(x, axis=1) / tf.reduce_sum(mask, axis=1)

class ParticleMaskLayer(layers.Layer):
    def call(self, x):
        return tf.reduce_any(tf.not_equal(x, 0), axis=-1)  # shape: (batch, seq)

def build_transformer_model(input_shape=(12, 4),
                            num_heads=4,
                            ff_dim=64,
                            num_transformer_blocks=2,
                            dropout_rate=0.1):

    inputs = Input(shape=input_shape)

    # Get attention mask: True where not padded
    mask = ParticleMaskLayer()(inputs)  # shape: (batch, 11)

    # Learned positional embedding
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embed = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    x = inputs + pos_embed  # broadcasting position embedding

    for _ in range(num_transformer_blocks):
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_shape[1]
        )(x, x, attention_mask=mask[:, tf.newaxis, :])  # shape: (batch, 1, 11)

        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        ff_output = layers.Dense(ff_dim, activation='relu')(x)
        ff_output = layers.Dense(input_shape[1])(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    # Use masked average pooling instead of GlobalAveragePooling1D
    x = MaskedAveragePooling()(x, mask)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs)

def make_mc_subsample(loader, reco_cuts_mask, N_data):
    """
    Erzeuge ein MC-Subsample mit gleicher statistischer Genauigkeit wie die Daten.

    Parameters
    ----------
    loader : DataLoader
        DataLoader-Objekt mit Attributen `.reco`, `.gen`, `.weight`
    reco_cuts_mask : np.ndarray(bool)
        Maske der Events, die die Reco-Level-Cuts bestehen (analog zu den Daten)
    N_data : int
        Anzahl an Daten-Events, die die Reco-Level-Cuts bestehen
    """

    # Gesamtanzahl an MC-Events
    n_events = len(loader.weight)

    # Init: alle Flags auf False
    flags = np.zeros(n_events, dtype=bool)

    # Summen
    sum_w = 0.0
    sum_w2 = 0.0

    # Ziel-Statistik (Daten haben alle w=1)
    target_rel_unc = 1.0 / np.sqrt(N_data)

    # Loop über Events
    for idx in range(n_events):
        if not reco_cuts_mask[idx]:
            continue

        w = loader.weight[idx]
        sum_w += w
        sum_w2 += w**2

        flags[idx] = True  # Event geht ins Subsample

        # relative Unsicherheit bisher
        rel_unc = np.sqrt(sum_w2) / sum_w

        if rel_unc < target_rel_unc:
            break

    return flags