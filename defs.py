
import cv2
from IPython.display import display, Image, clear_output
import time
import numpy as np
import os 
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_matrix(matrix):
    """Per-node, per-channel z-score over time, ignoring invalid frames.

    matrix: (T, N_NODES, 4) with channels [x, y, green, valid].
    Stats are computed only over rows where valid == 1, separately for each
    (node, channel). Invalid rows stay zero in the output.
    """
    out = matrix.astype(np.float32, copy=True)
    valid = out[..., 3] > 0.5                          # (T, N)
    feat = out[..., :3]                                # (T, N, 3) view

    cnt = valid.sum(axis=0).astype(np.float32)         # (N,)
    cnt_safe = np.maximum(cnt, 1.0)
    m3 = valid[..., None].astype(np.float32)           # (T, N, 1)

    mean = (feat * m3).sum(axis=0) / cnt_safe[:, None] # (N, 3)
    diff = (feat - mean) * m3                          # zeros at invalid rows
    var = (diff * diff).sum(axis=0) / cnt_safe[:, None]
    std = np.sqrt(var)

    normed = (feat - mean) / (std + 1e-8)
    normed *= m3                                       # zero invalid rows
    normed[:, cnt < 2, :] = 0.0                        # match original <2-valid behavior

    out[..., :3] = normed
    return out


class ST_GNN(nn.Module):
    def __init__(self, in_features, hidden_dim, num_nodes):
        super(ST_GNN, self).__init__()

        self.adj = nn.Parameter(torch.ones(num_nodes, num_nodes) / num_nodes)

        self.spatial_linear = nn.Linear(in_features, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, 1)

        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=31,
            padding=15
        )

    def forward(self, x, mask):
        h = F.relu(self.spatial_linear(x))                          # (B, T, N, H)                                                                                                                      
                                                                                                                                                                                                          
        m   = mask.unsqueeze(-1)                                    # (B, T, N, 1)                                                                                                                      
        num = torch.einsum("ij,btjh->btih", self.adj, h * m)                                                                                                                                            
        den = torch.einsum("ij,btj->bti", self.adj.abs(), mask).clamp_min(1e-6).unsqueeze(-1)                                                                                                           
        h   = num / den                                             # (B, T, N, H)                                                                                                                      
                                                                                                                                                                                                        
        h = (h * m).sum(dim=2) / m.sum(dim=2).clamp_min(1e-6)       # (B, T, H)                                                                                                                         
                    
        h = self.temporal_conv(h.permute(0, 2, 1)).permute(0, 2, 1) # (B, T, H)                                                                                                                         
        h = F.relu(h)
                                                                                                                                                                                                          
        return self.output_layer(h).squeeze(-1) 
    
from scipy.signal import butter, filtfilt                                                                                                                                                               
                                                                                                                                                                                                          
def bandpass(sig, fs, low=0.7, high=4.0, order=4):                                                                                                                                                      
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")                                                                                                                                       
    return filtfilt(b, a, sig)  


def neg_pearson_loss(pred, y, mask=None):                                                                                                                                                               
    # pred, y: (B, T)                                                                                                                                                                                   
    if mask is not None:
        w = mask                                # (B, T), values in {0,1}                                                                                                                               
        pred = pred - (pred * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)                                                                                                             
        y    = y    - (y    * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)                                                                                                             
        pred = pred * w                                                                                                                                                                                 
        y    = y    * w                                                                                                                                                                                 
    else:                                                                                                                                                                                               
        pred = pred - pred.mean(1, keepdim=True)
        y    = y    - y.mean(1, keepdim=True)                                                                                                                                                           
                                                                                                                                                                                                        
    num = (pred * y).sum(1)
    den = torch.sqrt((pred ** 2).sum(1) * (y ** 2).sum(1) + 1e-8)                                                                                                                                       
    r   = num / den                              # (B,)                                                                                                                                                 
    return 1.0 - r.mean()

class rPPGWindow(Dataset):
    def __init__(self, file_list, v_dir, p_dir, window_size=256, stride= 128):
        self.v_dir = v_dir
        self.p_dir = p_dir
        self.window_size = window_size
        self.samples = []

        for fname in file_list:
            v_path = os.path.join(v_dir, fname)

            v_shape = np.load(v_path, mmap_mode="r").shape
            total_frames = v_shape[0]

            for start in range(0, total_frames - self.window_size +1, stride):
                end = start + self.window_size 
                self.samples.append((fname, start, end))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        fname, start, end = self.samples[index]
        v = np.load(os.path.join(self.v_dir, fname), mmap_mode="r")[start:end]   # (T, N, 4)                                                                                                                
        y = np.load(os.path.join(self.p_dir, fname), mmap_mode="r")[start:end]                                                                                                                              
        x_tensor    = torch.tensor(v[..., :3].copy(), dtype=torch.float32)       # (T, N, 3)                                                                                                                
        mask_tensor = torch.tensor(v[..., 3].copy(),  dtype=torch.float32)       # (T, N)                                                                                                                   
        y_tensor    = torch.tensor(y.copy(),          dtype=torch.float32)                                                                                                                                  
        return x_tensor, mask_tensor, y_tensor      