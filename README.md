# rPPG — Remote Photoplethysmography

Estimating blood-volume-pulse (BVP) signals from facial video using a
Spatio-Temporal Graph Neural Network (ST-GNN).

## Idea

Each frame of a webcam recording is reduced to a small graph of facial
landmark "nodes" carrying `(x, y, green-channel, valid)` features. A sliding
window over time is fed to an ST-GNN that mixes information across nodes
(spatial) and across time (temporal 1D conv) to predict the rPPG waveform.
Training uses a negative-Pearson loss against ground-truth contact PPG.

## Pipeline

1. **Capture / sync** — raw video and reference PPG aligned per recording.
2. **Feature extraction** — per-frame node matrices `(T, N_NODES, 4)`.
3. **Normalization** — per-node, per-channel z-score over valid frames
   (`normalize_matrix` in `defs.py`).
4. **Windowing** — `rPPGWindow` cuts each clip into overlapping windows
   (default 256 frames, stride 128).
5. **Model** — `ST_GNN` (`defs.py`): learnable adjacency, spatial linear,
   temporal Conv1d (kernel 31), output head.
6. **Loss** — `neg_pearson_loss` with a validity mask.
7. **Eval** — bandpass filter (0.7–4 Hz) on the predicted signal before
   comparing to ground truth.

## Layout

```
defs.py           # model, dataset, losses, helpers
notebook.ipynb    # training / evaluation notebook
pre.ipynb         # data preparation
model.pt          # trained checkpoint
```

Raw data directories (`final_matrix_video/`, `ppg_sync/`, etc.) are **not**
tracked in this repo — they are generated locally from the source recordings.

## Reproducing

You need the source recordings to regenerate the `.npy` matrices. Once they
exist locally, open `notebook.ipynb` and run end-to-end.

## Note on AI use

Throughout this project, **Claude (Anthropic)** was used as a *mentor* —
to discuss design choices, clarify concepts, and review code.
