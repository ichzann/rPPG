 
 This also explains the 2000 number exactly: var(y) ≈ 49.7² ≈ 2470. A model emitting ~0 produces MSE ≈ 2470, and you got 2451 in epoch 1. Almost no learning happened — the loss is just var(y).         
                                                                                                                                                                                                          
  Fix                                                                                                                                                                                                     
                  
  In sync_matrix_ppg, after interpolation and before return:                                                                                                                                              
   
  ppg_resampled = (ppg_resampled - ppg_resampled.mean()) / (ppg_resampled.std() + 1e-8)                                                                                                                   
  return ppg_resampled                                                                                                                                                                                    
   
  Then nuke the cached folder so the cell actually re‑runs (it currently skips existing files):                                                                                                           
                  
  import shutil                                                                                                                                                                                           
  shutil.rmtree("ppg_sync_final", ignore_errors=True)

  Re‑run the sync cell. Re‑check:                                                                                                                                                                         
   
  y = np.load("ppg_sync_final/" + os.listdir("ppg_sync_final")[0])                                                                                                                                        
  print(y.mean(), y.std(), y.min(), y.max())                                                                                                                                                              
                                                                                                                                                                                                          
  Expected: mean ≈ 0, std ≈ 1, min/max in roughly [-4, 4].                                                                                                                                                
                                                                                                                                                                                                          
  One thing to think about before you regenerate                                                                                                                                                          
                  
  Per‑file z‑scoring is fine, but if you want the model to learn heart rate and not residual baseline drift, bandpass first:                                                                              
                  
  from scipy.signal import butter, filtfilt                                                                                                                                                               
                                                                                                                                                                                                          
  def bandpass(sig, fs, low=0.7, high=4.0, order=4):                                                                                                                                                      
      b, a = butter(order, [low/(fs/2), high/(fs/2)], btype="band")                                                                                                                                       
      return filtfilt(b, a, sig)                                                                                                                                                                          
                                                                                                                                                                                                          
  # Use the video FPS as fs (you'll need to save it during extraction —                                                                                                                                   
  # for now, ~30 Hz is a safe default for these webcam recordings).                                                                                                                                       
  ppg_resampled = bandpass(ppg_resampled, fs=30.0)                                                                                                                                                        
  ppg_resampled = (ppg_resampled - ppg_resampled.mean()) / (ppg_resampled.std() + 1e-8)
                                                                                                                                                                                                          
  Without the bandpass, y.std() after z‑scoring is dominated by motion and slow drift — you're then asking the model to fit a baseline it has no input for, and MSE plateaus. With the bandpass, std is   
  dominated by the cardiac waveform itself, which is what the green channel actually carries information about.                                                                                           
                                                                                                                                                                                                          
  If you want to ship this in two steps: regenerate without the bandpass first to confirm the loop converges (epoch‑1 loss should land around 0.7–0.9, dropping over epochs). Then add the bandpass and   
  retrain.
                                                                                                                                                                                                          
  What to expect after the fix                                                                                                                                                                            
   
  - Epoch 1: train ≈ val ≈ 0.7–0.9                                                                                                                                                                        
  - By epoch 5–10 they diverge a bit (val flattens, train keeps dropping) — that's normal generalization gap.
  - If train drops below ~0.1 while val stays at 0.8, you're overfitting and we'll talk regularization.                                                                                                   
                                                                                                                                                                                                          
  Don't restart training until that 5‑line check shows y mean≈0, std≈1.  
 
 
 
 
 Bugs / issues to fix                                                                                                                                                      
                                                                                                                                                                            
  3. PPG sync via linspace(0,1,...) assumes video and PPG have identical start/end times. If they don't, you're stretching the cardiac signal — frequency content shifts and
   HR labels become wrong. You need actual timestamps from the dataset (or at least matched fps + offset), not just length-based interpolation.                             
  4. Normalization of x,y: z-scoring near-constant landmark coords amplifies sub-pixel jitter into pure noise. Either drop x,y, keep them un-normalized as positional       
  features, or use relative geometry (Δ from face center).                                                                                                                  
  5. 5×5 green patch is small and noisy. Standard rPPG uses 10–20 px patches or full ROI mean. Also: keeping only green throws away the info POS/CHROM rely on — saving R   
  and B too costs almost nothing and unlocks better baselines.                                                                                                              
  6. No bandpass filter on PPG/rPPG (0.7–4 Hz). Train without it and the model wastes capacity learning to ignore baseline drift and respiration.                           
  7. Different cameras = different fps. Confirm fps is stored per video; if fps varies, the same node-time series means different things across samples. Resample to a      
  common fps (e.g., 30 Hz) before training.                                                                                                                                 
  8. Stray final_matrix_video.npy at repo root collides with the folder name — looks like an accidental save.   



  Roadmap (short)                                                                                                                                                           
                                                                                                                                                                            
  1. Fix the 3 bugs above (token, sync, normalization scope).                                                                                                               
  2. Add bandpass filter + common fps resample in a preprocessing step.
  3. Save R, G, B (not just G) per patch — cheap, future-proof.                                                                                                             
  4. Build a torch.utils.data.Dataset that yields windows (e.g., 10 s = 300 frames) of (T, N, C) + PPG target.                                                              
  5. Define the graph: start with a fixed anatomical adjacency (forehead nodes connected, cheek nodes connected, mirror symmetry). Add learned edge weights later.          
  6. Baseline first: a 1D-CNN or simple GRU on mean(green) across nodes — gives you a number to beat. Then ST-GCN / GAT-temporal.                                           
  7. Loss: negative Pearson correlation on bandpassed waveform (standard in rPPG), plus optional HR-MAE metric.                                                             
  8. Sanity check pipeline on 1 subject end-to-end before scaling — make sure HR from your extracted signal matches PPG-derived HR with classical POS/CHROM. If classical   
  methods fail on your features, a GNN won't save it.       









