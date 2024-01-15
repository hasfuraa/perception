# Overview
Reimplementation of CNN and transformer based 2d object detection. Used purely as a peronsal learning tool. No guarantees on quality or correctness :)

# Quickstart
```bash
# Install requirements
pip install -r requirements.txt

# Execute training from scratch.
# Depending on the configuration could take from ~3hrs to ~10hrs on CPU.
python jobs/train.py

# Evaluate the checkpoints dumped to disk.
python jobs/eval.py --ckpt <PATH_TO_CKPT>

# (Optional) Tensorboard
tensorboard --logdir <PATH_TO_LOGDIR>
```

# Sample Output

#### Eval
```bash
$ time python ./jobs/eval.py 

GPU available: False, used: False
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
Testing DataLoader 0: 100%|███████████████████████████████████████████████████| 625/625 [00:13<00:00, 46.14it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      avg_accuracy           0.89410001039505
      avg_test_loss         0.30652835965156555
────────────────────────────────────────────────────────────────────────────────────────────────────────────────

real    0m23.549s
user    0m50.646s
sys     0m8.310s
```
