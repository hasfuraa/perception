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
