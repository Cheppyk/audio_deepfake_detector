from pathlib import Path

# ASVSpoof2019 LA Audio Spoofing Detection (Wav2Vec2 + HuBERT Ensemble)

## Overview
This project trains a binary classifier to detect **spoofed** vs **bonafide** (real) speech using the **ASVspoof 2019 Logical Access (LA)** dataset.

The model is an ensemble of two pretrained Transformer encoders:
- **Wav2Vec2** (`facebook/wav2vec2-base`)
- **HuBERT** (`facebook/hubert-base-ls960`)

Their embeddings are concatenated and passed into a linear classifier.

---

## Files
### `preprocessing_train_pipeline.py`
Main training script. It:
- loads `.flac` audio files
- reads protocol labels (`bonafide` / `spoof`)
- crops or pads each sample to a fixed duration (4 seconds at 16 kHz)
- trains the model and logs loss values to TensorBoard
- saves the best checkpoint by validation loss

---

## Dataset Structure
The script expects the ASVspoof2019 dataset in the following structure:

    code/experiments/data/ASVSpoof2019/
    └── LA/
        ├── ASVSpoof2019_LA_train/
        │   └── flac/
        ├── ASVSpoof2019_LA_eval/
        │   └── flac/
        └── ASVspoof2019_LA_cm_protocols/
            ├── ASVSpoof2019.LA.cm.train.trn.txt
            └── ASVSpoof2019.LA.cm.eval.trl.txt

If your dataset is located elsewhere, update this line in the script:

```python
DATA_DIR = PROJECT_ROOT / "code" / "experiments" / "data" / "ASVSpoof2019" 
```

## Model Architecture

The model consists of:
1. Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
2. HubertModel.from_pretrained("facebook/hubert-base-ls960")
3. Mean pooling over time for both outputs
4. Concatenation of both embeddings
5. Final linear classifier (nn.Linear)
Output: 2 logits (CrossEntropyLoss)

## Training

Run the script:

python preprocessing_train_pipeline.py

The script uses:

- Train set: ASVSpoof2019_LA_train/flac
- Validation set: ASVSpoof2019_LA_eval/flac (used as validation in this script)

Training parameters (current defaults):

- batch size: 16
- epochs: 5
- optimizer: Adam
- learning rate: 1e-5
- loss: CrossEntropyLoss