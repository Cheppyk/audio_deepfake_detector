#!/usr/bin/env python
# coding: utf-8
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import scipy.signal as signal
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from pathlib import Path
import pandas
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel
import torch.optim as optim
import torchaudio
import gc
import torchaudio
import soundfile as sf
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
#c:/Users/egorv/Desktop/BProj


class ASVSpoofDataset(Dataset):

    def __init__(self, flac_dir, labels_path):
        """
        Returns all the directory where the flac_files are located, returns the files itself,
        returns the dataset with filenames, targets, speaker ID, and type attack ID.
        Also returns the list of filenames, and target dictionary
        """
        self.flac_dir = flac_dir
        self.files = sorted(Path(flac_dir).glob("*.flac"))
        self.labels_df = pandas.read_csv(labels_path, sep=r"\s+", header=None)
        self.file_names = self.labels_df[1]
        self.target = dict(zip(self.labels_df[1], self.labels_df[4]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        file_path = self.files[idx]

        audio, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
        x = torch.from_numpy(audio.T)
        x = x.mean(dim=0)     

        x = self.normalize_duration(x)
        file_name = file_path.stem
        target_str = self.target.get(file_name)

        y = 1 if target_str == 'bonafide' else 0
        return x, torch.tensor(y).long()

    def normalize_duration(self, x):
        """
        x: torch.Tensor формы (samples,) или (1, samples)
        """
        TARGET_SEC = 4.0
        TARGET_LEN = int(16000 * TARGET_SEC)

        if x.ndim > 1:
            x = x.squeeze()

        cur_len = x.shape[0]

        if cur_len > TARGET_LEN:
            start = torch.randint(0, cur_len - TARGET_LEN + 1, (1,)).item()
            return x[start : start + TARGET_LEN].float()

        elif cur_len < TARGET_LEN:
            pad_len = TARGET_LEN - cur_len
            return torch.nn.functional.pad(x, (0, pad_len), mode='constant', value=0).float()

        return x.float()



import transformers
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        combined_dim = self.wav2vec2.config.hidden_size + self.hubert.config.hidden_size

        self.classifier = nn.Linear(combined_dim, num_classes)

    def forward(self, x):
        x = x.float()
        w2v_out = self.wav2vec2(x).last_hidden_state.mean(dim=1)
        hubert_out = self.hubert(x).last_hidden_state.mean(dim=1)

        combined = torch.cat((w2v_out, hubert_out), dim=1)

        return self.classifier(combined)



def train_one_epoch(epoch_index, tb_writer, train_loader, optimizer, loss_fn, model, device):
    print("Entered train_one_epoch")

    total_loss = 0.0

    for batch_idx, (audios, labels) in enumerate(train_loader):
        batch_loss = 0.0

        if batch_idx == 0:
            print("first batch ok", audios.shape, labels.shape)

        optimizer.zero_grad()
        audios, labels = audios.to(device), labels.to(device)
        outputs = model(audios)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        batch_loss += loss_val
        tb_writer.add_scalar("Loss/Train_batch", batch_loss, batch_idx + 1)
        print(f"Batch {batch_idx + 1} loss: {batch_loss}")

    epoch_avg = total_loss / max(1, len(train_loader))
    tb_writer.add_scalar("Loss/Train_epoch", epoch_avg, epoch_index + 1)
    tb_writer.flush()
    return epoch_avg



def main():
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / "code" / "experiments" / "data" / "ASVSpoof2019"

    train_flac_dir = (DATA_DIR/ "LA" / "ASVSpoof2019_LA_train" / "flac")
    labels_file = DATA_DIR / "LA" / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"

    flac_dataset = ASVSpoofDataset(train_flac_dir, labels_file)


    indices = list(range(len(flac_dataset)))

    train_split, temp_split = train_test_split(indices, train_size=0.8, shuffle=True, random_state=10)
    val_split, test_split = train_test_split(temp_split, train_size=0.5, random_state=10)

    train_subset = Subset(flac_dataset, train_split)
    val_subset = Subset(flac_dataset, val_split)
    test_subset = Subset(flac_dataset, test_split)

    train_loader = DataLoader(train_subset, batch_size=16, num_workers=8, persistent_workers=True, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=16, num_workers=8, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=16, num_workers=8, shuffle=False)

    # persistent_workers=True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    num_epochs = 5
    best_vloss = 1_000_000.

    project_root = Path(r"C:\Users\egorv\Desktop\BProj")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = project_root / "runs" / f"fashion_trainer_{timestamp}"
    writer = SummaryWriter(str(log_dir))

    print("Writing logs to:", log_dir)

    for epoch in range(num_epochs):

        model.train(True)
        avg_loss = train_one_epoch(epoch, writer, train_loader, optimizer, criterion, model, device)
        avg_loss_f = float(avg_loss)

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i+1)
        avg_vloss_f = float(avg_vloss)

        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss_f, 'Validation' : avg_vloss_f },
                        epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = project_root / f"model_{timestamp}_{epoch}.pt"
            torch.save(model.state_dict(), str(model_path))

    writer.close()

if __name__ == "__main__":
    main()

