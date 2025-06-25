import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from speechbrain.inference.classifiers import EncoderClassifier

# --- Ajuste caminho FreeVC ---
sys.path.append("FreeVC")  # Ajuste se necessário
from FreeVC.models import SynthesizerTrn
from FreeVC.utils import HParams, load_checkpoint

# --- Configurações ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CONTENT_ENCODER_MODEL_ID = "lengyue233/content-vec-best"  # ContentVec
CONTENT_ENCODER_PROCESSOR_ID = "facebook/wav2vec2-base-960h"  # feature extractor do ContentVec

FREEVC_CONFIG = "FreeVC/configs/freevc.json"
FREEVC_CHECKPOINT = "FreeVC/checkpoints/freevc.pth"

# --- Dimensões (ajuste conforme seu modelo e config) ---
CONTENT_VEC_DIM = 768   # embedding ContentVec (input)
FREEVC_CONTENT_DIM = 1024  # dimensão interna do FreeVC que queremos replicar (target)

# --- Modelo MLP ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class ContentVecFreeVCDataset(Dataset):
    def __init__(self, file_list, content_encoder, content_processor, device):
        self.file_list = file_list
        self.content_encoder = content_encoder
        self.content_processor = content_processor
        self.device = device

        # This conv1d projects ContentVec to FreeVC expected input channels (1024)
        self.proj = torch.nn.Conv1d(768, 1024, 1).to(self.device)
        self.proj.eval()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]

        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav = resampler(wav)
        wav = wav.to(self.device).squeeze(0)

        # Get ContentVec embedding
        with torch.no_grad():
            inputs = self.content_processor(wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            content_embedding = self.content_encoder(**inputs).last_hidden_state.squeeze(0)  # [T, 768]

        # Get target c_proj = conv1d projection of content_embedding, matching FreeVC input to enc_p
        with torch.no_grad():
            c = content_embedding.transpose(0, 1).unsqueeze(0)  # [1, 768, T]
            c_proj = self.proj(c)  # [1, 1024, T]
            c_proj = c_proj.squeeze(0).transpose(0, 1)  # [T, 1024]

        return content_embedding, c_proj


# --- Função de treino ---
def train_mlp(file_list, content_encoder, content_processor, freevc_decoder, device):
    dataset = ContentVecFreeVCDataset(file_list, content_encoder, content_processor, device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    mlp = MLP(CONTENT_VEC_DIM, 1024, 1024).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    mlp.train()
    for epoch in range(100):
        total_loss = 0
        for content_in, target_out in dataloader:
            content_in = content_in.to(device)      # [T, 768]
            target_out = target_out.to(device)      # [T, 1024]

            optimizer.zero_grad()
            output = mlp(content_in)                 # [T, 1024]
            loss = criterion(output, target_out)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

    torch.save(mlp.state_dict(), "contentvec_to_freevc_mlp.pth")
    print("Treinamento da MLP finalizado e modelo salvo.")

    return mlp

# --- Setup modelos ---
def load_models(device):
    print("Carregando ContentVec...")
    content_processor = Wav2Vec2FeatureExtractor.from_pretrained(CONTENT_ENCODER_PROCESSOR_ID)
    content_encoder = Wav2Vec2Model.from_pretrained(CONTENT_ENCODER_MODEL_ID).to(device)
    content_encoder.eval()

    print("Carregando FreeVC decoder...")
    with open(FREEVC_CONFIG, 'r') as f:
        config_json = json.load(f)
    hps = HParams(**config_json)

    decoder = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)
    load_checkpoint(FREEVC_CHECKPOINT, decoder, None)
    decoder.eval()

    return content_encoder, content_processor, decoder

# --- Uso ---
if __name__ == "__main__":
    # Liste seus arquivos WAV para treinamento
    wav_dir = "/home/lettuce/Downloads/trabalho_final(1)/trabalho_final/mario-voice-dataset/wavs"

    train_files = [os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")]

    content_encoder, content_processor, freevc_decoder = load_models(device)
    mlp_model = train_mlp(train_files, content_encoder, content_processor, freevc_decoder, device)
