import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
import json
import sys
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# --- SpeechBrain Import for ECAPA-TDNN ---
try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ImportError:
    print("Please install SpeechBrain: pip install speechbrain")
    sys.exit(1)

# --- FreeVC imports ---
try:
    sys.path.append('FreeVC')
    from FreeVC.models import SynthesizerTrn
    from FreeVC.utils import HParams, load_checkpoint
except ImportError:
    print("Make sure FreeVC repo is cloned and in your PYTHONPATH")
    sys.exit(1)

# --- Paths & configs ---
SOURCE_AUDIO_PATH = "/home/lettuce/Downloads/trabalho_final(1)/trabalho_final/yu_narukami.wav"
REFERENCE_AUDIO_PATH = "/home/lettuce/Downloads/trabalho_final(1)/trabalho_final/mario.wav"
OUTPUT_AUDIO_PATH = "converted_audio_mlp_contentvec_ecapa.wav"

CONTENT_ENCODER_MODEL_ID = "lengyue233/content-vec-best"
CONTENT_ENCODER_PROCESSOR_ID = "facebook/wav2vec2-base-960h"
SPEAKER_ENCODER_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

FREEVC_CHECKPOINT = "/home/lettuce/Downloads/trabalho_final(1)/trabalho_final/contentvec+freevc/FreeVC/checkpoints/freevc.pth"
FREEVC_CONFIG = "/home/lettuce/Downloads/trabalho_final(1)/trabalho_final/contentvec+freevc/FreeVC/configs/freevc.json"

MLP_CHECKPOINT_PATH = "contentvec_to_freevc_mlp.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Dimensions (must match training) ---
CONTENT_EMBEDDING_DIM_IN = 768  # ContentVec embedding dim
MLP_HIDDEN_DIM = 1024
MLP_OUTPUT_DIM = 1024  # This is the FreeVC inter_channels for content embedding

SPEAKER_EMBEDDING_DIM_IN = 192  # ECAPA embedding dim (usually 192)
FREEVC_SPEAKER_DIM = 256  # FreeVC speaker embedding dim

# --- Define MLP ---
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --- Load models ---
print("Loading Speaker Encoder...")
speaker_encoder = EncoderClassifier.from_hparams(
    source=SPEAKER_ENCODER_MODEL_ID,
    savedir=f"pretrained_models/{SPEAKER_ENCODER_MODEL_ID.replace('/', '_')}",
    run_opts={"device": device}
)
speaker_encoder.eval()

print("Loading Content Encoder and Processor...")
content_processor = Wav2Vec2FeatureExtractor.from_pretrained(CONTENT_ENCODER_PROCESSOR_ID)
content_encoder = Wav2Vec2Model.from_pretrained(CONTENT_ENCODER_MODEL_ID).to(device)
content_encoder.eval()

print("Loading FreeVC Decoder...")
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

print("Loading trained MLP for Content Projection...")
content_projection = MLP(CONTENT_EMBEDDING_DIM_IN, MLP_HIDDEN_DIM, MLP_OUTPUT_DIM).to(device)
content_projection.load_state_dict(torch.load(MLP_CHECKPOINT_PATH, map_location=device))
content_projection.eval()

# --- Audio processing helper ---
def process_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze(0).to(device)

# --- Inference pipeline ---
print("Processing source audio...")
source_wav = process_audio(SOURCE_AUDIO_PATH)

print("Extracting content embedding from source...")
with torch.no_grad():
    inputs = content_processor(source_wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    raw_content_embedding = content_encoder(**inputs).last_hidden_state.squeeze(0)  # [T, 768]

    projected_content_embedding = content_projection(raw_content_embedding)  # [T, 192]

print("Processing reference audio...")
ref_wav = process_audio(REFERENCE_AUDIO_PATH)

print("Extracting speaker embedding from reference...")
with torch.no_grad():
    raw_speaker_embedding = speaker_encoder.encode_batch(ref_wav.unsqueeze(0), wav_lens=torch.tensor([1.0]).to(device))
    raw_speaker_embedding = raw_speaker_embedding.squeeze(0).squeeze(0)  # [192]

# If FreeVC expects 256D speaker embedding, zero-pad or modify as needed
# Here we do zero-pad to 256D as example:
if raw_speaker_embedding.shape[0] != FREEVC_SPEAKER_DIM:
    padding = FREEVC_SPEAKER_DIM - raw_speaker_embedding.shape[0]
    if padding > 0:
        raw_speaker_embedding = torch.cat([raw_speaker_embedding, torch.zeros(padding, device=device)])

print("Synthesizing audio...")
with torch.no_grad():
    # FreeVC expects content embedding shape: [B, C, T]
    content_for_decoder = projected_content_embedding.transpose(0, 1).unsqueeze(0)  # [1, 192, T]

    # Speaker embedding shape: [B, C]
    speaker_for_decoder = raw_speaker_embedding.unsqueeze(0)  # [1, 256]

    audio_out = decoder.infer(content_for_decoder, speaker_for_decoder)
    audio_out = audio_out.squeeze().cpu().numpy()

print(f"Saving output to {OUTPUT_AUDIO_PATH}...")
sf.write(OUTPUT_AUDIO_PATH, audio_out, hps.data.sampling_rate)

print("Done! Your converted audio is ready.")
