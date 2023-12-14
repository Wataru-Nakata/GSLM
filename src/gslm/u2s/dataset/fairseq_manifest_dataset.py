import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
class FairseqManifestDataset(Dataset):
    def __init__(self,manifest_path,resampled_sr,extracted_feature_path) -> None:
        super().__init__()
        manifest_path = Path(manifest_path)
        feature_path = Path(extracted_feature_path)
        assert manifest_path.exists()
        assert feature_path.exists()
        with manifest_path.open() as f:
            lines = f.readlines()
        self.path_root = Path(lines[0].strip())
        assert self.path_root.exists()
        self.wav_files = [line.strip().split('\t')[0] for line in lines[1:]]
        self.resampled_sr = resampled_sr

        with feature_path.open() as f:
            lines = f.readlines()
        self.feature = dict()
        for line in lines:
            wav_id, tokens = line.strip().split('|')
            self.feature[wav_id] = [int(token) for token in tokens.split(' ')]
    def __len__(self):
        return len(self.wav_files)

    @torch.inference_mode()
    def __getitem__(self, index):
        wav_file = self.wav_files[index]
        wav_path = self.path_root / wav_file
        wav,sr = torchaudio.load(wav_path)
        assert wav.size(0) == 1
        resampled_wav = torchaudio.functional.resample(wav,sr,self.resampled_sr)
        feature = torch.tensor(self.feature[wav_file]).view(-1)
        return resampled_wav.view(-1),feature,wav_path.stem

