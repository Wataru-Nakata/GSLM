import hydra
from omegaconf import DictConfig, OmegaConf
from torchaudio._backend import torchaudio
from gslm.preprocess.kmeans_model import ExtractSSLFeatures, KmeansModel
from torch.utils.data import DataLoader, DistributedSampler
import torch
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
import os

from gslm.preprocess.preprocess_dataset import GlobWavDataset, HQYoutubeDataset


@torch.inference_mode()
def main(dataset):
    dl = DataLoader(dataset,batch_size=1,num_workers=30,shuffle=False)
    logging.info(f"dataset length {len(dataset)}")
    manifest_path = Path(f'/mnt/hdd/datasets/hq-youtube/manifest.txt')
    with manifest_path.open('w') as f:
        f.write("/mnt/hdd/hq-youtube/\n")
        for idx,(utt_id,(wav,sr)) in enumerate(tqdm(dl)):
            if utt_id[0] != 'error':
                torchaudio.save(f"/mnt/hdd/datasets/hq-youtube/{utt_id[0]}.flac", src=wav.view(1,-1),sample_rate=sr)
                f.write(f"{utt_id[0]}.flac\t{wav.view(1,-1).size(1)}\n")

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dataset = HQYoutubeDataset(
        "/mnt/nas/disk1/takamichi/share/corpus/hq-youtube/2023-08-01/",
    )
    raise ValueError
    main(dataset)
