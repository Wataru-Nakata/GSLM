from torch.utils.data import DataLoader
from gslm.preprocess.preprocess_dataset import GlobWavDataset
from gslm.s2u.tokenizer import SSLKmeansTokenizer
from omegaconf import DictConfig
import torchaudio
import hydra
import numpy as np
from tqdm import tqdm
import torch

def collate_fn(batch):
    wav_id = [b[0] for b in batch]
    wav = [b[1][0].view(-1) for b in batch]
    sr = [b[1][1] for b in batch]
    wav_path = [b[2] for b in batch]
    wav = torch.nn.utils.rnn.pad_sequence(wav,batch_first=True)
    return wav_id, (wav,sr), wav_path

@hydra.main(config_path='config', config_name='config',version_base=None)
@torch.inference_mode()
def main(cfg:DictConfig):
    device = 'cuda'
    dataset = GlobWavDataset(
        ['/mnt/hdd/datasets/hq-youtube/'],
        ['**/*.flac'],
        False,False
    )

    dl = DataLoader(dataset,batch_size=cfg.s2u.tokenizer.batch_size,num_workers=8,collate_fn=collate_fn)
    kmeans_model = SSLKmeansTokenizer(cfg.s2u.tokenizer)
    kmeans_model.to(device)
    features = []
    for idx,batch in enumerate(tqdm(dl)):
        wav_id, (wav,sr), wav_path = batch
        wav = wav.to(device)
        wav_16k = torchaudio.functional.resample(wav,sr[0],16_000)
        kmeans_model.train_one_iter(wav_16k)
        
    kmeans_model.save(cfg.s2u.tokenizer.save_path)
if __name__ == "__main__":
    main()
