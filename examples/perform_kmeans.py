from torch.utils.data import DataLoader
from gslm.preprocess.kmeans_model import ExtractSSLFeatures, KmeansModel
from gslm.preprocess.preprocess_dataset import GlobWavDataset
from omegaconf import DictConfig
import webdataset
import torchaudio
import hydra
import numpy as np
from tqdm import tqdm
import torch

@hydra.main(config_path='config', config_name='config',version_base=None)
@torch.inference_mode()
def main(cfg:DictConfig):
    device = 'cuda'
    dataset = GlobWavDataset(
        ['/mnt/hdd/datasets/hq-youtube/'],
        ['**/*.flac'],
        False,False
    )

    dl = DataLoader(dataset,1,num_workers=8)

    ssl_model = ExtractSSLFeatures(cfg.preprocess).to(device)
    sink = webdataset.ShardWriter(cfg.preprocess.feature_output_path)
    features = []
    for idx,batch in enumerate(tqdm(dl)):
        wav_id, (wav,sr), wav_path = batch
        wav = wav.to(device)
        assert wav.size(1) == 1
        wav_16k = torchaudio.functional.resample(wav,sr,16_000)
        wav_16k = wav_16k.view(1,-1)
        outputs = ssl_model(wav_16k)
        feature_size = outputs.size(2)
        feature = outputs.view(-1,feature_size)
        features.extend(feature.cpu().numpy())
        sink.write(
            {
                "__key__": str(wav_id),
                "wav_path.txt": str(wav_path),
                "feature.pth": webdataset.torch_dumps(feature.cpu())
            }
        )
    sink.close()
    kmeans_model = KmeansModel(cfg.preprocess,feature_size)
    feature = np.stack(features)
    print(feature.shape)
    kmeans_model.train(feature)
    kmeans_model.save()
if __name__ == "__main__":
    main()
