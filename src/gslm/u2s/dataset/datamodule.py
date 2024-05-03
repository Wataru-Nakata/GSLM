from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import torch
import hydra
from torchaudio._extension import torchaudio


class u2sDataModule(LightningDataModule):
    def __init__(self,cfg:DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset = hydra.utils.instantiate(
            self.cfg.train_dataset
        )
        self.val_dataset = hydra.utils.instantiate(
            self.cfg.val_dataset
        )
        if 'xvector' in cfg.keys():
            self.xvector_model = hydra.utils.instantiate(cfg.xvector.model)
            self.xvector_sr = cfg.xvector.sr
        else:
            self.xvector_model = None
            self.xvector_sr = None
        if 'use_pitch' in cfg.keys():
            self.use_pitch = cfg.use_pitch
        else:
            self.use_pitch = False

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=lambda batch: self.collate_fn(
                batch, self.cfg.data.segment_size.train
            ),
            shuffle=True,
            num_workers=9,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=lambda batch: self.collate_fn(
                batch, self.cfg.data.segment_size.val
            ),
            num_workers=9,
        )
    @torch.no_grad()
    def collate_fn(self, batch, segment_size: int = -1):
        batch = [
            {
                'resampled_speech.pth': b[0],
                'input_feature': b[1],
                '__key__': b[2],
            } for b in batch
        ]

        outputs = dict()

        if self.xvector_model is not None:
            embeddings = []
            for sample in batch:
                wav = sample['resampled_speech.pth']
                wav_xvector = torchaudio.functional.resample(wav,self.cfg.sample_rate,self.xvector_sr)
                xvector = self.xvector_model.encode_batch(wav_xvector.unsqueeze(0))
                embedding_size = xvector.size(-1)
                embeddings.append(xvector.squeeze(0))
            outputs['xvector'] = torch.stack(embeddings).view(-1,embedding_size)
        if self.use_pitch:
            import pyworld 
            import numpy as np
            logf0s = []
            for sample in batch:
                wav = sample['resampled_speech.pth']
                cut_samples = int(self.cfg.data.target_feature.bias*self.cfg.sample_rate)
                wav = wav[cut_samples:-cut_samples]
                # wav = torch.cat((padding_tensor,wav),dim=0)
                # wav = torch.cat((wav,padding_tensor),dim=0)
                sr = self.cfg.sample_rate
                _f0, time = pyworld.dio(wav.view(-1).numpy().astype(np.double),sr,frame_period=20)
                f0 = pyworld.stonemask(wav.view(-1).numpy().astype(np.double), _f0, time, sr)
                f0 = torch.log10(torch.from_numpy(f0).float())
                logf0s.append(f0[1:].view(-1))

        if segment_size != -1:
            cropped_speeches = []
            input_features = []
            for idx,sample in enumerate(batch):
                wav = sample["resampled_speech.pth"]
                input_feature= sample[self.cfg.data.target_feature.key]
                feature_len = input_feature.size(0)
                if self.use_pitch:
                    assert feature_len == logf0s[idx].size(0)
                if feature_len > (segment_size+1):
                    feature_start = random.randint(
                        0, feature_len - segment_size - 1
                    )
                    feature_end = segment_size + feature_start
                    speech_start_sec = feature_start / self.cfg.data.target_feature.samples_per_sec + self.cfg.data.target_feature.bias
                    speech_end_sec = (feature_start + segment_size) / self.cfg.data.target_feature.samples_per_sec + self.cfg.data.target_feature.bias
                    cropped_speeches.append(
                        wav.squeeze()[
                            int(speech_start_sec * self.cfg.sample_rate) : int(speech_end_sec * self.cfg.sample_rate)
                        ]
                    )
                    input_features.append(
                        input_feature[
                            feature_start:feature_end
                        ]
                    )
                    if self.use_pitch:
                        logf0s[idx] = logf0s[idx][feature_start:feature_end]
                else:
                    cropped_speeches.append(wav.squeeze())
                    input_features.append(
                        input_feature
                    )
            outputs["resampled_speech.pth"] = pad_sequence(
                cropped_speeches, batch_first=True
            )
            outputs["input_feature"] = pad_sequence(
                input_features, batch_first=True
            )
        else:
            outputs["resampled_speech.pth"] = pad_sequence(
                [b["resampled_speech.pth"].squeeze() for b in batch], batch_first=True
            )
            outputs["input_feature"] = pad_sequence(
                [b["input_feature"] for b in batch], batch_first=True
            )
        
        if self.use_pitch:
            outputs['lf0'] = pad_sequence(logf0s,batch_first=True)
        outputs["wav_lens"] = torch.tensor(
            [b["resampled_speech.pth"].size(0) for b in batch]
        )

        outputs["filenames"] = [b["__key__"] for b in batch]
        return outputs

class dgslmDataModule(u2sDataModule):
    @torch.no_grad()
    def collate_fn(self, batch, segment_size: int = -1):
        batch = [
            {
                'resampled_speech.pth': b[0],
                'input_feature': b[1],
                '__key__': b[2],
            } for b in batch
        ]

        outputs = dict()

        if self.use_pitch:
            import pyworld 
            import numpy as np
            logf0s = []
            for sample in batch:
                wav = sample['resampled_speech.pth']
                cut_samples = int(self.cfg.data.target_feature.bias*self.cfg.sample_rate)
                wav = wav[cut_samples:-cut_samples]
                # wav = torch.cat((padding_tensor,wav),dim=0)
                # wav = torch.cat((wav,padding_tensor),dim=0)
                sr = self.cfg.sample_rate
                _f0, time = pyworld.dio(wav.view(-1).numpy().astype(np.double),sr,frame_period=20)
                f0 = pyworld.stonemask(wav.view(-1).numpy().astype(np.double), _f0, time, sr)
                f0 = torch.log10(torch.from_numpy(f0).float())
                logf0s.append(f0[1:].view(-1))

        if segment_size != -1:
            cropped_speeches = []
            input_features = []
            embeddings = []
            for idx,sample in enumerate(batch):
                wav = sample["resampled_speech.pth"]
                input_feature= sample[self.cfg.data.target_feature.key]
                feature_len = input_feature.size(0)
                if self.use_pitch:
                    assert feature_len == logf0s[idx].size(0)
                if feature_len > (segment_size+1):
                    feature_start = random.randint(
                        0, feature_len - segment_size - 1
                    )
                    feature_end = segment_size + feature_start
                    speech_start_sec = feature_start / self.cfg.data.target_feature.samples_per_sec + self.cfg.data.target_feature.bias
                    speech_end_sec = (feature_start + segment_size) / self.cfg.data.target_feature.samples_per_sec + self.cfg.data.target_feature.bias
                    cropped_speeches.append(
                        wav.squeeze()[
                            int(speech_start_sec * self.cfg.sample_rate) : int(speech_end_sec * self.cfg.sample_rate)
                        ]
                    )
                    input_features.append(
                        input_feature[
                            feature_start:feature_end
                        ]
                    )
                    if self.use_pitch:
                        logf0s[idx] = logf0s[idx][feature_start:feature_end]
                else:
                    cropped_speeches.append(wav.squeeze())
                    input_features.append(
                        input_feature
                    )
                if self.xvector_model is not None:
                    wav = cropped_speeches[-1]
                    wav_xvector = torchaudio.functional.resample(wav,self.cfg.sample_rate,self.xvector_sr)
                    xvector = self.xvector_model.encode_batch(wav_xvector.unsqueeze(0))
                    embedding_size = xvector.size(-1)
                    embeddings.append(xvector.squeeze(0))
            outputs["resampled_speech.pth"] = pad_sequence(
                cropped_speeches, batch_first=True
            )
            outputs["input_feature"] = pad_sequence(
                input_features, batch_first=True
            )
            outputs['xvector'] = torch.stack(embeddings).view(-1,embedding_size)
        else:
            outputs["resampled_speech.pth"] = pad_sequence(
                [b["resampled_speech.pth"].squeeze() for b in batch], batch_first=True
            )
            outputs["input_feature"] = pad_sequence(
                [b["input_feature"] for b in batch], batch_first=True
            )
        
        if self.use_pitch:
            outputs['lf0'] = pad_sequence(logf0s,batch_first=True)
        outputs["wav_lens"] = torch.tensor(
            [b["resampled_speech.pth"].size(0) for b in batch]
        )

        outputs["filenames"] = [b["__key__"] for b in batch]
        return outputs