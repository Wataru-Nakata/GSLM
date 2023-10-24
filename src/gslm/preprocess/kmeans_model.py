import numpy as np
import torch
import transformers
from transformers.modeling_outputs import BaseModelOutput


class KmeansModel():
    def __init__(self,cfg,data_dim:int) -> None:
        self.model = faiss.Kmeans(
            d=data_dim,
            **cfg.kmeans.cfg,
        )
        self.cfg = cfg
    def train(self,data):
        self.model.train(data)
    def save(self):
        np.save(self.cfg.kmeans.model_path,self.model.centroids)

class ExtractSSLFeatures(torch.nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.ssl_model = transformers.AutoModel.from_pretrained(
            cfg.speech_ssl.model_path,
            output_hidden_states=True
        ).eval()
        self.cfg = cfg
    
    def forward(self,wav_16k:torch.Tensor) -> BaseModelOutput:
        assert (wav_16k.size(0) == 1) and (len(wav_16k.size()) == 2)
        outputs:BaseModelOutput = self.ssl_model(wav_16k,output_hidden_states=True).hidden_states[self.cfg.layer]
        return outputs

