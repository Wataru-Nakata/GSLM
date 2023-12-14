from typing import List
import transformers
from sklearn.cluster import MiniBatchKMeans
import torch
import pickle


class SSLKmeansTokenizer():
    def __init__(self,cfg) -> None:
        self.ssl_model = transformers.AutoModel.from_pretrained(cfg.ssl_model_name,output_hidden_states=True)
        self.ssl_model = self.ssl_model.eval()
        self.quantizer = MiniBatchKMeans(**cfg.kmeans)
        self.cfg = cfg
    @torch.inference_mode()
    def tokenize(self,x:torch.Tensor):
        ssl_feature = self.ssl_model(x,output_hidden_states=True).hidden_states[self.cfg.ssl_layer]
        quantized_feature = self.quantizer.predict(ssl_feature.view(-1, ssl_feature.size(-1)))
        return quantized_feature

    @torch.inference_mode()
    def train_one_iter(self, wavs:List[torch.Tensor]):
        ssl_features = []
        for wav in wavs:
            ssl_feature = self.ssl_model(wav.to(self.ssl_model.device).view(1,-1),output_hidden_states=True).hidden_states[self.cfg.ssl_layer]
            feature_size = ssl_feature.size(-1)
            ssl_features.append(ssl_feature.view(-1,feature_size))
        ssl_feature = torch.concatenate(ssl_features,dim=0)
        self.quantizer.partial_fit(ssl_feature.view(-1, feature_size).cpu())
    def to(self,device:torch.device):
        self.ssl_model.to(device)
    def save(self,path):
        with open(path,'wb') as f:
            pickle.dump(self.quantizer,f)
    def load(self,path):
        with open(path,'rb') as f:
            self.quantizer = pickle.load(f)
class DACTokenizer():
    def __init__(self) -> None:
        pass
