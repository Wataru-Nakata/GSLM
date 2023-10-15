from omegaconf import DictConfig
from lightning_vocoders.models.hifigan.lightning_module import HiFiGANLightningModule
from .generator_with_embedding import GeneratorWithEmbedding, GeneratorWithEmbeddingXVector

class HiFiGANEmbeddingLightningModule(HiFiGANLightningModule,object):
    def __init__(self, cfg: DictConfig) -> None:
        HiFiGANLightningModule.__init__(self,cfg)
        self.generator = GeneratorWithEmbedding(cfg.model.generator)



class HiFiGANEmbeddingXvectorLightningModule(HiFiGANLightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        HiFiGANLightningModule.__init__(self,cfg)
        self.generator = GeneratorWithEmbeddingXVector(cfg.model.generator)
    def generator_forward(self, batch):
        wav_generator_out = self.generator(batch["input_feature"],batch['xvector'])
        return wav_generator_out


