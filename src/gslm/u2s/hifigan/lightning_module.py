from omegaconf import DictConfig
from lightning_vocoders.models.hifigan.lightning_module import HiFiGANLightningModule
from .generator_with_embedding import GeneratorWithEmbedding

class HiFiGANEmbeddingLightningModule(HiFiGANLightningModule,object):
    def __init__(self, cfg: DictConfig) -> None:
        HiFiGANLightningModule.__init__(self,cfg)
        self.generator = GeneratorWithEmbedding(cfg.model.generator)