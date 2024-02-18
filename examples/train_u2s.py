import hydra


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    datamodule = hydra.utils.instantiate(cfg.u2s.datamodule)
    lightning_module = hydra.utils.instantiate(cfg.u2s.lightning_module)

    loggers = hydra.utils.instantiate(cfg.u2s.logger)
    trainer = hydra.utils.instantiate(cfg.u2s.trainer,logger=loggers)
    trainer.fit(lightning_module,datamodule,ckpt_path=cfg.u2s.ckpt_path)

if __name__ == '__main__':
    main()
