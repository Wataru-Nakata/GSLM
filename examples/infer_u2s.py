import argparse
from gslm.u2s.hifigan.lightning_module import HiFiGANEmbeddingXvectorLightningModule
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import hydra
@torch.inference_mode()
def get_xvector():
    import torchaudio
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
    signal, fs = torchaudio.load('/home/acf16285io/nakata/GSLM/examples/jvs001_VOICEACTRESS100_001.wav')
    if fs != 16_000:
        signal = torchaudio.functional.resample(signal,fs,16_000)
    embeddings = classifier.encode_batch(signal)
    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path")
    parser.add_argument("--input_path")
    parser.add_argument("--xvector_path")
    parser.add_argument("--output_path")
    args = parser.parse_args()
    with open(args.input_path) as f:
        lines = f.readlines()
        lines = [l.strip().split(" ") for l in lines]
        lines = [[int(x) for x in line] for line in lines]
    cfg = torch.load(args.ckpt_path,map_location='cpu')['hyper_parameters']['cfg']
    lightning_module = HiFiGANEmbeddingXvectorLightningModule.load_from_checkpoint(args.ckpt_path,map_location='cpu')
    lightning_module.eval()
    with torch.inference_mode():
        xvector = get_xvector()
        outputs = []
        for line in lines:
            input_feature = torch.tensor(line).long()
            batch = {}
            print(input_feature.size(),xvector.size())
            batch['input_feature'] = input_feature.unsqueeze(0)
            batch['xvector'] = xvector.squeeze(0)
            output = lightning_module.generator_forward(batch)
            outputs.append(output.cpu().view(-1))
        torchaudio.save("sampled.wav",torch.stack(outputs),sample_rate=cfg.sample_rate)