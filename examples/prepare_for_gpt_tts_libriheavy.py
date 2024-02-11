from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers import Tokenizer
import hydra
from pathlib import Path
import json
import lhotse
from phonemizer.backend import EspeakBackend
from tqdm import tqdm
import random

unique_phones = set()

@hydra.main(config_path='config', config_name='config',version_base=None)
def prepare_dataset_as_json(cfg):
    output_stream = Path(cfg.ulm.librilight.output_jsonl_path).open('w')
    root = Path("/mnt/hdd/datasets/libritts")
    backend = EspeakBackend('en-us')
    speeches = dict()
    features = dict()
    with open(cfg.ulm.librilight.feature_path) as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            features[line.split('|')[0].strip()] = line.split('|')[1].strip().split()
    manifest: lhotse.CutSet= lhotse.load_manifest(cfg.ulm.librilight.manifest)
    for x in manifest:
        wav_path = f"{x.id}.flac"
        normalized_text = x.supervisions[0].custom['texts'][0]
        data = features[wav_path]
        speaker = x.id.split('/')[1]
        speeches[wav_path] = {
            'wav_path':wav_path,
            'text':normalized_text,
            'phones':phones,
            'data': data,
            'speaker': speaker
        }
    speech_by_speaker = dict()
    for speech in speeches.values():
        speaker = speech['speaker']
        if speaker not in speech_by_speaker.keys():
            speech_by_speaker[speaker] = []
        speech_by_speaker[speaker].append(speech)

    for speaker in tqdm(speech_by_speaker.keys()):
        for speech in speech_by_speaker[speaker]:
            for i in range(10):
                same_speaker_speeches = speech_by_speaker[speaker]
                if len(same_speaker_speeches) == 1:
                    continue
                random_speech = random.choice(same_speaker_speeches)
                while random_speech['wav_path'] == speech['wav_path']:
                    random_speech = random.choice(same_speaker_speeches)
                prompt_data = random_speech['data']
                prompt_phones = random_speech['phones']
                data = speech['data']
                phones = speech['phones']
                output_stream.write(
                    json.dumps(
                        {'text':f"<s> {' '.join(list(prompt_phones))} {' '.join(list(phones))} [SEP] {prompt_data} {data} </s>"}
                    ,ensure_ascii=False)
                )
                output_stream.write('\n')
    output_stream.close()
    return unique_phones

if __name__ == "__main__":
    phones = prepare_dataset_as_json()
