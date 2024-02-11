from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers import Tokenizer
import hydra
from pathlib import Path
import json
import lhotse
from phonemizer.backend import EspeakBackend
from phonemizer.logger import get_logger
from tqdm import tqdm
import random
import tempfile

unique_phones = set()

def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
@hydra.main(config_path='config', config_name='config',version_base=None)
def prepare_dataset_as_json(cfg):
    output_stream = Path(cfg.ulm.librilight.output_jsonl_path).open('w')
    backend = EspeakBackend('en-us',logger=get_logger(verbosity='quiet'),words_mismatch='ignore')
    manifest_paths = list(Path(cfg.ulm.librilight.manifest_root).glob(cfg.ulm.librilight.manifest_pattern))
    for sub_paths in tqdm(divide_chunks(manifest_paths,10)):
        speech_by_speaker = dict()
        feature_paths = list(Path(cfg.ulm.librilight.feature_root).glob(cfg.ulm.librilight.feature_pattern))
        feature_paths = [x for x in feature_paths if x.stem.split('.')[0] in [y.stem.split('.')[0] for y in sub_paths]]
        features = dict()
        for feature_path in feature_paths:
            with open(feature_path) as f:
                lines = [l.strip() for l in f.readlines()]
                for line in lines:
                    features[line.split('|')[0].strip()] = line.split('|')[1].strip().split()
        manifest = lhotse.load_manifest(str(sub_paths[0]))
        for manifest_path in sub_paths[1:]:
            manifest: lhotse.CutSet=  manifest + lhotse.load_manifest(manifest_path) 
        for x in tqdm(manifest):
            wav_path = f"{x.id}.flac"
            normalized_text = x.supervisions[0].custom['texts'][0]
            try:
                data = features[wav_path]
            except:
                print('skpping', wav_path, manifest_path)
                continue
            speaker = x.id.split('/')[1]
            phones = backend.phonemize([normalized_text])[0]
            phones = phones.replace(' ','|')
            if speaker not in speech_by_speaker.keys():
                speech_by_speaker[speaker]=[]
            speech_by_speaker[speaker].append({
                'wav_path':wav_path,
                'text':normalized_text,
                'phones':phones,
                'data': data,
                'speaker': speaker
            })
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
                            {'text':f"<s> {' '.join(list(prompt_phones))} {' '.join(list(phones))} [SEP] {' '.join(prompt_data)} {' '.join(data)} </s>"}
                        ,ensure_ascii=False)
                    )
                    output_stream.write('\n')
    output_stream.close()
    return unique_phones

if __name__ == "__main__":
    phones = prepare_dataset_as_json()
