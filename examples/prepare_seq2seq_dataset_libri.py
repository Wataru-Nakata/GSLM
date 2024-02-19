import hydra
from pathlib import Path
import pandas as pd
import json
from phonemizer.backend import EspeakBackend
from tqdm import tqdm
import random

@hydra.main(config_path='config', config_name='config',version_base=None)
def main(cfg):
    dataset_folder = Path(cfg.ulm.dataset.folder)
    Path(cfg.ulm.output_json_tts_t5.train).parent.mkdir(exist_ok=True,parents=True)
    train_json_stream = Path(cfg.ulm.output_json_tts_t5.train).open('w')
    valid_json_stream = Path(cfg.ulm.output_json_tts_t5.valid).open('w')
    test_json_stream = Path(cfg.ulm.output_json_tts_t5.test).open('w')
    root = Path("/mnt/hdd/datasets/libritts")
    train_dataset_files = dataset_folder /cfg.ulm.dataset.train.pattern
    dev_dataset_files = dataset_folder/ cfg.ulm.dataset.dev.pattern
    test_dataset_files = dataset_folder/ cfg.ulm.dataset.test.pattern
    backend = EspeakBackend('en-us')
    for output_stream,dataset_file in [(train_json_stream,train_dataset_files),(valid_json_stream,dev_dataset_files),(test_json_stream,test_dataset_files)]:
        speeches = dict()
        with dataset_file.open() as f:
            for line in  tqdm(f.readlines()):
                data = line.strip().split('|')[-1]
                path = line.strip().split('|')[0]
                wav_path = root/path
                with wav_path.with_suffix('.normalized.txt').open() as f:
                    normalized_text = f.read().strip()
                    phones = backend.phonemize([normalized_text])[0]
                    phones = phones.replace(' ','|')
                speeches[path] = {
                    'id': wav_path.stem,
                    'wav_path':wav_path,
                    'text':normalized_text,
                    'phones':phones,
                    'data': data,
                    'speaker': wav_path.stem.split('_')[0]
                }
            speech_by_speaker = dict()
            for speech in speeches.values():
                speaker = speech['speaker']
                if speaker not in speech_by_speaker.keys():
                    speech_by_speaker[speaker] = []
                speech_by_speaker[speaker].append(speech)

            for speaker in tqdm(speech_by_speaker.keys()):
                for speech in speech_by_speaker[speaker]:
                    same_speaker_speeches = speech_by_speaker[speaker]
                    if len(same_speaker_speeches) == 1:
                        continue
                    for random_speech in random.choices(same_speaker_speeches,k=10):
                        if random_speech['wav_path'] == speech['wav_path']:
                            continue
                        prompt_phones = random_speech['phones']
                        prompt_data = random_speech['data']
                        data = speech['data']
                        phones = speech['phones']
                        speech_prompt = " ".join([f"speech_{y}" for y in prompt_data.split(' ')])
                        output_stream.write(
                            json.dumps(
                                {
                                    'phones':f"{speech_prompt} <extra_id_0> {' '.join(list(phones))}",
                                    'feature': f"{data}",
                                    'id' :speech['id'],
                                    'prompt_data': random_speech['id'],
                                    'orig_text': speech['text'],
                                    'orig_prompt_text': random_speech['text'],
                                }
                            ,ensure_ascii=False)
                        )
                        output_stream.write('\n')
    train_json_stream.close()
    valid_json_stream.close()
    test_json_stream.close()
if __name__ == "__main__":
    main()
