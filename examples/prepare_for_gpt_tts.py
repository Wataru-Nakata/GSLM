from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers import Tokenizer
import hydra
from pathlib import Path
import json
import phonemizer
from phonemizer.backend import EspeakBackend
from tqdm import tqdm
import random

unique_phones = set()

@hydra.main(config_path='config', config_name='config',version_base=None)
def prepare_tokenizer(cfg):
    vocab = {
        "<s>": 0,
        "</s>": 1,
        "[SEP]": 2
    }
    num_k_means_cluster = 1000

    for i in range(num_k_means_cluster):
        vocab[f"{i}"] = i + 3
    for i in range(len(unique_phones)):
        vocab[f"{unique_phones.pop()}"] = i + 3 + num_k_means_cluster
    vocab['[UNK]'] = len(vocab.keys())
    pretokenizer = WhitespaceSplit()
    model = WordLevel(vocab=vocab,unk_token="[UNK]")
    tokenizer = Tokenizer(model=model)
    tokenizer.pre_tokenizer = pretokenizer
    print(tokenizer.encode("<s> 22 344 </s>").tokens)
    tokenizer.save(cfg.ulm.tokenizer_path)
@hydra.main(config_path='config', config_name='config',version_base=None)
def prepare_dataset_as_json(cfg):
    dataset_folder = Path(cfg.ulm.dataset.folder)
    train_json_stream = Path(cfg.ulm.output_json_tts.train).open('w')
    valid_json_stream = Path(cfg.ulm.output_json_tts.valid).open('w')
    test_json_stream = Path(cfg.ulm.output_json_tts.test).open('w')
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
                    [unique_phones.add(p) for p in phones]
                speeches[path] = {
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
                        output_stream.write(
                            json.dumps(
                                {'text':f"<s> {' '.join(list(prompt_phones))} {' '.join(list(phones))} [SEP] {prompt_data} {data} </s>"}
                            ,ensure_ascii=False)
                        )
                        output_stream.write('\n')
    train_json_stream.close()
    valid_json_stream.close()
    test_json_stream.close()
    return unique_phones



if __name__ == "__main__":
    phones = prepare_dataset_as_json()
    prepare_tokenizer()
