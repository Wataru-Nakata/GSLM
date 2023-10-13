from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
import hydra
from pathlib import Path
import json


@hydra.main(config_path='config', config_name='config',version_base=None)
def prepare_tokenizer(cfg):
    vocab = {
        "<s>": 0,
    }
    num_k_means_cluster = cfg.preprocess.kmeans.cfg.k

    for i in range(num_k_means_cluster):
        vocab[f"{i}"] = i + 1
    vocab['[UNK]'] = len(vocab.keys())
    pretokenizer = Whitespace()
    model = WordLevel(vocab=vocab,unk_token="[UNK]")
    tokenizer = Tokenizer(model=model)
    tokenizer.pre_tokenizer = pretokenizer
    
    tokenizer.save(cfg.ulm.tokenizer_path)
@hydra.main(config_path='config', config_name='config',version_base=None)
def prepare_dataset_as_json(cfg):
    dataset_folder = Path(cfg.ulm.dataset.folder)
    train_json_stream = Path(cfg.ulm.output_json.train).open('w')
    valid_json_stream = Path(cfg.ulm.output_json.valid).open('w')
    test_json_stream = Path(cfg.ulm.output_json.test).open('w')
    data_count = 0
    output_stream = test_json_stream
    for dataset_file in dataset_folder.glob(cfg.ulm.dataset.pattern):
        with dataset_file.open() as f:
            for line in  f.readlines():
                if data_count > cfg.ulm.num_data.test:
                    output_stream = valid_json_stream
                if data_count > cfg.ulm.num_data.test + cfg.ulm.num_data.valid:
                    output_stream = train_json_stream
                data = line.strip().split('|')[-1]
                output_stream.write(
                    json.dumps(
                        {'text':f"<s> {data}"}
                    )
                )
                output_stream.write('\n')
                data_count += 1
    train_json_stream.close()
    valid_json_stream.close()
    test_json_stream.close()



if __name__ == "__main__":
    prepare_tokenizer()
    prepare_dataset_as_json()