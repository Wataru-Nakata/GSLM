from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers import Tokenizer
import hydra
from pathlib import Path
import json


@hydra.main(config_path='config', config_name='config',version_base=None)
def prepare_tokenizer(cfg):
    vocab = {
        "<s>": 0,
        "</s>": 1,
    }
    num_k_means_cluster = 1000

    for i in range(num_k_means_cluster):
        vocab[f"{i}"] = i + 2
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
    train_json_stream = Path(cfg.ulm.output_json.train).open('w')
    valid_json_stream = Path(cfg.ulm.output_json.valid).open('w')
    test_json_stream = Path(cfg.ulm.output_json.test).open('w')
    train_dataset_files = dataset_folder /cfg.ulm.dataset.train.pattern
    dev_dataset_files = dataset_folder/ cfg.ulm.dataset.dev.pattern
    test_dataset_files = dataset_folder/ cfg.ulm.dataset.test.pattern
    for output_stream,dataset_file in [(train_json_stream,train_dataset_files),(valid_json_stream,dev_dataset_files),(test_json_stream,test_dataset_files)]:
        with dataset_file.open() as f:
            for line in  f.readlines():
                data = line.strip().split('|')[-1]
                output_stream.write(
                    json.dumps(
                        {'text':f"<s> {data} </s>"}
                    )
                )
                output_stream.write('\n')
    train_json_stream.close()
    valid_json_stream.close()
    test_json_stream.close()



if __name__ == "__main__":
    prepare_tokenizer()
    prepare_dataset_as_json()
