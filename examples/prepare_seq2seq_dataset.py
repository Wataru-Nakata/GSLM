import hydra
from pathlib import Path
import pandas as pd
import json
from phonemizer.backend import EspeakBackend

@hydra.main(config_path='config', config_name='config',version_base=None)
def main(cfg):
    root = Path(cfg.ulm.ljspeech.manifest_root)
    df = pd.read_csv(root/'metadata.csv',sep='|',names=['id','text','norm_text'],quotechar=None,quoting=3)
    backend = EspeakBackend('en-us')
    for subset in ['train', 'dev', 'test']:
        ids = []
        manifest_path = root/ f"manifest_{subset}.{cfg.ulm.ljspeech.feature_file_suffix}"
        with manifest_path.open() as f: 
            lines = f.readlines()
        ids = [Path(line.split('|')[0]).stem for line in lines]
        features = [line.split('|')[1].strip() for line in lines]
        feature_dict = dict(zip(ids,features))
        subset_df = df[df['id'].isin(ids)]
        output_path = Path(cfg.ulm.ljspeech.dataset_path)
        output_path.mkdir(exist_ok=True)
        with (output_path/f"{subset}.json").open('w') as f:
            for idx,item in subset_df.iterrows():
                id = item['id']
                text = item['text']
                norm_text = item['norm_text']
                feature = feature_dict[id]
                phones = backend.phonemize([norm_text])[0]
                phones = phones.replace(' ','|')
                phones = ' '.join(list(phones))
                f.write(
                    json.dumps(
                        {
                            'id':id,
                            'text':text,
                            'norm_text':norm_text,
                            'phones': phones,
                            'feature':feature
                        }
                    ,ensure_ascii=False)
                )
                f.write('\n')
if __name__ == "__main__":
    main()
