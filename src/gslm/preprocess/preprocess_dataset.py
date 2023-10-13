import torch
from torch.utils.data.dataset import Dataset
import torchaudio
from pathlib import Path
import random
import string

def generate_random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

class GlobWavDataset(Dataset):
    def __init__(self, roots, patterns, shuffled: bool = True,add_random_string=True) -> None:
        self.wav_files = []
        for root,pattern in zip(roots,patterns):
            self.root = Path(root)
            self.wav_files.extend(list(self.root.glob(pattern)))
        if shuffled:
            random.shuffle(self.wav_files)
        self.add_random_string = add_random_string

    def __len__(self):
        return len(self.wav_files)


    def __getitem__(self,idx):
        wav_path = self.wav_files[idx]
        if self.add_random_string:
            return wav_path.stem + generate_random_string(5), torchaudio.load(wav_path)
        else:
            return wav_path.stem , torchaudio.load(wav_path),str(wav_path)
class HQYoutubeDataset(Dataset):
    def __init__(self,root) -> None:
        super().__init__()

        self.root = Path(root)
        score_paths = (self.root / 'score').glob("**/*.csv")

        self.audio_elements = []
        total_secs = 0
        for score_path in score_paths:
            with score_path.open() as f:
                lines = f.readlines()
                for line in lines: 
                    utt_id,sound_label,start_sec,end_sec,nisqa_mos,nisqa_noise,nisqa_discont,nisqa_color,nisqa_loud = line.strip().split(',')
                    if sound_label == 'voice':
                        self.audio_elements.append(
                            {
                                "wav_path": self.root/'wav24k/original'/score_path.stem[:2]/ (score_path.with_suffix(".wav")).name,
                                "utt_id": utt_id,
                                "label" :sound_label,
                                "start_sec": start_sec,
                                "end_sec": end_sec,
                                "nisqa_mos": nisqa_mos,
                                "nisqa_noise": nisqa_noise,
                                "nisqa_discont": nisqa_discont,
                                "nisqa_color" : nisqa_color,
                                "nisqa_loud" : nisqa_loud
                            }
                        )
                        total_secs += float(end_sec) - float(start_sec)
        print('total_sec', total_secs)
        raise ValueError
    def __len__(self):
        return len(self.audio_elements)


    def __getitem__(self,idx):
        try:
            audio_element = self.audio_elements[idx]
            info = torchaudio.info(audio_element['wav_path'])
            sample_rate = info.sample_rate
            wav,sr = torchaudio.load(
                audio_element['wav_path'],
                frame_offset= int(float(audio_element['start_sec'])* sample_rate),
                num_frames = int((float(audio_element['end_sec']) - float(audio_element['start_sec']))*sample_rate)
            )
            wav= wav[0]
            return audio_element['utt_id'], ( wav.view(1,-1), sr)
        except:
            print(self.audio_elements[idx])
            return "error", ( 'error', 'error')


        



