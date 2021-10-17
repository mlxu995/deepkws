import os
import random
import pandas as pd
import torch
from speechbrain.lobes.augment import _prepare_csv
from speechbrain.dataio.dataio import read_audio


class AddNoise(torch.nn.Module):
    def __init__(self, noise_folder, noise_csv, noise_prob=0.8, noise_snr_low=0, noise_snr_high=15):
        # TODO add noise with a predefined SNR
        super().__init__()
        if not os.path.exists(noise_csv):
            # make noise csv
            noise_files = [file+'\n' for file in os.listdir(noise_folder) if file.endswith(".wav")]
            filelist = os.path.join(noise_folder, "noise_files.txt")
            with open(filelist, mode='w', encoding='utf-8') as output:
                output.writelines(noise_files)
            if not os.path.exists(os.path.dirname(noise_csv)):
                os.makedirs(os.path.dirname(noise_csv))
            _prepare_csv(noise_folder, filelist, noise_csv)

        self.noise_prob = noise_prob
        df = pd.read_csv(noise_csv)
        noise_files = [read_audio(file) for file in df['wav']]
        noise_len = [len(noise) for noise in noise_files]
        self.noise_files = torch.zeros((len(noise_files), max(noise_len)))
        for i, noise in enumerate(noise_files):
            self.noise_files[i] = torch.cat([noise, noise[:max(noise_len)-len(noise)]], dim=0)
        

    def forward(self, waveforms, lengths):
        self.noise_files = self.noise_files.to(device=lengths.device)
        b, n = waveforms.shape
        random_index = torch.randint(0, len(self.noise_files), (b, ), device=lengths.device)
        noise_audios = self.noise_files[random_index]
        max_chop = self.noise_files.shape[1] - n
        start_index = torch.randint(high=max_chop, size=(1,), device=lengths.device)
        noise_batch = noise_audios[:, start_index : start_index + n]
        if random.random() < self.noise_prob:
            a = random.random() * 0.1
            waveforms = torch.clip(a * noise_batch + waveforms, -1, 1)
        
        return waveforms