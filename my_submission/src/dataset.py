import os
import random
import numpy as np
import torch
import soundfile as sf
import pickle
from tqdm import tqdm
from glob import glob

from utils import load_chunk

    
    
class MSSDatasets(torch.utils.data.Dataset):
    def __init__(self, config, data_root):       
        
        self.config = config
        
        self.instruments = instruments = config.training.instruments
        
        data_paths = [path for path in glob(data_root+'/*') if os.path.basename(path)[0]!='.' and os.path.isdir(path)]
        
        self.metadata = []
        for data_path in data_paths:
            metadata_path = data_path+'/metadata'
            try:
                metadata = pickle.load(open(metadata_path, 'rb'))
            except Exception:   
                print('Collecting metadata for', data_path)
                metadata = []
                track_paths = sorted(glob(data_path+'/*'))
                track_paths = [path for path in track_paths if os.path.basename(path)[0]!='.' and os.path.isdir(path)]
                for path in tqdm(track_paths, ncols=50, desc='Collecting metadata'):
                    length = len(sf.read(path+f'/{instruments[0]}.wav')[0])
                    metadata.append((path, length))
                pickle.dump(metadata, open(metadata_path, 'wb'))
            
            self.metadata += metadata
          
        self.chunk_size = config.audio.hop_length * (config.audio.dim_t-1)
               
    def __len__(self):
        return self.config.training.num_steps * self.config.training.batch_size
    
    def load_source(self, metadata, i):
        track_path, track_length = random.choice(metadata)
        source = load_chunk(track_path+f'/{i}.wav', track_length, self.chunk_size)
        return torch.tensor(source, dtype=torch.float32)
    
    def __getitem__(self, index):
        return torch.stack([self.load_source(self.metadata, i) for i in self.instruments])
    