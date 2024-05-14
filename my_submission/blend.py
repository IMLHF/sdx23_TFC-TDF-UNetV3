from my_submission.tfc_tdf import MusicSeparationModel as TFC_TDF
from my_submission.demucs import MusicSeparationModel as DEMUCS
import numpy as np
import soundfile as sf
import tqdm



blend_weights = [.2, .2, .6, .8]


class MusicSeparationModel:
    
    def __init__(self):     
        
        self.modelA = TFC_TDF()
        self.modelB = DEMUCS()
        self.blend_weights = {k:v for k,v in zip(self.instruments, blend_weights)}
               
    @property
    def instruments(self):
        """ DO NOT CHANGE """
        return ['bass', 'drums', 'other', 'vocals']

    def raise_aicrowd_error(self, msg):
        """ Will be used by the evaluator to provide logs, DO NOT CHANGE """
        raise NameError(msg)
    
    def separate_music_file(self, mixed_sound_array, sample_rate):
   
        a, sr = self.modelA.separate_music_file(mixed_sound_array, sample_rate)
        # print("modelA complete")
        b, sr = self.modelB.separate_music_file(mixed_sound_array, sample_rate)
        # print("modelB complete")
        
        sources = {i: a[i] * w + b[i] * (1-w) for i,w in self.blend_weights.items()} 
            
        return sources, sr
    
def test_one(wav_path, model):
    instruments = ['bass', 'drums', 'other', 'vocals']
    mixed_sound_array = sf.read(wav_path)[0]
    # print(mixed_sound_array.shape)
    sample_rate = 44100
    separated_music_arrays, output_sample_rates = model.separate_music_file(mixed_sound_array, sample_rate)
    for instrument in instruments:
        # print(instrument, separated_music_arrays[instrument].shape, output_sample_rates[instrument])
        sf.write(f'{wav_path[:-4]}-{instrument}-blend.wav', separated_music_arrays[instrument], output_sample_rates[instrument])

if __name__ == '__main__':
    model = MusicSeparationModel()

    paths = [
        # 'MUSDB18HQ-test/test/AM Contra - Heart Peripheral/mixture.wav',
        'MUSDB18HQ-test/stest/蔡琴-渡口.wav',
        'MUSDB18HQ-test/stest/李健-贝加尔湖畔.wav',
        'MUSDB18HQ-test/stest/王菲-红豆_01.wav',
        'MUSDB18HQ-test/stest/周杰伦-稻香.wav',
        'MUSDB18HQ-test/stest/周深-起风了.wav'
        ]

    for path in tqdm.tqdm(paths, ncols=50, desc="Processing"):
        test_one(path, model)