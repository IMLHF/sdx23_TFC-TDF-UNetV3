### Setup environment

```bash
sudo apt-get install soundstretch
conda create -n mdx-net pip
conda activate mdx-net
pip install -r requirements.txt
```

### Pitch Shift and Time Stretch
- Before training, you need to store augmented data
    - The WSOLA-like algorithm used by Soundstretch was too slow to be applied on-the-fly.
    - On the other hand, the [Torchaudio implementation](https://pytorch.org/audio/main/generated/torchaudio.transforms.TimeStretch.html) (phase vocoder algorithm) was fast enough but did not improve SDR.
- Run ```python data_augmentation.py --data_dir ${your_musdb_path} --train True --test False```


### Training
- Run train.py **3 times** with the following arguments
    ```bash
    python train.py --data_root path/to/musdbHQ --model_path ../ckpts/model1 --device_ids 0 --num_steps 1200000
    ```
    ```bash
    python train.py --data_root path/to/musdbHQ --model_path ../ckpts/model2 --device_ids 0 --num_steps 700000
    ```
    ```bash
    python train.py --data_root path/to/musdbHQ --model_path ../ckpts/model3 --device_ids 0 --num_steps 1000000
    ```
    - for multi-gpu
        ```bash
        --device_ids 0 1 2 3
        ```
- You will need at least 48GB of GPU memory for the largest model ('model2'). The other 2 models need 24GB.
