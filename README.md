# Submission

## Submission Summary

* MDX Leaderboard A
	* Submission ID: 216209
	* Submitter: kim_min_seok
	* Final rank: 
	* Final scores:
	  |  SDR_song | SDR_bass | SDR_drums | SDR_other | SDR_vocals |
	  | :------: | :------: | :-------: | :-------: | :--------: |
	  |   0.00   |   0.00   |   0.00   |   0.00    |    0.00    |
	  	  
* MDX Leaderboard B
	* Submission ID: 216211
	* Submitter: kim_min_seok
	* Final rank: 
	* Final scores:
	  |  SDR_song | SDR_bass | SDR_drums | SDR_other | SDR_vocals |
	  | :------: | :------: | :-------: | :-------: | :--------: |
	  |   0.00   |   0.00   |   0.00   |   0.00    |    0.00    |

## Model Summary

* Data
  * All 203 tracks of the Moises dataset was used for training (no validation split)
  * Augmentation
    * Random chunking and mixing sources from different tracks ([1])
* Model
  * A 'multi-source' version of TFC-TDF U-Net[2, 3] with some architectural improvements, including Channel-wise Subband[4]
  * Final submission is an ensemble of 3 models with identical architecture and training procedure but with different random seeds 
* Noise-robust Training
  * Leaderboard A: Loss masking
      * Intuitively, data with high training loss is likely to be audio chunks with labelnoise
	  * For each training batch, discard (=don't use for weight update) batch elements with higher loss than some quantile 
		  * ex) only use half of the training batch for each weight update
  * Leaderboard B: Loss masking (along temporal dimension)
      * Compared to labelnoise, data with bleeding seemed to vary less in terms of the amount of noise
      * A more fine-grained masking method performed better (discarding temporal bins with high loss) 

[1] S. Uhlich, et al., "Improving music source separation based on deep neural networks through data augmentation and network blending", ICASSP 2017.

[2] W. Choi, et al. "Investigating u-nets with various intermediate blocks for spectrogram-based singing voice separation", ISMIR 2020.

[3] M. Kim, et al. “Kuielab-mdx-net: A two-stream neural network for music demixing”, MDX Workshop at ISMIR 2021.

[4] H. Liu, et al. "Channel-wise Subband Input for Better Voice and Accompaniment Separation on High Resolution Music", INTERSPEECH 2020.


# Reproduction

## How to reproduce the submission
- Run submit.sh after configuring [my_submission/user_config.py](my_submission/user_config.py)
	- Leaderboard A
		- set ```MySeparationModel = A```
	- Leaderboard B
		- set ```MySeparationModel = B```

## How to reproduce the training
- All code needed to reproduce training is in [my_submission/src](my_submission/src)
- See [HOW_TO_TRAIN.md](my_submission/src/HOW_TO_TRAIN.md)
