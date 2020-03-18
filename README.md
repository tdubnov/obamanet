# ObamaNet: Talking Presidents
	
## Requirements

You may install the requirements by running the following command

```
pip install --user -r requirements.txt
```

The tools below to extract and manipulate the data:

* ffmpeg
* [YouTube-dl](https://github.com/rg3/youtube-dl#video-selection)

## Data Extraction

We extract the president weekly addresses videos from youtube using `youtube-dl`

```sh
cd data
# Download Trump train videos (first 10 videos of trump_addresses.txt)
youtube-dl --batch-file trump/trump_addresses_train10.txt --sub-lang en --skip-download --write-sub --output 'trump/captions/%(autonumber)s.%(ext)s' --ignore-config
youtube-dl --batch-file trump/trump_addresses_train10.txt -o 'trump/videos/%(autonumber)s.%(ext)s' -f "best[height=720]" --autonumber-start 1

# Segment video by captions
python processing.py trump --trim True
python processing.py trump --extract_images True --extract_audio True
```

```sh
cd data
# Download Trump test video (video 50 of trump_addresses.txt)
youtube-dl --batch-file trump_test/trump_addresses_test.txt --sub-lang en --skip-download --write-sub --output 'trump_test/captions/%(autonumber)s.%(ext)s' --ignore-config
youtube-dl --batch-file trump_test/trump_addresses_test.txt -o 'trump_test/videos/%(autonumber)s.%(ext)s' -f "best[height=720]" --autonumber-start 1

# Segment video by captions. Trimming video takes ~8 minutes, extracting images and audios takes ~2 minutes.
python processing.py trump_test --trim True
python processing.py trump_test --extract_images True --extract_audio True
```

## Pix2pix

You may use [this](https://drive.google.com/open?id=1zKip_rlNY2Dk14fzzOHQm03HJNvJTjGT) pretrained model or train pix2pix from scratch using [this](https://drive.google.com/open?id=1sJBp5bYe3XSyE7ys5i7ABORFZctWEQhW) dataset. Unzip the dataset into the [pix2pix](https://github.com/affinelayer/pix2pix-tensorflow) main directory.

```sh
# Takes ~10 hours
cd data
python prepare_test_video.py trump
cd ..
python pix2pix.py --mode train --input_dir data/trump/pix2pix_input_lip --output_dir data/trump/pix_model_lip --which_direction AtoB --max_epochs 10

# Takes ~10 hours
cd data
python prepare_test_video.py trump --jaw True
cd ..
python pix2pix.py --mode train --input_dir data/trump/pix2pix_input_jaw --output_dir data/trump/pix_model_jaw --which_direction AtoB --max_epochs 10
```

```sh
cd data
python prepare_test_video.py trump_test
cd ..
python pix2pix.py --mode test --input_dir data/trump_test/pix2pix_input_lip --output_dir data/trump_test/pix2pix_output_lip --checkpoint data/trump/pix_model_lip

cd data
python prepare_test_video.py trump_test --jaw True
cd ..
python pix2pix.py --mode test --input_dir data/trump_data/pix2pix_input_jaw --output_dir data/trump_test/pix2pix_output_jaw --checkpoint data/trump/pix_model_jaw
```

__Generate test videos__

```sh
ffmpeg -r 30 -f image2 -s 256x256 -pattern_type glob -i 'data/trump_test/pix2pix_output_lip/images/*-outputs.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p trump_test_lip_output.mp4
ffmpeg -r 30 -f image2 -s 256x256 -pattern_type glob -i 'data/trump_test/pix2pix_output_jaw/images/*-outputs.png' -vcodec libx264 -crf 25 -pix_fmt yuv420p trump_test_jaw_output.mp4

cd data/trump_test/audios
# Combine audio files
ffmpeg -safe 0 -f concat -i <( for f in *.wav; do echo "file '$(pwd)/$f'"; done ) ../../../trump_test.wav

# Append audio to video
cd ../../..
ffmpeg -i trump_test_lip_output.mp4 -i trump_test.wav -c:v copy -c:a aac -strict experimental trump_test_lip.mp4
ffmpeg -i trump_test_jaw_output.mp4 -i trump_test.wav -c:v copy -c:a aac -strict experimental trump_test_jaw.mp4
rm trump_test_lip_output.mp4
rm trump_test_jaw_output.mp4
rm trump_test.wav
```

## Pix2pix HD

1. Run Workplace.pynb to split the pictures to Train_A and Train_B (and crop pictures accordingly)
2. Move the train_A and train_B folder to ~/pix2pixHD/dataset/[SOME_Name]

3. Run python train.py:
First time:
python train.py --name trumpChin --dataroot ./datasets/[SOME_Name]/ --label_nc 0 --loadSize 256 --fineSize 256 --no_instance
To Continue:
python train.py --name trumpChin --dataroot ./datasets/[SOME_Name]/ --label_nc 0 --loadSize 256 --fineSize 256 --no_instance --continue_train

4. To test it / run it on new images run
python test.py --name trumpChin --dataroot ./datasets/trumpChin/ --label_nc 0 --loadSize 256 --fineSize 256 --no_instance
where trumpChin is the model name and ./datasets/trumpFinal/  is the path to the directory with the images you want to run

5. To combine the images into a video run
ffmpeg -r 30 -f image2 -s 256x256 -pattern_type glob -i '00050*synthesized_image.jpg' -vcodec libx264 -crf 25 -pix_fmt yuv420p ../trump_test_jaw_hd_output.mp4

## References

Much of the code we use came from this GitHub repo: https://github.com/karanvivekbhargava/obamanet