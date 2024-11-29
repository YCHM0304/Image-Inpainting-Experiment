# Image Inpainting

- [Image Inpainting](#image-inpainting)
  - [Environment Building](#environment-building)
  - [Dataset](#dataset)
  - [Pretained Model](#pretained-model)
  - [Training](#training)
    - [Hyperparameters setting](#hyperparameters-setting)
    - [Training](#training-1)

## Environment Building


I personally use anaconda to manage my environment. You can create a new environment with the following command:

```bash
conda create -n ImageInpainting python=3.9
```

The name of the environment is set to `ImageInpainting`, so to activate the environment, use the following command:

```bash
conda activate ImageInpainting
```

> [!IMPORTANT]
> From now on, all the commands should be executed in the environment you just created.


Then, install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

Run the following shell script to download dataset:

```bash
./scripts/download_data.sh
```

## Pretained Model
We provide a pretrained model for the image inpainting task. You can download the model by running the following command:

```bash
./scripts/download_pretrained.sh
```

## Training

### Hyperparameters setting

To modify the hyperparameters, you can change the values in the `config.json` file.

### Training

To train the model, run the following command:

```bash
python ./train.py
```