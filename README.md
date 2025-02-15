# Blur Magnitude Estimator(BME)
This task aims to estimate the blur magnitude for each pixel from a blurry image as below example. The details will be in our [ECCV 2024 Paper](https://arxiv.org/abs/2407.09059). The downstream task is our [DADeblur](https://github.com/Jin-Ting-He/DADeblur) for video deblurring.
<p align="center">
  <img src="assets/blur_img.png" alt="Blurry Image" width="45%" style="margin-right: 10px;">
  <img src="assets/blur_mag.png" alt="Blur Magnitude" width="45%">
</p>
<!-- ![Blurry Image](assets/blur_img.png)
![Blur Magnitude](assets/blur_mag.png) -->

## Prepare BME dataset
In this task, we use the [RAFT](https://github.com/princeton-vl/RAFT) and [GoPro](https://seungjunnah.github.io/Datasets/gopro.html) to generate training dataset for BME. The details will be in our [ECCV 2024 Paper](https://arxiv.org/abs/2407.09059)

```bash
python generate_dataset/generate_dataset.py
```

## Pretrained Model Weight
You can download the our model weight from this link
[BME Model Weight](https://drive.google.com/file/d/1ASHteVTpdjI3CPBULtfMPsJyArVlnw6k/view?usp=sharing)

## Dataset Structure
```bash
dataset/
├── video1/
│ ├── blur_image/
│ └── blur_mag_np/
├── video2/
│ ├── blur_image/
│ └── blur_mag_np/
├── video3/
│ ├── blur_image/
│ └── blur_mag_np/
```

## Train
```bash
python main.py --training_dataset_path="your training dataset" --testing_dataset_path="your testing dataset" --weight_path="weight output path"
```

## Inference
```bash
python main.py --infer_dataset_path="your inference dataset"  --infer_output_path="your output folder path"  --weight_path="model weight path" --test_only
```