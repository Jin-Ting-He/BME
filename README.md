# Blur Magnitude Estimator(BME)

## Prepare BME dataset
```bash
python generate_dataset/generate_dataset.py
```

## Dataset Structure
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

## Inference
```bash
python main.py --infer_dataset_path="your inference dataset"  --infer_output_path="your output folder path"  --weight_path="model weight path" --test_only
```