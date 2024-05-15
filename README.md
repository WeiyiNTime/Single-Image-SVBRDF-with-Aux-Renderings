# Single-Image-SVBRDF-with-Aux-Renderings

## Set up environment
To set up environment, please run this command below:
```
conda env create -f env.yml
```
## About folders
1. trained_models:  they are pre-trained models
2. test_data: we provide testing images from Des18, Des19, real images, and high-resolution real images
3. out_results: some results are shown at here

## Inference

To test Des18:
python val.py --modelPath ./trained_models --outDir ./out_result --inDir ./test_data/Des18/input --gtDir ./test_data/Des18/gt-material --TestType Des18

To test Des19:
python val.py --modelPath ./trained_models --outDir ./out_result --inDir ./test_data/Des19/input --gtDir ./test_data/Des19/gt-material --TestType Des19

To test real:
python val.py --modelPath ./trained_models --outDir ./out_result --inDir ./test_data/real --TestType real

To test high resolution:
python val.py --modelPath ./trained_models --outDir ./out_result --inDir ./test_data/high-resolution --TestType real
