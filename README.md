# LJMU_Thesis
LJMU thesis work, focus on medical segmentation with ViTs

## Environment setup
```
    conda create -n FFESNet python=3.10 ipython
    conda activate FFESNet

    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    pip install timm
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
    pip install matplotlib seaborn scikit-learn scikit-image jupyter 
```