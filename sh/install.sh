pip install torch==1.11+cu113 torchvision==0.12+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install keras==2.7.0
pip install opencv-python==4.5.5.64
pip install pandas==1.3.1
pip install scikit-learn==0.24.2
pip install scikit-image==0.18.1#安装会卸载pillow(由于imageio),安装更高版本
pip install imageio==2.18.0
pip install Pillow==8.2.0#imageio要求>=8.3.2,torchvision不支持8.3,因此实际安装8.4
pip install tqdm==4.61.0
pip install pyyaml==5.4.1
pip install tensorboard==2.7.0
pip install Kornia==0.5.0

pip install matplotlib==3.5.1
pip install scipy==1.3.1#会报错,因此选择采用conda install scipy==1.3.1安装

# for visualization only
pip install seaborn==0.11.2
## Shapely Value
pip install shap==0.40.0
## Grad-CAM
pip install grad-cam==1.3.9
## Feature Map & Feature Visualization
pip install omnixai==1.2.3
pip install plotly==5.11.0
## UMAP
pip install umap-learn==0.5.3
## Network Structure
pip install graphviz#conda install graphviz
pip install hiddenlayer==0.3
#pip install -U git+https://github.com/szagoruyko/pytorchviz.git@master
pip install torchviz
## Landscape
pip install PyHessian==0.1

## Quality
pip install torchmetrics[image]

pip install pytorch-wavelets
