#installing main requirments
conda create -n mcmr python=3.6
#conda activate mcmr  #sometimes 'conda activate' doesn't work in bash shell
source activate mcmr 

pip install -r requirements.txt --ignore-installed

# install pytorch with corresponding cuda version
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit==10.2 -c pytorch

#installing SoftRas Renderer 
export CUDA_HOME=/usr/local/cuda
python setup_softras.py build develop       # install to workspace