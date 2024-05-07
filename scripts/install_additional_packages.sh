#First,  comment “convert_2to3_doctests” lines in  /home/ubuntu/anaconda3/envs/mcmr/lib/python3.6/sitepackages/setuptools/command/build_py.py

#conda activate mcmr  #sometimes 'conda activate' doesn't work in bash shell
source activate mcmr 


# adds packages for Soft Render that will be installed later
export CUDA_HOME=/usr/local/cuda
python setup_softras.py build develop       # install to workspace



#install pytorch3D and previous requirements for SoftRender
conda install -c bottler nvidiacub
pip install fvcore iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.4.0"

#install ligigl for geoemtry processign  (more simple than Pytorch3D)
conda install -c conda-forge igl

#installing packages for jupyter notebook 
conda deactivate 
conda create --name mcmr_ipy --clone mcmr
conda source mcmr_ipy
conda install ipykernel #another option to open 'mcmr_ipy'  in VS code and press 'install ipy kernel'
pip install --upgrade jupyter_client #doesn't work without this line 
conda install -c conda-forge meshplot 
# open a notebook ipynb file in VS code, it will  install  3D tools for interactive visualion 

#install open3d package (for icp)
# pip install open3d
#it should include pyvista package (for voxelization)
