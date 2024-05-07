#conda activate mcmr
source activate mcmr 

pip install --upgrade --no-cache-dir gdown #command line downloader for google drive 
cd ~/research/mcmr/
mdkir -p weights

cd weights
#PasCAl3D+ models
gdown --id 1XZIpDJNyPQa3IFDaDiqFf8ClCSUK0Ck_ -O plane_8_MRCNN.pth #airplane 8 meanshapes (MASKRCNN)
gdown --id 1r03Ci63J5FpyDxsGYOva_DSYqNw15MLQ -O car_10_MRCNN.pth #car 10 meanshapes

gdown --id 1m2ff_wkPEh1hLN-hsBoXCMmcjxlbs6p2 -O plane_car_2_MRCNN.pth #plane and car 2 meanshapes
gdown --id 18xEQC500Yp_nIt-WHMYpx7mDbITYf9hd -O plane_car_1_PointRend.pth #plane and car 1 meanshape
gdown --id 14e4oi2nUfxSauRgG5BaNxPG_Rjwm_hkV -O plane_car_2_PointRend.pth #plane and car 2 meanshape

gdown --id 1dAetRjglUnjbeCUbEa5uC0P8gorqPssN -O bicycle_bus_car_bike_1_PointRend.pth #Bicycle,Bus,Car,Motorbike - 2 meanshape
gdown --id 1TFKrxbH_bCxvt75wIZqWIxrzTnRtxVdU -O bicycle_bus_car_bike_4_PointRend.pth #Bicycle,Bus,Car,Motorbike - 4 meanshape

gdown --id 13XuQ4A7cunDZQ4w-GwLqzNyvttQxXhAY -O Pascal3D_12_classes_1_PointRend.pth #12 Pascal3D+ classes - 1 meanshape
gdown --id 1AbAMASl62PJCNkG7VrpOOgyZI9JSeY2o -O Pascal3D_12_classes_12_PointRend.pth #12 Pascal3D+ classes - 12 meanshapes


#bird models from https://github.com/aimagelab/mcmr/blob/main/models/MODELZOO.md
gdown --id 1Nz5sZS7kXWqX1A2_g3LyzQzsb8AoOiwz -O bird_1_net_latest.pth #1 meanshape 
gdown --id 1PtKRzgGO7CrpIehBlWipj2SRPPBKTD10 -O bird_14_net_latest.pth #14 meanshape