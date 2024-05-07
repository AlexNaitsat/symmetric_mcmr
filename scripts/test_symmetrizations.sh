source activate  mcmr_ipy

# car 10 network 
# #losses for non symmetric network 
# bash scripts/test_mcmr_pascal3d.sh car_10_MRCNN 10  car  small "--qualitative_results --symmetrize 0" non_symmetric 

# #losses for different type of symmetrization 
# bash scripts/test_mcmr_pascal3d.sh car_10_MRCNN 10  car  small "--qualitative_results --symmetrize 1" symmetric_1 
# bash scripts/test_mcmr_pascal3d.sh car_10_MRCNN 10  car  small "--qualitative_results --symmetrize 2" symmetric_2 
# bash scripts/test_mcmr_pascal3d.sh car_10_MRCNN 10  car  small "--qualitative_results --symmetrize 3" symmetric_3 

# aeroplane network
#datasize="small" # 10% of data 
datasize="_"     # 100% of data  
#datasize="tiny"     # 100% of data  


# #  ===== aeroplane car network "aeroplane car"  =====
# bash scripts/test_mcmr_pascal3d.sh plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 0" non_symmetric& 
# bash scripts/test_mcmr_pascal3d.sh plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 1" symmetric_1&
# bash scripts/test_mcmr_pascal3d.sh plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 2" symmetric_2&
# bash scripts/test_mcmr_pascal3d.sh plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 3" symmetric_3&
# # addtional symmetrization with top/buttom slices
# bash scripts/test_mcmr_pascal3d.sh  plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 1 2" symmetric_1_2&
# bash scripts/test_mcmr_pascal3d.sh  plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 1 3" symmetric_1_3&
# bash scripts/test_mcmr_pascal3d.sh  plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 2 1" symmetric_2_1&
# bash scripts/test_mcmr_pascal3d.sh  plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 2 3" symmetric_2_3&
# bash scripts/test_mcmr_pascal3d.sh  plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 3 1" symmetric_3_1&
# bash scripts/test_mcmr_pascal3d.sh  plane_car_2_PointRend 2  "aeroplane car"  $datasize   "--qualitative_results --symmetrize 3 2" symmetric_3_2&



# #==========================  aeroplane network =====================
# # constant semmytrization for the whole mesh
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 0" non_symmetric& 
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 1" symmetric_1& 
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 2" symmetric_2& 
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 3" symmetric_3& 

# # aeroplane network  with top/bottom slices 
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 1 2" symmetric_1_2
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 1 3" symmetric_1_3
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 2 1" symmetric_2_1
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 2 3" symmetric_2_3
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 3 1" symmetric_3_1
# bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  $datasize   "--qualitative_results --symmetrize 3 2" symmetric_3_2



# #======networkd trained on all classes with 12 shapes ===============================
# # constant semmytrization for the whole mesh
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 0" non_symmetric& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 1" symmetric_1& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 2" symmetric_2& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 3" symmetric_3&

# # # different symmetrization for top/bottom slices 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 1 2" symmetric_1_2& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 1 3" symmetric_1_3& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 2 1" symmetric_2_1& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 2 3" symmetric_2_3& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 3 1" symmetric_3_1& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 3 2" symmetric_3_2& 


# #===== networkd trained on all classes with 1 shape ==============
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend   1  "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 0" non_symmetric& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend   1  "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 1" symmetric_1& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend   1  "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 2" symmetric_2& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend   1  "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 3" symmetric_3&

# # different symmetrization for top/bottom slices 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend  1 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 1 2" symmetric_1_2& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend  1 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 1 3" symmetric_1_3& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend  1 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 2 1" symmetric_2_1& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend  1 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 2 3" symmetric_2_3& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend  1 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 3 1" symmetric_3_1& 
# bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend  1 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  $datasize   "--qualitative_results --symmetrize 3 2" symmetric_3_2& 




# # ============= bicycle bus car motorbike 1 meanshape  simple symmetrization =====================
# bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 0" non_symmetric& 
# bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 1" symmetric_1& 
# bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 2" symmetric_2& 
# bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 3" symmetric_3&
# # different symmetrization for top/bottom slices 
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 1 2" symmetric_1_2&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 1 3" symmetric_1_3&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 2 1" symmetric_2_1&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 2 3" symmetric_2_3&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 3 1" symmetric_3_1&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 3 2" symmetric_3_2

# # ============ bicycle bus car motorbike 4 meanshape ===================
# bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 0" non_symmetric& 
# bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 1" symmetric_1& 
# bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 2" symmetric_2& 
# bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 3" symmetric_3&
wait 
# # different symmetrization for top/bottom slices 
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 1 2" symmetric_1_2&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 1 3" symmetric_1_3&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 2 1" symmetric_2_1&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 2 3" symmetric_2_3&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 3 1" symmetric_3_1&
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4   "bicycle bus car motorbike"  $datasize   "--qualitative_results --symmetrize 3 2" symmetric_3_2

# # ====== car network  ========
# bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 0" non_symmetric 
# bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 1" symmetric_1 
# bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 2" symmetric_2 
# bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 3" symmetric_3
# addtional symmetrization with top/buttom slices

wait 
bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 1 2" symmetric_1_2&
bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 1 3" symmetric_1_3&
bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 2 1" symmetric_2_1&
bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 2 3" symmetric_2_3&
bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 3 1" symmetric_3_1&
bash scripts/test_mcmr_pascal3d.sh  car_10_MRCNN 10  "car"  $datasize   "--qualitative_results --symmetrize 3 2" symmetric_3_2


#Post processing evaluation of different metrics for different symmetrization types 
wait 
conda deactivate 
source activate  mcmr_p3d
python auxilary/test_postprocessing_par.py
