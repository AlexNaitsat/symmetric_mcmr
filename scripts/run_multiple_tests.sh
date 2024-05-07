
# Run small scale tests and save all qualitetive results 
bash scripts/test_mcmr_pascal3d.sh car_10_MRCNN 10  car small --qualitative_results
bash scripts/test_mcmr_pascal3d.sh plane_car_1_PointRend 1  "aeroplane car" small --qualitative_results
bash scripts/test_mcmr_pascal3d.sh plane_car_2_PointRend 2  "aeroplane car" small --qualitative_results
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1  "bicycle bus car motorbike" small --qualitative_results
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4  "bicycle bus car motorbike" small  --qualitative_results



# Run full scale tests and save all qualitetive results 
bash scripts/test_mcmr_pascal3d.sh car_10_MRCNN 10  car _ --qualitative_results
bash scripts/test_mcmr_pascal3d.sh plane_car_1_PointRend 1  "aeroplane car" _ --qualitative_results
bash scripts/test_mcmr_pascal3d.sh plane_car_2_PointRend 2  "aeroplane car" _ --qualitative_results
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_1_PointRend 1 "bicycle bus car motorbike" _ --qualitative_results
bash scripts/test_mcmr_pascal3d.sh bicycle_bus_car_bike_4_PointRend 4 "bicycle bus car motorbike" _ --qualitative_results

# #create small data index files for remained classes 
# bash scripts/fix_path_in_index_files.sh   /home/ubuntu/research/datasets/Pascal3D  "boat  bottle chair diningtable sofa train tvmonitor" "test train eval"
# bash scripts/sample_from_index_files.sh   /home/ubuntu/research/datasets/Pascal3D  "boat  bottle chair diningtable sofa train tvmonitor" "test train eval" small 10 10


#bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  small  --qualitative_results

#networkd trained on all classes 
bash scripts/test_mcmr_pascal3d.sh  plane_8_MRCNN  8  "aeroplane"  _  --qualitative_results
bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_1_PointRend   1  "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  _  --qualitative_results
bash scripts/test_mcmr_pascal3d.sh Pascal3D_12_classes_12_PointRend  12 "aeroplane bicycle boat bottle bus car chair diningtable motorbike sofa train tvmonitor"  _  --qualitative_results
