source activate  mcmr
#Input params:
# > bash  <weights_file_name> <num of mean shapes>  <classes to test>  <data index name> <additiona-flags>
# Examples:

# bash scripts/test_mcmr_pascal3d.sh car_10_MRCNN 10  car  small --qualitative_results
# bash scripts/test_mcmr_pascal3d.sh plane_car_1_PointRend 1  "aeroplane car" small


# bash scripts/train_mcmr_pascal3d.sh plane_car_1_PointRend   "aeroplane car" small --symmetrize
# bash scripts/train_mcmr_pascal3d.sh plane_car_1_PointRend   "aeroplane car" small ""


arrIN=(${1//_/ })
shape_num = ${arrIN[-2]} 

##  Changes to support "--num_learned_shapes 1":
# if [ $shape_num = 1 ]
# then
#     shape_num_flag = "--single_mean_shape"
# else
#     shape_num_flag = "--num_learned_shapes --$shape_num"
# fi

num_learned_shape_flag =  $2
--num_learned_shapes $2
outdir = 


echo "============================================================================================================================================================="
echo -e "python main.py  --dataset_name pascal --dataset_dir ~/research/datasets/Pascal3D \
                --cam_loss_wt 20.0 --cam_reg_wt 0.1 --mask_loss_wt 100.0 --deform_reg_wt 0.05 --laplacian_wt 6.0 --laplacian_delta_wt 1.8 \
                --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 \
                --sub_classes --cmr_mode --subdivide 4 --sdf_subdivide_steps 351  --use_learned_class \
                --num_learned_shapes $shape_num --pretrained_weights weights/$1.pth  --classes $2 \
                --checkpoint_dir checkpoint/$1_$4 --log_dir log/$1_$4  --save_dir save/$1_$4  --save_results \
                --data_index_name $4 \
                $5" 
echo "============================================================================================================================================================="

python main.py  --dataset_name pascal --dataset_dir ~/research/datasets/Pascal3D \
                --cam_loss_wt 20.0 --cam_reg_wt 0.1 --mask_loss_wt 100.0 --deform_reg_wt 0.05 --laplacian_wt 6.0 --laplacian_delta_wt 1.8 \
                --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 \
                --sub_classes --cmr_mode --subdivide 4 --sdf_subdivide_steps 351  --use_learned_class \
                --num_learned_shapes $2 --pretrained_weights weights/$1.pth  --classes $3\
                --checkpoint_dir checkpoint/$1_$4 --log_dir log/$1_$4  --save_dir save/$1_$4  --save_results \
                --data_index_name $4 \
                $5


