source activate  mcmr_ipy
#Input params:
# > bash  <weights_file_name> <num of mean shapes>  <classes to test>  <data index name> <additiona-flags>
# Examples:

# bash scripts/test_mcmr_pascal3d.sh car_10_MRCNN 10  car  small --qualitative_results
# bash scripts/test_mcmr_pascal3d.sh plane_car_1_PointRend 1  "aeroplane car" small
echo "bumber of inputs=$#"
echo "============================================================================================================================================================="
logdir=$1_$4 
if test "$#" -ge 6; then
    logdir="$1_$4-$6" 
fi

echo -e "python main.py  --dataset_name pascal --dataset_dir ~/research/datasets/Pascal3D \
                --cam_loss_wt 20.0 --cam_reg_wt 0.1 --mask_loss_wt 100.0 --deform_reg_wt 0.05 --laplacian_wt 6.0 --laplacian_delta_wt 1.8 \
                --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 \
                --sub_classes --cmr_mode --subdivide 4 --sdf_subdivide_steps 351  --use_learned_class \
                --num_learned_shapes $2 --pretrained_weights weights/$1.pth  --classes $3 \
                --checkpoint_dir checkpoint/$logdir --log_dir log/$logdir  --save_dir save/$logdir --save_results \
                --data_index_name $4 \
                $5" 
echo "============================================================================================================================================================="

# python main.py  --dataset_name pascal --dataset_dir ~/research/datasets/Pascal3D \
#                 --cam_loss_wt 20.0 --cam_reg_wt 0.1 --mask_loss_wt 100.0 --deform_reg_wt 0.05 --laplacian_wt 6.0 --laplacian_delta_wt 1.8 \
#                 --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 \
#                 --sub_classes --cmr_mode --subdivide 4 --sdf_subdivide_steps 351  --use_learned_class \
#                 --num_learned_shapes $2 --pretrained_weights weights/$1.pth  --classes $3\
#                 --checkpoint_dir checkpoint/$1_$4 --log_dir log/$1_$4  --save_dir save/$1_$4  --save_results \
#                 --data_index_name $4 \
#                 $5

python main.py  --dataset_name pascal --dataset_dir ~/research/datasets/Pascal3D \
                --cam_loss_wt 20.0 --cam_reg_wt 0.1 --mask_loss_wt 100.0 --deform_reg_wt 0.05 --laplacian_wt 6.0 --laplacian_delta_wt 1.8 \
                --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 \
                --sub_classes --cmr_mode --subdivide 4 --sdf_subdivide_steps 351  --use_learned_class \
                --num_learned_shapes $2 --pretrained_weights weights/$1.pth  --classes $3 \
                --checkpoint_dir checkpoint/$logdir --log_dir log/$logdir  --save_dir save/$logdir --save_results \
                --data_index_name $4 \
                $5