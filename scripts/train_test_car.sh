source activate mcmr
#small subset 
# car_10_MRCNN +  100 more train  epochs om 10% of data  (default learning rate 1e-4)
python main.py --dataset_name pascal --dataset_dir ~/research/datasets/Pascal3D \
       --classes car --sub_classes --cmr_mode --subdivide 4 \
       --sdf_subdivide_steps 351 --use_learned_class --num_learned_shapes 10 \
       --checkpoint_dir checkpoint/car_small_train_0.1lr --log_dir log/car_small_train_0.1lr \
       --pretrained_weights weights/car_10_MRCNN.pth \
       --cam_loss_wt 20.0   --cam_reg_wt 0.1 --mask_loss_wt 100.0 --deform_reg_wt 0.05 \
       --laplacian_wt 6.0 --laplacian_delta_wt 1.8 --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 \
       --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 --is_training --num_epochs 600 --display_freq 50 \
       --data_index_name small \
       --G_learning_rate 0.00001

# testing it 
python main.py  --dataset_name pascal  --dataset_dir ~/research/datasets/Pascal3D \
        --classes car --sub_classes --cmr_mode --subdivide 4 \
        --sdf_subdivide_steps 351 --use_learned_class --num_learned_shapes 10 \
        --checkpoint_dir checkpoint/car_small_test_0.1lr \
        --log_dir log/car_small_test_0.1lr \
        --pretrained_weights checkpoint/car_small_train/net_latest.pth \
        --cam_loss_wt 20.0 --cam_reg_wt 0.1 --mask_loss_wt 100.0 \
        --deform_reg_wt 0.05 --laplacian_wt 6.0 --laplacian_delta_wt 1.8 --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 \
        --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 \
        --save_dir save/car_small_test_0.1lr \
        --save_results --qualitative_results \
        --data_index_name small \
        --G_learning_rate 0.00001



#full dataset 
# car_10_MRCNN +  100 more train  epochs om 10% of data  (default learning rate 1e-4)
python main.py --dataset_name pascal --dataset_dir ~/research/datasets/Pascal3D \
       --classes car --sub_classes --cmr_mode --subdivide 4 \
       --sdf_subdivide_steps 351 --use_learned_class --num_learned_shapes 10 \
       --checkpoint_dir checkpoint/car__train_0.1lr --log_dir log/car_train_0.1lr \
       --pretrained_weights weights/car_10_MRCNN.pth \
       --cam_loss_wt 20.0   --cam_reg_wt 0.1 --mask_loss_wt 100.0 --deform_reg_wt 0.05 \
       --laplacian_wt 6.0 --laplacian_delta_wt 1.8 --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 \
       --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 --is_training --num_epochs 600 --display_freq 50 \
       --G_learning_rate 0.00001

# testing it 
python main.py  --dataset_name pascal  --dataset_dir ~/research/datasets/Pascal3D \
        --classes car --sub_classes --cmr_mode --subdivide 4 \
        --sdf_subdivide_steps 351 --use_learned_class --num_learned_shapes 10 \
        --checkpoint_dir checkpoint/car_test_0.1lr \
        --log_dir log/car_small_test_0.1lr \
        --pretrained_weights checkpoint/car_small_train/net_latest.pth \
        --cam_loss_wt 20.0 --cam_reg_wt 0.1 --mask_loss_wt 100.0 \
        --deform_reg_wt 0.05 --laplacian_wt 6.0 --laplacian_delta_wt 1.8 --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 \
        --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 \
        --save_dir save/car_test_0.1lr \
        --save_results --qualitative_results \
        --G_learning_rate 0.00001