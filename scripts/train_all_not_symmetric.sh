source activate mcmr


python main.py --dataset_name pascal --dataset_dir ~/research/datasets/Pascal3D \
       --classes car --sub_classes --cmr_mode --subdivide 4 \
       --sdf_subdivide_steps 351 --use_learned_class --num_learned_shapes 1 \
       --checkpoint_dir checkpoint/all_small_not_symmetric --log_dir log/all_small_not_symmetric \
       --pretrained_weights weights/Pascal3D_12_classes_1_PointRend.pth \
       --cam_loss_wt 20.0   --cam_reg_wt 0.1 --mask_loss_wt 100.0 --deform_reg_wt 0.05 \
       --laplacian_wt 6.0 --laplacian_delta_wt 1.8 --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 \
       --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 --is_training --num_epochs 650 \
       --display_freq 50 --data_index_name small \
