source activate  mcmr
python main.py  --dataset_name pascal  --dataset_dir ~/research/datasets/Pascal3D \
        --classes car --sub_classes --cmr_mode --subdivide 4 \
        --sdf_subdivide_steps 351 --use_learned_class --num_learned_shapes 10 \
        --checkpoint_dir checkpoint/car_all_test \
        --log_dir log/car_all_test \
        --pretrained_weights checkpoint/car_small_train/net_latest.pth \
        --cam_loss_wt 20.0 --cam_reg_wt 0.1 --mask_loss_wt 100.0 \
        --deform_reg_wt 0.05 --laplacian_wt 6.0 --laplacian_delta_wt 1.8 --graph_laplacian_wt 0.0 --tex_percept_loss_wt 0.8 \
        --tex_color_loss_wt 0.03 --tex_pixel_loss_wt 0.005 \
        --save_dir save/car_all_test \
        --save_results --qualitative_results \
        --G_learning_rate 0.00001