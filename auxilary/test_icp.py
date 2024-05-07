import numpy as np
import utils.geometry as gu
import igl
import glob
from auxilary.test_postprocessing import read_cad_mesh
#run it wuith mcmr_ipy
from scipy.spatial.transform import Rotation as R
from auxilary.test_postprocessing import  clean_CAD_format, SourceTargetMeshPipeline
from tqdm import tqdm

def check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, ICP_fun, args):
    
    num_of_checks = 2
    Vs_icp_list =[]
    alpha = np.pi/10
    #Rmat  = (R.from_quat([0, 0, np.sin(alpha), np.cos(alpha) ])).as_matrix()
    #random.seed(1234)
    
    #ICP_mean_norm_list      = []
    #ICP_max_deviation_list  = []
    #ICP_mean_deviation_list = [] 
    ICP_haus_dist_list = []
    ICP_mean_max_dist_list = []
    ICP_mean_mean_dist_list = []


    ICP_mean_randomness_list =[]
    ICP_max_randomness_list  =[]
    verbose = False
    for li in range(len(Vs_list)):
        Vs = Vs_list[li]
        Fs = Fs_list[li]
        Vt = Vt_list[li]
        Ft = Ft_list[li]

        for i in range(num_of_checks):
            Vs_icp, Rot, t = ICP_fun(Vs,Fs, Vt,Ft, *args)
            #Vs = np.matmul(Vs,Rmat)
            Vs_size = Vs.max() - Vs.min()
            #Vst_icp_diff = np.abs(Vs_icp - Vt)
            #ICP_mean_norm = np.linalg.norm(Vst_icp_diff)/(Vs.shape[0]*Vs_size)
            #ICP_mean_deviation = (Vst_icp_diff).mean()/Vs_size
            #ICP_max_deviation = Vst_icp_diff.max()/Vs_size
            ICP_haus_dist,ICP_mean_max_dist, ICP_mean_mean_dist = gu.max_mean_mesh_distance(Vs_icp,Fs, Vt, Ft)
            if verbose:
                print(f"Itertion {i}: \nQuality (metric/source_size):")
                print(f"    ICP_haus_dist   =  {ICP_haus_dist} ")
                print(f"    ICP_mean_max_dist =  {ICP_mean_max_dist} ")
                print(f"    ICP_mean_mean_dist  =  {ICP_mean_mean_dist} ")
                # print(f"Itertion {i}: \nQuality (metric/source_size):")
                # print(f"    mean norm of ICP   =  {ICP_mean_norm} ")
                # print(f"    mean ICP deviation =  {ICP_mean_deviation} ")
                # print(f"    max ICP deviation  =  {ICP_max_deviation} ")

            if i>0:
                Vs_prev_diff        = np.abs(Vs_icp_list[-1] -  Vs_icp)
                ICP_mean_randomness = Vs_prev_diff.mean()/Vs_size
                ICP_max_randomness  = Vs_prev_diff.max()/Vs_size

                if verbose:
                    print(f"Randomness w.r.t. prev iter. (metric/source_size):")
                    print(f"    max differ from prev iter by  {ICP_max_randomness} ")
                    print(f"    mean differ from prev iter by {ICP_mean_randomness} ")
                ICP_mean_randomness_list.append(ICP_mean_randomness)
                ICP_max_randomness_list.append(ICP_max_randomness)
                
            else:
                # ICP_mean_norm_list.append(ICP_mean_norm)
                # ICP_max_deviation_list.append(ICP_max_deviation)
                # ICP_mean_deviation_list.append(ICP_mean_deviation)
                ICP_haus_dist_list.append(ICP_haus_dist)
                ICP_mean_max_dist_list.append(ICP_mean_max_dist)
                ICP_mean_mean_dist_list.append(ICP_mean_mean_dist)
            Vs_icp_list.append(Vs_icp)
    mean_list = lambda lst: sum(lst) / len(lst)
    print(f"==================== Average metrics per sample ===================: ")
    # print(f"  mean norm of ICP measures    =  {mean_list(ICP_mean_norm_list)} ")
    # print(f"  mean norm of ICP   =  {mean_list(ICP_mean_norm_list)} ")
    # print(f"  mean ICP deviation =  {mean_list(ICP_mean_deviation_list)} ")
    # print(f"  max ICP deviation  =  {mean_list(ICP_max_deviation_list)} ")
    print(f"    ICP_mean_mean_dist =  {mean_list(ICP_mean_mean_dist_list)} ")
    print(f"    ICP_mean_max_dist  =  {mean_list(ICP_mean_max_dist_list)} ")
    print(f"    ICP_haus_dist      =  {mean_list(ICP_haus_dist_list)} ")

    print(f"    max randomness error = {mean_list(ICP_max_randomness_list)} ")
    print(f"    mean randomness error =  {mean_list(ICP_mean_randomness_list)} ")
    return ICP_mean_mean_dist_list, ICP_mean_max_dist_list, ICP_haus_dist_list,ICP_mean_randomness_list, ICP_max_randomness_list


def identity_transform(Vs,Fs, Vt,Ft, sample_num, iter_num,  enable_rescale = False):
    return np.copy(Vs), np.eye(3), np.zeros(3)


def distances_between_aligned_meshes(s_file_list,t_file_list, ICP_fun, ICP_args,  dist_fun_list, dist_args_list=[]):
    sample_num = len(s_file_list)
    dist_num = 0
    distances = None 
    for i in range(sample_num):
        Vs, Fs = igl.read_triangle_mesh(s_file_list[i])
        Vt, Ft = igl.read_triangle_mesh(t_file_list[i])
        Vs_icp, Rot, t = ICP_fun(Vs,Fs, Vt,Ft, *ICP_args)
        current_dist = []
        for fi in range(len(dist_fun_list)):
            dist_args = dist_args_list[fi] if i < len(dist_args_list) else () 
            dist_i = dist_fun_list[fi](Vs_icp,Fs, Vt, Ft, *dist_args)
            if type(dist_i) is list:
                current_dist +=  dist_i #one dist function can return several metrics 
            else:
                current_dist.append(dist_i)
                
        if distances is None:
            distances = -np.ones((sample_num, len(current_dist)))
        distances[i,:] = current_dist
    return distances

if __name__ == "__main__":
    #obj_file =[glob.glob('/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD/car/*.off')[0]] 
    icp_iter       = 10
    icp_sample_num = 2000
    enable_rescale = True

    ## run once per folder:
    #clean_CAD_format("../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri", ['car']) #after Blender and manual clean-ups
    print('=============== testing 3D IoU with oepn3d PC ICP =======================================================')

    t_files = [ "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_01_clean_solid_voxrem_dec_smooth_symm_tri.obj",\
                "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_02_clean_solid_voxrem_dec_smooth_symm_tri.obj",\
                "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_03_clean_solid_voxrem_dec_smooth_symm_tri.obj",\
                "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_04_clean_manual_symm_tri.obj",\
                "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_05_clean_manual_v4_symm_tri.obj",\
                "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_06_clean_manual_symm_tri.obj",\
                "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_07_clean_manual_v4_symm_tri.obj",\
                "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_08_solid_symm_tri.obj",\
                "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_09_clean_solid_voxrem_dec_symm_tri.obj",\
                "../datasets/Pascal3D+_release1.1/CAD_preprocessed_tri/car/car_10_clean_manual_v9_symm_tri.obj" ] 
    #s_files = ["draft/data/car_aligned.obj"] * len(t_files)
    s_files = ["draft/data/car.obj"] * len(t_files)
    print(f'\nUsing gu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')

    # dist= distances_between_aligned_meshes( s_files, t_files, \
    #                                         identity_transform, (icp_sample_num,icp_iter,enable_rescale), \
    #                                         [gu.Compute3DIoU, gu.relative_symmetric_mesh_distances], ()  ) 

    # print(f'\nMeasuring (3D IoU, hausdorf dist. , max-mean dist., mean mean dist.):\n {dist} \n ----------- \n average = {dist.mean(axis =0)}')

    metrics = SourceTargetMeshPipeline(s_files, t_files, \
                                 [(gu.align_open3d_PC_ICP, icp_sample_num,icp_iter,enable_rescale)], \
                                 [(gu.Compute3DIoU,), (gu.relative_symmetric_mesh_distances,)]  ) 
    print(f'\nMeasured (3D IoU, hausdorf dist. , max-mean dist., mean mean dist.):\n {metrics} \n ----------- \n average = {metrics.mean(axis =0)}')


    print(f'\nUsing gu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) and Symmetrization [igl mesh-ICP]')
    data = np.load("save/symmetric/car_10_MRCNN__/car-qualitative/shape_symmetries.npy", allow_pickle=True).item()
    

    metrics = SourceTargetMeshPipeline(s_files, t_files, \
                                [ (gu.symmetrize_source_mesh, data , True),\
                                  (gu.align_open3d_PC_ICP, icp_sample_num,icp_iter,enable_rescale),\
                                ], \
                                [(gu.Compute3DIoU,), (gu.relative_symmetric_mesh_distances,)]  ) 

    print(f'\nMeasuring (3D IoU, hausdorf dist. , max-mean dist., mean mean dist.):\n {metrics} \n ----------- \n average = {metrics.mean(axis =0)}')



    
    #                                gu.align_open3d_PC_ICP, (icp_sample_num,icp_iter,enable_rescale),
                                    


    print('==================  Voxelized CADs with the choosen reconstructed mesh   ================================ ')
    print('=================================================================================================== ')
    
    vox_file_list    = glob.glob('../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/*.obj') 
    #orig_files_list  = glob.glob('/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD/car/*.off')
    
    Vs_list = []
    Fs_list = []
    Vt_list = []
    Ft_list = []
    # obj_file = ['../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/car_01_clean_solid_voxrem_dec_smooth_symm_cub_tri.obj',\
    #             '../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/car_02_clean_solid_voxrem_dec_smooth_symm_cub_tri.obj' ]
    #for i in range(len(vox_file_list)):


    for i in range(10):
        Vs, Fs, vox_file = read_cad_mesh('../datasets/Pascal3D+_release1.1/CAD_vox_tri', 'car', i)
        #Vs, Fs = igl.read_triangle_mesh(mesh_file)
        Vt, Ft = igl.read_triangle_mesh('../datasets/reconstructed_car_sub.obj')
        
        if Vt.shape[0] < icp_sample_num:
            Vt, Ft =  igl.upsample(Vt,  Ft, 1)
        #Vt, Ft, orig_file  = read_cad_mesh('../datasets/Pascal3D+_release1.1/CAD', 'car', i)
        #Vt, Ft, orig_file  = read_cad_mesh('../datasets/Pascal3D+_release1.1/CAD_obj_prep1', 'car', i)

        print(f'Read {(vox_file)}')
        Vs_list.append(Vs)
        Fs_list.append(Fs)
        Vt_list.append(Vt)
        Ft_list.append(Ft)

    enable_rescale = True
    print(f'\ngu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_open3d_PC_ICP, (icp_sample_num,icp_iter,enable_rescale))
    enable_rescale = False
    print(f'\ngu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_open3d_PC_ICP, (icp_sample_num,icp_iter,enable_rescale))


    print(f'\ngu.align_source2target_with_icp({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_with_icp, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_source2target_pc_icp({(icp_sample_num,icp_iter,enable_rescale)}) [my PointCloud-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_pc_icp, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_source2target_PCA({(icp_sample_num,icp_iter,enable_rescale)}) [my PCA aigment for angles < 45 deg] ')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_PCA, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_PC_ICP_init_with_PCA({(icp_sample_num,icp_iter,enable_rescale)})  [my PC-ICP intialized with  my PCA] ')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_PC_ICP_init_with_PCA, (icp_sample_num,icp_iter,enable_rescale))



    print('==================  Voxelized CADs with preprocessed CADs         ================================= ')
    print('=================================================================================================== ')
    
    vox_file_list    = glob.glob('../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/*.obj') 
    #orig_files_list  = glob.glob('/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD/car/*.off')
    
    Vs_list = []
    Fs_list = []
    Vt_list = []
    Ft_list = []
    # obj_file = ['../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/car_01_clean_solid_voxrem_dec_smooth_symm_cub_tri.obj',\
    #             '../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/car_02_clean_solid_voxrem_dec_smooth_symm_cub_tri.obj' ]
    #for i in range(len(vox_file_list)):
    for i in range(10):
        Vs, Fs, vox_file = read_cad_mesh('../datasets/Pascal3D+_release1.1/CAD_vox_tri', 'car', i)
        #Vs, Fs = igl.read_triangle_mesh(mesh_file)
        #Vt, Ft = igl.read_triangle_mesh(mesh_file)
        #Vt, Ft, orig_file  = read_cad_mesh('../datasets/Pascal3D+_release1.1/CAD', 'car', i)
        Vt, Ft, orig_file  = read_cad_mesh('../datasets/Pascal3D+_release1.1/CAD_obj_prep1', 'car', i)

        print(f'Pair {(vox_file,orig_file)}')
        Vs_list.append(Vs)
        Fs_list.append(Fs)
        Vt_list.append(Vt)
        Ft_list.append(Ft)

    enable_rescale = True 
    print(f'\ngu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_open3d_PC_ICP, (icp_sample_num,icp_iter,enable_rescale))
    enable_rescale = False
    print(f'\ngu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_open3d_PC_ICP, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_source2target_with_icp({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_with_icp, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_source2target_pc_icp({(icp_sample_num,icp_iter,enable_rescale)}) [my PointCloud-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_pc_icp, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_source2target_PCA({(icp_sample_num,icp_iter,enable_rescale)}) [my PCA aigment for angles < 45 deg] ')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_PCA, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_PC_ICP_init_with_PCA({(icp_sample_num,icp_iter,enable_rescale)})  [my PC-ICP intialized with  my PCA] ')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_PC_ICP_init_with_PCA, (icp_sample_num,icp_iter,enable_rescale))

    
    print('==================  Voxelized CADs with orignal  CADs         ================================= ')
    print('=================================================================================================== ')
    vox_file_list    = glob.glob('../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/*.obj') 
    orig_files_list  = glob.glob('/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD/car/*.off')
    
    Vs_list = []
    Fs_list = []
    Vt_list = []
    Ft_list = []
    for i in range(10):
        Vs, Fs, vox_file = read_cad_mesh('../datasets/Pascal3D+_release1.1/CAD_vox_tri', 'car', i)
        Vt, Ft, orig_file  = read_cad_mesh('../datasets/Pascal3D+_release1.1/CAD', 'car', i)

        print(f'Pair {(vox_file,orig_file)}')
        Vs_list.append(Vs)
        Fs_list.append(Fs)
        Vt_list.append(Vt)
        Ft_list.append(Ft)
    
    enable_rescale = True 
    print(f'\ngu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_open3d_PC_ICP, (icp_sample_num,icp_iter,enable_rescale))
    enable_rescale = False
    print(f'\ngu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_open3d_PC_ICP, (icp_sample_num,icp_iter,enable_rescale))

    enable_rescale = False
    print(f'\ngu.align_source2target_with_icp({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_with_icp, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_source2target_pc_icp({(icp_sample_num,icp_iter,enable_rescale)}) [my PointCloud-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_pc_icp, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_source2target_PCA({(icp_sample_num,icp_iter,enable_rescale)}) [my PCA aigment for angles < 45 deg] ')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_PCA, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_PC_ICP_init_with_PCA({(icp_sample_num,icp_iter,enable_rescale)})  [my PC-ICP intialized with  my PCA] ')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_PC_ICP_init_with_PCA, (icp_sample_num,icp_iter,enable_rescale))


    print('==================  Voxelized CADs with their  Rotatations ================================== ')
    print('============================================================================================= ')
    alpha  = 10 #degrees
    Rmat  = (R.from_quat([0, 0, np.sin((alpha/180)*np.pi), np.cos((alpha/180)*np.pi) ])).as_matrix()

    #obj_file =[glob.glob('../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/*.obj')[0]] 
    obj_file = glob.glob('../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/*.obj') 
    
    Vs_list = []
    Fs_list = []
    Vt_list = []
    Ft_list = []
    # obj_file = ['../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/car_01_clean_solid_voxrem_dec_smooth_symm_cub_tri.obj',\
    #             '../datasets/Pascal3D+_release1.1/CAD_vox_tri/car/car_02_clean_solid_voxrem_dec_smooth_symm_cub_tri.obj' ]
    for mesh_file in obj_file:
        Vs, Fs = igl.read_triangle_mesh(mesh_file)
        Vt, Ft = igl.read_triangle_mesh(mesh_file)
        Vs_list.append(Vs)
        Fs_list.append(Fs)
        Vt = np.matmul(Vt,Rmat)
        Vt_list.append(Vt)
        Ft_list.append(Ft)

    enable_rescale = False
    print(f'\ngu.align_source2target_with_icp({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_with_icp, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_source2target_pc_icp({(icp_sample_num,icp_iter,enable_rescale)}) [my PointCloud-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_pc_icp, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_source2target_PCA({(icp_sample_num,icp_iter,enable_rescale)}) [my PCA aigment for angles < 45 deg] ')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_source2target_PCA, (icp_sample_num,icp_iter,enable_rescale))

    print(f'\ngu.align_PC_ICP_init_with_PCA({(icp_sample_num,icp_iter,enable_rescale)})  [my PC-ICP intialized with  my PCA] ')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_PC_ICP_init_with_PCA, (icp_sample_num,icp_iter,enable_rescale))

    enable_rescale = True 
    print(f'\ngu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_open3d_PC_ICP, (icp_sample_num,icp_iter,enable_rescale))
    enable_rescale = False
    print(f'\ngu.align_open3d_PC_ICP({(icp_sample_num,icp_iter,enable_rescale)}) [igl mesh-ICP]')
    check_multiple_ICP_results(Vs_list,Fs_list, Vt_list,Ft_list, gu.align_open3d_PC_ICP, (icp_sample_num,icp_iter,enable_rescale))