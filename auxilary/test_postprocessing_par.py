# %% 
import os
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import time

import cv2
import imageio
import lpips
import numpy as np
from tenacity import futures
import torch
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.structures.meshes import Meshes
from torch.utils.tensorboard import SummaryWriter

from datasets.cub.dataset import CUBDataset
from datasets.pascal3d.dataset import PascalDataset
from datasets.pascal3d.split_train_val_test_VOC import cad_num_per_class
from models.inceptionV3 import InceptionV3
from models.mcmr import MCMRNet
from models.renderer_softras import NeuralRenderer as SOFTRAS_renderer
from utils.geometry import y_rot
from utils.lab_color import rgb_to_lab, lab_to_rgb
from utils.losses import kp_l2_loss, deform_l2reg, camera_loss, quat_reg, GraphLaplacianLoss
from utils.metrics import get_IoU, get_L1, get_SSIM, get_FID, compute_mean_and_cov, get_feat
from utils.transformations import quaternion_matrix, euler_from_matrix
from utils.visualize_results import vis_results
import  utils.geometry as gu
from   utils.debug_utils  import out, dict2table
import  utils.debug_utils as du
import igl
import glob
import fnmatch
import re
from tqdm import tqdm
# ==> tested this script with "mcmr_ipy" coda env 
from tabulate import tabulate
import pickle
import itertools 
from functools import partial
import copy 
import concurrent.futures 
import pandas as pd 

def get_obj_files(data_path):
    class_and_type = os.path.split(data_path)[-1].split("-")[0].split("_")
    test_type_prefix = (class_and_type[1] if type(class_and_type) is list and len(class_and_type) > 1 else '')
    obj_file_glob = f'{data_path}/obj/*{test_type_prefix}_{"[0-9]"*4}.obj' 
    obj_pred_files = glob.glob(obj_file_glob)
    #Exclude weighted  mean shape obj files 
    wms_obj_files = glob.glob(f'{data_path}/obj/*_wms_{"[0-9]"*4}.obj')
    obj_files = list( set(obj_pred_files) - set(wms_obj_files) )

    #find indices for each class
    sample_num = len(obj_files)
    obj_files_str = " ".join(obj_files)
    index_list = [int(ind_str) for  ind_str in re.findall("\_([0-9]+)\.obj *", obj_files_str)]
    class_list = re.findall("obj\/([a-z]+)\_", obj_files_str)
    unique_classes = list(set(class_list))
    class_indices = dict(zip(unique_classes, [ [] for i in range(sample_num)]) )    
    [class_indices[class_list[i]].append(index_list[i]) for i in range(sample_num)]
    return  list( set(obj_pred_files) - set(wms_obj_files) ), index_list, class_list, class_indices

def get_obj_class_by_index(obj_pred_files, i):
    obj_files =fnmatch.filter(obj_pred_files, f'{data_path}/obj/*{test_type_prefix}_{i:04}.obj')
    if (len(obj_files) > 1) or (not obj_files):
        print(f'Error: For smaple {i} instead of a  single obj file, found the following files: {obj_files}')
    obj_file = obj_files[0]
    class_name = os.path.split(obj_file)[-1].split("_")[0]
    return obj_file, class_name 

def clean_CAD_format(CAD_path, class_list = ['aeroplane','bicycle', 'boat',  'bottle',  'bus',  'car',  'chair',  'diningtable',  'motorbike',  'sofa',  'train',  'tvmonitor']):
    for cad_class in class_list:
        cad_files = glob.glob( f'{CAD_path}/{cad_class}/{cad_class}_*.obj')
        cad_files.append(glob.glob( f'{CAD_path}/{cad_class}/{cad_class}_*.off'))
        print(f'Class {cad_class}')
        for cad_file in cad_files:
            if cad_file and  os.path.exists(cad_file):
                Vcad, Fcad = igl.read_triangle_mesh(cad_file)
                print(f'  Cleaning format of {cad_file}')
                igl.write_triangle_mesh(cad_file,Vcad, Fcad)




def read_cad_mesh(CAD_path, cad_class, cad_index, zero_based_index = True):
    cad_i = (cad_index + 1 if zero_based_index else cad_index)
    cad_file = f'{CAD_path}/{cad_class}/{cad_i:02d}.off'
    if not  os.path.exists(cad_file):
        cad_file = glob.glob( f'{CAD_path}/{cad_class}/{cad_class}_{cad_i:02d}*.obj')[0]
    Vcad, Fcad = igl.read_triangle_mesh(cad_file)
    return Vcad, Fcad, cad_file

def read_CAD_mesh(CAD_path,class_name,cad_i): #check wich function is better ?
    cad_file = f'{CAD_path}/{class_name}/{cad_i+1:02d}.off'
    if not  os.path.exists(cad_file):
        cad_file = glob.glob( f'{CAD_path}/{class_name}/{class_name}_{cad_i+1:02d}*.obj')[0]
        Vcad, Fcad = igl.read_triangle_mesh(cad_file)
    return  Vcad, Fcad, cad_file

def compare_reconstructed_mesh_with_CAD(V,F, Vcad, Fcad, symmetry_data, icp_sample_num = 200, icp_iter =20):
    normlaize_by_width = True
    V_right    = symmetry_data['V_right']
    V_opposite = symmetry_data['V_opposite']
    V_center   = np.nonzero(np.arange(0,V_opposite.shape[0]) == V_opposite)[0]
    V_left     = V_opposite[V_right]

    V -= np.mean(V,axis=0) #align shape to 0,0,0  
    V_bbox    = V.max(0)- V.min(0)
    #V_size = V_bbox.max() #longest dimension or width ?
    V_size = (V_bbox[0] if normlaize_by_width else V_bbox.max()) #longest dimension or width ?
    #mesh_size[i] = V_size #can be used to get obsolute distances 
    #V_bbox_dim[i,:] = V_bbox

    Vcad -=  np.mean(Vcad,axis=0) #align shape to 0,0,0  
    Vcad_bbox = Vcad.max(0)- Vcad.min(0)
    V_icp,_,_ = gu.align_source2target_with_icp(V,F, Vcad,Fcad, icp_sample_num,icp_iter)
    #mesh2CAD_dist = igl.hausdorff(V_icp,F, Vcad,Fcad)/V_size
    mesh2CAD_dist = gu.max_mean_mesh_distance(V_icp,F, Vcad,Fcad)/V_size


    V_l2r =  np.copy(V)
    V_l2r[V_right,:] =   gu.reflectX(V[V_opposite[V_right],:])
    V_l2r[V_center,0] = 0
    V_l2r_icp,_,_  = gu.align_source2target_with_icp(V_l2r,F, Vcad,Fcad, icp_sample_num,icp_iter)
    #LR_mesh2CAD_dist = igl.hausdorff(V_l2r_icp,F, Vcad,Fcad)/V_size
    LR_mesh2CAD_dist = gu.max_mean_mesh_distance(V_l2r_icp,F, Vcad,Fcad)/V_size

    V_r2l =  np.copy(V)
    V_r2l[V_left,:] =   gu.reflectX(V[V_opposite[V_left],:] )
    V_r2l[V_center,0] = 0
    V_r2l_icp,_,_  = gu.align_source2target_with_icp(V_r2l,F, Vcad,Fcad, icp_sample_num,icp_iter)
    #RL_mesh2CAD_dist = igl.hausdorff(V_r2l_icp,F, Vcad,Fcad)/V_size
    RL_mesh2CAD_dist = gu.max_mean_mesh_distance(V_r2l_icp,F, Vcad,Fcad)/V_size

    return mesh2CAD_dist, LR_mesh2CAD_dist, RL_mesh2CAD_dist 

def save_cad_metadata(CAD_path, class_list, opt):
    for class_name in class_list:
        if opt.verbose:
            print(f'Get {class_name} metadata from {CAD_path}\n     voxel_num = {opt.voxel_num} ')

        cad_file_list = glob.glob( f'{CAD_path}/{class_name}/{class_name}_*.obj')
        for cad_file in tqdm(cad_file_list):
            cad_npy = os.path.splitext(cad_file)[0] + f"-v{opt.voxel_num}.npy"
            if (not opt.rewrite) and os.path.exists(cad_npy):
                continue 
            # if os.path.exists(cad_npy):
            #     cad_metadata = np.load(cad_npy, allow_pickle = True).item()
            # else:
            #     cad_metadata = {'ugrid':dict(),'mask_t':dict()}
            cad_metadata = dict()
            
            Vt, Ft = igl.read_triangle_mesh(cad_file)
            ugrid, mask_t, grid_shape = gu.VoxelizeMesh(Vt,Ft, opt)
            cad_metadata['ugrid']  =  ugrid
            cad_metadata['mask_t'] =  mask_t
            cad_metadata['grid_shape'] = grid_shape
            #cad_metadata['ugrid'][opt.voxel_num]   =  ugrid
            #cad_metadata['mask_t'][opt.voxel_num]  =  mask_t
            np.save(cad_npy,cad_metadata,  allow_pickle =True)



        
def load_meshes_from_files(mesh_file_list):
    V_list=[]
    F_list=[]
    for mesh_file in mesh_file_list:
        V, F = igl.read_triangle_mesh(mesh_file)
        V_list.append(V)
        F_list.append(F)
    return V_list, F_list 



def get_mesh_cad_file_pairs(test_result_path, CAD_path):
    mesh_file_list = []
    cad_file_list  = []
    for data_path in test_result_path:
        obj_pred_files, index_list, class_list, class_indices  = get_obj_files(data_path)
        class_of_index = dict(zip(index_list, class_list))
        obj_file_of_index= dict(zip(index_list, obj_pred_files))
        num_of_samples = len(obj_pred_files)
        data = np.load( data_path + '/shape_symmetries.npy', allow_pickle = True).item()
        cad_idx = data['cad_idx']

        for i in range(num_of_samples):
            class_name = class_of_index[i] 
            obj_file   = obj_file_of_index[i]

            cad_i = int(cad_idx[i])
            cad_file = f'{CAD_path}/{class_name}/{cad_i+1:02d}.off'
            if not  os.path.exists(cad_file):
                #cad_file = ad_file = f'{CAD_path}/{class_name}/{cad_i+1:02d}.obj'
                cad_file = glob.glob( f'{CAD_path}/{class_name}/{class_name}_{cad_i+1:02d}*.obj')[0]
            
            mesh_file_list.append(obj_file)
            cad_file_list.append(cad_file)

    return  mesh_file_list, cad_file_list

def compute_additional_test_metrics(test_result_path, CAD_path, suffix = ""):
    compute_symmetries = False
    compare_to_CAD     = True
    enable_rescale_for_ICP = True

    opt = {'verbose':False, 'align':False}

    icp_sample_num = 200
    icp_iter = 20
    normlaize_by_width = True
    for data_path in test_result_path:
        #obj_file_glob = f'{data_path}/obj/{class_name}_+([0-9]).obj'
        #obj_pred_files = os.popen(f'find {obj_file_glob}').read().split('\n')

        #class_name = os.path.split(data_path)[-1].split("-")[0] #wrong for multi-class tests 
        #short_class_name = class_name.split('_')[0]
        #obj_file_glob = f'{data_path}/obj/{class_name}_????.obj'
        #test_type = os.path.split(data_path)[-1].split("-")[0].split("_")[1]

        # class_and_type = os.path.split(data_path)[-1].split("-")[0].split("_")
        # test_type_prefix = (class_and_type[1] if type(class_and_type) is list and len(class_and_type) > 1 else '')
        # obj_file_glob = f'{data_path}/obj/*{test_type_prefix}_{"[0-9]"*4}.obj' 
        # obj_pred_files = glob.glob(obj_file_glob)
        # #Exclude weighted  mean shape obj files 
        # wms_obj_files = glob.glob(f'{data_path}/obj/*_wms_{"[0-9]"*4}.obj')
        # obj_pred_files  = list( set(obj_pred_files) - set(wms_obj_files) )
        obj_pred_files, index_list, class_list, class_indices  = get_obj_files(data_path)
        class_of_index = dict(zip(index_list, class_list))
        obj_file_of_index= dict(zip(index_list, obj_pred_files))

        num_of_samples = len(obj_pred_files)

        LR_max  = np.zeros(num_of_samples)
        LR_mean = np.zeros(num_of_samples)
        H_symm  = np.zeros(num_of_samples)
        symmetry_str = ""
        num_of_metrics = 3
        mesh_to_cad_dist    = np.zeros((num_of_samples, num_of_metrics))
        LR_mesh_to_cad_dist = np.zeros((num_of_samples, num_of_metrics))
        RL_mesh_to_cad_dist = np.zeros((num_of_samples, num_of_metrics))

        compare_to_CAD_str  = ""
        mesh_size           =  np.zeros(num_of_samples)

        data = np.load( data_path + '/shape_symmetries.npy', allow_pickle = True).item()
        LR_max = data['LR_max']
        V_right = data['V_right']
        V_opposite = data['V_opposite']
        V_left     = V_opposite[V_right]
        V_center   = np.nonzero(np.arange(0,V_opposite.shape[0]) == V_opposite)[0]
        cad_idx = data['cad_idx']
        V_bbox_dim = np.zeros((num_of_samples,3))
    
        #alternatively run 
        # "sphere_data = np.load('../auxilary/icoshpere_mesh_subdivision_1.npy', allow_pickle = True).item()"
        # amd use the code from "def visualize_sphere(p, V, selected_indices, selected_col)" to V_right, V_opposite
        if compare_to_CAD:
            compare_to_CAD_str = f'===== Relative Hausdorff distances from CAD to meshes and their reflections (path ={data_path} )\n ' + \
                            f'  icp (sample_num, iter)={(icp_sample_num, icp_iter)},' + \
                            f'distances are normalized by bbox {"width" if normlaize_by_width else "longest axis"}\n'
            print(compare_to_CAD_str)

        for i in range(num_of_samples):
            # obj_files =fnmatch.filter(obj_pred_files, f'{data_path}/obj/*{test_type_prefix}_{i:04}.obj')
            # if (len(obj_files) > 1) or (not obj_files):
            #     print(f'Error: For smaple {i} instead of a  single obj file, found the following files: {obj_files}')
            # obj_file = obj_files[0]
            # class_name = os.path.split(obj_file)[-1].split("_")[0]
            class_name = class_of_index[i] 
            obj_file   = obj_file_of_index[i]

            cad_i = int(cad_idx[i])
            cad_file = f'{CAD_path}/{class_name}/{cad_i+1:02d}.off'
            if not  os.path.exists(cad_file):
                #cad_file = ad_file = f'{CAD_path}/{class_name}/{cad_i+1:02d}.obj'
                cad_file = glob.glob( f'{CAD_path}/{class_name}/{class_name}_{cad_i+1:02d}*.obj')[0]
            V, F = igl.read_triangle_mesh(obj_file)
            #LR_max_i, LR_mean_i, LR_mean_i = gu.measure_symmetries(V,F,V_right, V_opposite, opt = {'verbose':True})
            #Vcad, Fcad, _ = read_CAD_mesh(CAD_path,class_name,cad_i)
            #mesh2CAD_dist, LR_mesh2CAD_dist, RL_mesh2CAD_dist =  compare_reconstructed_mesh_with_CAD(V,F, Vcad, Fcad, data)

            if compute_symmetries:
                LR_max[i], LR_mean[i], H_symm[i] = gu.measure_symmetries(V,F,V_right, V_opposite, opt)
                symmetry_str += f'Sample={i:04}: Mean-max Symmetry errors = ({LR_mean[i]:.4f}, {LR_max[i]:.4f})\n'
            if compare_to_CAD:
                V -= np.mean(V,axis=0) #align shape to 0,0,0  
                V_bbox    = V.max(0)- V.min(0)
                #V_size = V_bbox.max() #longest dimension or width ?
                V_size = (V_bbox[0] if normlaize_by_width else V_bbox.max()) #longest dimension or width ?
                mesh_size[i] = V_size #can be used to get obsolute distances 
                V_bbox_dim[i,:] = V_bbox

                Vcad, Fcad = igl.read_triangle_mesh(cad_file)
                Vcad -=  np.mean(Vcad,axis=0) #align shape to 0,0,0  
                Vcad_bbox = Vcad.max(0)- Vcad.min(0)
                V_icp,_,_ = gu.align_source2target_with_icp(V,F, Vcad,Fcad, icp_sample_num,icp_iter)
                #mesh_to_cad_dist[i] = igl.hausdorff(V_icp,F, Vcad,Fcad)/V_size
                mesh_to_cad_dist[i,:] = gu.max_mean_mesh_distance(V_icp,F, Vcad,Fcad)/V_size


                V_l2r =  np.copy(V)
                V_l2r[V_right,:] =   gu.reflectX(V[V_opposite[V_right],:])
                V_l2r[V_center,0] = 0
                V_l2r_icp,_,_  = gu.align_source2target_with_icp(V_l2r,F, Vcad,Fcad, icp_sample_num,icp_iter)
                #LR_mesh_to_cad_dist[i] = igl.hausdorff(V_l2r_icp,F, Vcad,Fcad)/V_size
                LR_mesh_to_cad_dist[i,:] = gu.max_mean_mesh_distance(V_l2r_icp,F, Vcad,Fcad)/V_size

                V_r2l =  np.copy(V)
                V_r2l[V_left,:] =   gu.reflectX(V[V_opposite[V_left],:] )
                V_r2l[V_center,0] = 0
                V_r2l_icp,_,_  = gu.align_source2target_with_icp(V_r2l,F, Vcad,Fcad, icp_sample_num,icp_iter)
                #RL_mesh_to_cad_dist[i] = igl.hausdorff(V_r2l_icp,F, Vcad,Fcad)/V_size
                RL_mesh_to_cad_dist[i,:] = gu.max_mean_mesh_distance(V_r2l_icp,F, Vcad,Fcad)/V_size
                
                #f' ({mesh_to_cad_dist[i,:]:.4f}, {LR_mesh_to_cad_dist[i,:]:.4f}, {RL_mesh_to_cad_dist[i,:]:.4f})' + \
                compare2CAD_strl = f'Sample={i:04}: distances to mesh, its left-to-rigth and right-to-left reflections= ' + \
                                '({:.4f}, {:.4f})'.format(*mesh_to_cad_dist[i,:]) + ',  ({:.4f}, {:.4f})'.format(*LR_mesh_to_cad_dist[i,:])  + ',  ({:.4f}, {:.4f})'.format(*RL_mesh_to_cad_dist[i,:]) +\
                                f'\n     Improvments: Hausdorff = {mesh_to_cad_dist[i,0] - min([LR_mesh_to_cad_dist[i,0], RL_mesh_to_cad_dist[i,0]]):.4f}' +\
                                f', Symmetric dist. = {mesh_to_cad_dist[i,1] - min([LR_mesh_to_cad_dist[i,1], RL_mesh_to_cad_dist[i,1]]):.4f}'      
                compare_to_CAD_str +=   compare2CAD_strl + '\n'     
                print(compare2CAD_strl) 
        
        if compute_symmetries:
            symmetry_sum_str =  f'========================== test = {data_path}  (aligmenet={opt["align"]}) ============ \n'
            symmetry_sum_str +=  f'Average Mean-Max symmetry errors per sample =  ({LR_mean.sum()/num_of_samples:.4f}, {LR_max.sum()/num_of_samples:.4f})\n'
            symmetry_sum_str += f'Max Max-Symmetry Errors  =  {LR_max.max():.4f}  for sample {LR_max.argmax():04}\n'
            symmetry_sum_str += f'Average Hausdorff Symmetry Error per sample  =  {H_symm.sum()/num_of_samples:.4f}\n'
            symmetry_sum_str += f'Max Hausdorff Symmetry Errors  =  {H_symm.max():.4f}  for sample {H_symm.argmax():04}\n'
            #print(symmetry_str + symmetry_sum_str)
            print(symmetry_sum_str)
            save_dir = './save'
            np.save((data_path  + '/shape_symmetries.npy'), \
                    {'LR_mean':LR_mean, 'LR_max':LR_max, 'H_symm':H_symm, \
                    'V_right':V_right,'V_opposite':V_opposite},\
                    allow_pickle =True )
            with open(data_path +'/shape_symmetries.txt', 'w') as fd:
                fd.write(symmetry_str)
                fd.close()
                
        if compare_to_CAD:
            best_reflection_dist = np.stack( (LR_mesh_to_cad_dist, RL_mesh_to_cad_dist), axis=2).min(2)
            improved_reflection_dist = mesh_to_cad_dist - best_reflection_dist

            compare2CAD_strl  = f'========================== test = {data_path}  ============ \n'
            compare2CAD_strl +=  f'Average  distances from cad to mesh and their best reflections: \n' 
            compare2CAD_strl +=  f'  Hausdorff  = ({mesh_to_cad_dist[:,0].mean():.4f}, {best_reflection_dist[:,0].mean():.4f}),  improvement = {improved_reflection_dist[:,0].mean()/mesh_to_cad_dist[:,0].mean():.4f}\n'
            compare2CAD_strl +=  f'  Symmetric  = ({mesh_to_cad_dist[:,1].mean():.4f}, {best_reflection_dist[:,1].mean():.4f}),  improvement = {improved_reflection_dist[:,1].mean()/mesh_to_cad_dist[:,1].mean():.4f}\n'
            compare2CAD_strl +=  f'  Symmetric+ = ({mesh_to_cad_dist[:,2].mean():.4f}, {best_reflection_dist[:,2].mean():.4f}),  improvement = {improved_reflection_dist[:,2].mean()/mesh_to_cad_dist[:,2].mean():.4f}\n'
            #compare2CAD_strl +=  f'Average distance improvement = {improved_reflection_dist.mean():.4f}'
            print(compare2CAD_strl)     
            compare_to_CAD_str += compare2CAD_strl
            #symmetry_sum_str +=  f'({mesh_to_cad_dist.sum()/num_of_samples:.4f}, {LR_mesh_to_cad_dist.sum()/num_of_samples:.4f}, {RL_mesh_to_cad_dist.sum()/num_of_samples:.4f})'
            
            #print(compare_to_CAD_str)
            with open(f'{data_path}/cad_mesh_distances{suffix}.txt', 'w') as fd:
                fd.write(compare_to_CAD_str)
                fd.close()

            np.save(f'{data_path}/cad_mesh_distances{suffix}.npy', \
                    {'mesh_to_cad_dist':mesh_to_cad_dist, 'LR_mesh_to_cad_dist':LR_mesh_to_cad_dist, 'RL_mesh_to_cad_dist':RL_mesh_to_cad_dist, \
                    'best_reflection_dist':best_reflection_dist,'improved_reflection_dist':improved_reflection_dist, 'mesh_size': mesh_size, \
                    'V_bbox_dim':V_bbox_dim, 'icp_sample_num':icp_sample_num,'icp_iter':icp_iter} ,\
                    allow_pickle =True )
        


def SourceTargetMeshPipeline(s_file_list,t_file_list, transform_fun_args_list, metric_fun_args_list, verbose = False):
    metric_list = []
    Vs_list =[]
    for i in tqdm( range(len(s_file_list)) ):
        Vs, Fs = igl.read_triangle_mesh(s_file_list[i])
        Vt, Ft = igl.read_triangle_mesh(t_file_list[i])
        meta_data= dict(s_file = s_file_list[i], t_file=t_file_list[i])
        
        
        for cmd in transform_fun_args_list: #pipeline of geoemtric transformation 
            res = cmd[0](Vs,Fs, Vt,Ft,  *cmd[1:])
            Vs =  res[0] if type(res)==tuple else res
        Vs_list.append(Vs)

        current_metric = []
        for cmd in metric_fun_args_list: #pipeline of metric meausrement functions 
            current_metric.append(cmd[0](Vs,Fs, Vt, Ft,meta_data, *cmd[1:]) )

        metric_list.append( du.flatten_list(current_metric))
    
    metrics = np.array(metric_list)
    if verbose:
        print(f'Mean metric per pair = {metrics.mean(axis =0)}')
    return metrics, Vs_list


# a variant of "SourceTargetMeshPipeline" with more  options 
def compute_3d_metrics(source_data, target_data, transform_opt, pipeline_opt):
    if type(source_data) == str and  type(target_data) == str: #source tagets are paths to data files
        mesh_files, cad_files = get_mesh_cad_file_pairs([source_data], CAD_path)
        data = np.load( source_data + '/shape_symmetries.npy', allow_pickle = True).item() #symmetry data is the same for all classes 
    else: #source,targets are list of mesh files and the symmetry data is inside 'transform_opt'
        mesh_files  = source_data
        cad_files   = target_data
        data = transform_opt.symmetry_data 
    
    if pipeline_opt.enable_icp:
        ICP_cmd = [(gu.align_open3d_PC_ICP, transform_opt)]
    else:
        ICP_cmd = []
    
    metrics, Vs_list = SourceTargetMeshPipeline(mesh_files, cad_files, \
                            [ (gu.symmetrize_source_mesh_network, data , pipeline_opt.symmetrization_type)] + ICP_cmd , \
                            [ (pipeline_opt.IoU_3D_func,transform_opt), \
                             (gu.relative_symmetric_mesh_distances,)],\
                            verbose=pipeline_opt.verbose )
    return metrics, Vs_list


def test_symmetrization_3d_results():
    resume_pkl = None
    ####### comparing CADs with reconstructed meshes with mehs distances 3D IoU, ICP and symmetrization
    # icp_iter       = 10 #common ICP params
    # icp_sample_num = 2000
    # voxel_num = 200
    # enable_rescale = False

    # experiment_name = 'small'
    # result_path_list = ['save/Pascal3D_12_classes_12_PointRend_small/aeroplane_small-bicycle_small-boat_small-bottle_small-bus_small-car_small-chair_small-diningtable_small-motorbike_small-sofa_small-train_small-tvmonitor_small-qualitative', \
    #                     'save/Pascal3D_12_classes_1_PointRend_small/aeroplane_small-bicycle_small-boat_small-bottle_small-bus_small-car_small-chair_small-diningtable_small-motorbike_small-sofa_small-train_small-tvmonitor_small-qualitative', \
    #                     'save/plane_8_MRCNN_small/aeroplane_small-qualitative', \
    #                     'save/not_symmetric/car_10_MRCNN_small/car_small-qualitative', \
    #                     'save/plane_car_1_PointRend_small/aeroplane_small-car_small-qualitative',\
    #                     'save/plane_car_2_PointRend_small/aeroplane_small-car_small-qualitative', \
    #                     'save/bicycle_bus_car_bike_1_PointRend_small/bicycle_small-bus_small-car_small-motorbike_small-qualitative',\
    #                     'save/bicycle_bus_car_bike_4_PointRend_small/bicycle_small-bus_small-car_small-motorbike_small-qualitative' \
    #                     ]    
    

    experiment_name = 'full'
    result_path_list = [ \
                        'save/Pascal3D_12_classes_12_PointRend__/aeroplane-bicycle-boat-bottle-bus-car-chair-diningtable-motorbike-sofa-train-tvmonitor-qualitative', \
                        'save/Pascal3D_12_classes_1_PointRend__/aeroplane-bicycle-boat-bottle-bus-car-chair-diningtable-motorbike-sofa-train-tvmonitor-qualitative', \
                        'save/plane_8_MRCNN__/aeroplane-qualitative/', \
                        'save/not_symmetric/car_10_MRCNN__/car-qualitative', \
                        'save/plane_car_1_PointRend__/aeroplane-car-qualitative',\
                        'save/plane_car_2_PointRend__/aeroplane-car-qualitative', \
                        'save/bicycle_bus_car_bike_1_PointRend__/bicycle-bus-car-motorbike-qualitative',\
                        'save/bicycle_bus_car_bike_4_PointRend__/bicycle-bus-car-motorbike-qualitative' \
                        ] 

    CAD_path   = "/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD_preprocessed_tri"
    #clean_CAD_format(CAD_path, ['aeroplane']) 
    #result_path_list = result_path_list[0:2]
    str_list = []
    
    shortenPath =  lambda path: '/'.join(path.split('/')[-3:-1]).replace('/', '.') #simple version of path to test name  
    metric_names = [ '3d-iou', 'Hausdorf', 'max-mean', 'mean-mean']
    # reportOrig = du.MetricReport(metric_names, shortenPath)
    # reportSymm = du.MetricReport(metric_names, shortenPath)
    if resume_pkl:
        print(f'-I- Resume the experiment from {resume_pkl}')
        with open(resume_pkl, 'rb') as handle:
            experiment_data = pickle.load(handle)
            reportOrig = experiment_data['reportOrig']
            reportSymm = experiment_data['reportSymm']
            reportSymmType = experiment_data['reportSymmType']
    else:
        reportOrig = du.MetricReport(metric_names)
        reportSymm = du.MetricReport(metric_names)
        reportSymmType = du.MetricReport(metric_names)


    opt = gu.transformation_options(iter_num=10, sample_num = 2000, voxel_num =100,  enable_rescale = False, verbose =True)
    opt.rewrite = False
    #save_cad_metadata(CAD_path, ['car','aeroplane','bus','bicycle','motorbike'], opt)
    save_cad_metadata(CAD_path, [ 'car','aeroplane','bus','bicycle','motorbike', \
                                'boat', 'bottle', 'chair', 'diningtable',   \
                                'sofa', 'train', 'tvmonitor'                ],   opt) #all classes 
    opt.verbose = False
    symmetrization_dirs =['l2r','r2l','cent']    
    # symmetrization_dirs =['cent'] 
    for result_path in result_path_list: 
        out(f'\n ==========  Measuring 3D IoU, hausdorf dist. , max-mean dist., mean mean dist for: result_path={result_path}, CAD_path ={CAD_path}', str_list)
        out(f'    result_path = {result_path}\n    CAD_path = {CAD_path}\n    voxel_num = {opt.voxel_num}', str_list)
        mesh_files, cad_files = get_mesh_cad_file_pairs([result_path], CAD_path)
        test_name = shortenPath(result_path)

        out(f"   ------- Without post-processing symmetrization ({test_name})-----", str_list)
        # fast 3d IoU function that load cad voxels/mask from precomputed npy file 
        metrics,_  = SourceTargetMeshPipeline(mesh_files, cad_files, \
                            [(gu.align_open3d_PC_ICP, opt)], \
                            [(gu.Compute3DIoU_fast,opt), (gu.relative_symmetric_mesh_distances,)], verbose=True ) 

        reportOrig.add_test(test_name, metrics)
        #reportOrig.add_test(result_path, metrics)



        metrics_mean = metrics.mean(axis=0).tolist()
        out(f"   -------- With best symmetrization in post-processing  ({test_name}) -----", str_list)
        data = np.load( result_path + '/shape_symmetries.npy', allow_pickle = True).item() #symmetry data is the same for all classes 

        metric_symm = []

        # metric_l2r  
        if 'l2r'  in symmetrization_dirs:
            metric_symm.append( SourceTargetMeshPipeline(mesh_files, cad_files, \
                                        [ (gu.symmetrize_source_mesh, data , True),\
                                        (gu.align_open3d_PC_ICP, opt),\
                                        ], \
                                        [(gu.Compute3DIoU_fast,opt), (gu.relative_symmetric_mesh_distances,)],  verbose=True  ) [0])
        
        # metric_r2l  
        if 'r2l'  in symmetrization_dirs:
            metric_symm.append( SourceTargetMeshPipeline(mesh_files, cad_files, \
                                        [ (gu.symmetrize_source_mesh, data , False),\
                                        (gu.align_open3d_PC_ICP, opt),\
                                        ], \
                                        [(gu.Compute3DIoU_fast,opt), (gu.relative_symmetric_mesh_distances,)],  verbose=True   ) [0])
        # metric_cent
        if 'cent'  in symmetrization_dirs:
            metric_symm.append( SourceTargetMeshPipeline(mesh_files, cad_files, \
                                [ (gu.symmetrize_source_mesh, data , 3),\
                                (gu.align_open3d_PC_ICP, opt),\
                                ], \
                                [(gu.Compute3DIoU_fast,opt), (gu.relative_symmetric_mesh_distances,)],  verbose=True   ) [0])




        #only middle direction symmetrizarion 

        #metric_symm = np.stack((metric_l2r,metric_r2l),axis=2) #left-rigth and right-left directions
        #metric_symm = np.stack((metric_l2r,metric_r2l,metric_cent),axis=2) #left-rigth and right-left and central - 3 directios 
        metric_symm = np.stack(metric_symm,axis=2) #left-rigth and right-left and central - 3 directios 

        best_symm_type =np.column_stack( ( metric_symm[:,0,:].argmax(axis=-1),  metric_symm[:,1:,:].argmin(axis=-1) )) 
        reportSymmType.add_test(test_name,best_symm_type)
        metric_best_symm =np.column_stack( ( metric_symm[:,0,:].max(axis=-1),  metric_symm[:,1:,:].min(axis=-1) )) 
        #reportSymm.add_test(result_path, metric_best_symm)
        reportSymm.add_test(test_name, metric_best_symm)

        out(f"  best    symmetrization =  {metric_best_symm.mean(axis=0)}", str_list)
        improvment =  (metrics_mean - metric_best_symm.mean(axis=0)) / metrics_mean
        out(f"      improvements = {100*improvment} %", str_list)

        temp_report_file = f'draft/results-{experiment_name}-after-{test_name}-{opt.voxel_num}_voxel_{"with" if opt.enable_rescale else "without"}_rescaling_{"-".join(symmetrization_dirs)}'
        with open(temp_report_file + '.pkl', 'wb') as fd:
            print( f'-I- save temporal results in {temp_report_file}.pkl (after test: {test_name}, experiment: {experiment_name})')
            pickle.dump( {"reportOrig":reportOrig, "reportSymm":reportSymm}, fd)
        

        out(f"\n==============  Network results for {result_path}:", str_list)
        out(f" without symmetrization =  {metrics_mean}", str_list)


    report_orig_str   = reportOrig.get_table("\n=================== Without symmetrization =====================")[0]
    report_symm_str   = reportSymm.get_table("\n =================== With    symmetrization =====================")[0]
    report_symm_type  = reportSymmType.get_table("\n ----  Best   symmetrization type --- ")[0]

    report_improv_str= reportOrig.compare(reportSymm, "\n =================== Symmetrization Improvements (%) =====================")[0]
    str_list.insert(0, report_orig_str)
    str_list.insert(0, report_symm_str)
    str_list.insert(0, report_symm_type)
    str_list.insert(0, report_improv_str)
    


    #print(str_results)
    report_file_name = f'draft/estimatation_of_3D_metrics-{experiment_name}-{opt.voxel_num}_voxel_{"with" if opt.enable_rescale else "without"}_rescaling_{"-".join(symmetrization_dirs)}'
    with open(report_file_name + '.txt', 'w') as fd:
        fd.write("\n".join(str_list) )
    
    with open(report_file_name + '.pkl', 'wb') as fd:
        #pickle.dump( {"reportOrig":reportOrig.result_dict, "reportSymm":reportSymm.result_dict}, fd)
        pickle.dump( {"reportOrig":reportOrig, "reportSymm":reportSymm, "reportSymmType":reportSymmType}, fd)
    print( f'-I- save final results in {report_file_name}.pkl, *.txt ( experiment: {experiment_name})')
        

    return 0
    #the code below is for CAD format cleanup. It should be ran once for new CADs

    # #=============================full test datas ======================================
    # data_path_list = ['save/car_10_MRCNN__/car-qualitative',  \
    #                 'save/plane_car_1_PointRend__/aeroplane-car-qualitative', \
    #                 'save/plane_car_2_PointRend__/aeroplane-car-qualitative', \
    #                 'save/bicycle_bus_car_bike_1_PointRend__/bicycle-bus-car-motorbike-qualitative', \
    #                 'save/bicycle_bus_car_bike_4_PointRend__/bicycle-bus-car-motorbike-qualitative' \
    #                 ]

    # #========================== small scale tests 10% ==================================
    # data_path_list = ['save/car_10_MRCNN_small/car_small-qualitative',\
    #                   'save/plane_car_1_PointRend_small/aeroplane_small-car_small-qualitative',\
    #                   'save/plane_car_2_PointRend_small/aeroplane_small-car_small-qualitative', \
    #                   'save/bicycle_bus_car_bike_1_PointRend_small/bicycle_small-bus_small-car_small-motorbike_small-qualitative',\
    #                   'save/bicycle_bus_car_bike_4_PointRend_small/bicycle_small-bus_small-car_small-motorbike_small-qualitative' \
    #                ] 
    random.seed(1234)
    clean_CAD_format("/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD_obj_prep1", ['car']) #after Blender and manual clean-ups
    clean_CAD_format("/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD_vox_tri", ['car'])   #voxelized CADs

    #CAD_path = "/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD"
    #compute_additional_test_metrics(data_path_list,CAD_path, 'orig_cad')
    
    #compute_additional_test_metrics(['save/not_symmetric/car_10_MRCNN_small/car_small-qualitative'],"/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD_obj_prep1", '_prep')
    #compute_additional_test_metrics(['save/not_symmetric/car_10_MRCNN_small/car_small-qualitative'],"/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD", '_orig')

    # compute_additional_test_metrics(['save/symmetric/car_10_MRCNN_small/car_small-qualitative'],"/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD_obj_prep1", '_prep')
    # compute_additional_test_metrics(['save/not_symmetric/car_10_MRCNN_small/car_small-qualitative'],"/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD", '_orig')
    


    #compute_additional_test_metrics(['save/not_symmetric/car_10_MRCNN__/car-qualitative'],"/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD", '_orig')
    #compute_additional_test_metrics(['save/not_symmetric/car_10_MRCNN__/car-qualitative'],"/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD_obj_prep1", '_prep1')

    compute_additional_test_metrics(['save/symmetric/car_10_MRCNN__/car-qualitative'],"/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD", '_orig')
    compute_additional_test_metrics(['save/symmetric/car_10_MRCNN__/car-qualitative'],"/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD_obj_prep1", '_prep1')
    


def add_new_metric_combinations(losses):
    losses['1 - SSIM']      =  1 -  losses['SSIM Metric']
    losses['1 - IoU_2D']    =  1 -  losses['IoU Metric']
    losses['Texture Comb']  =  losses['1 - SSIM']  +  0.001* losses['FID Metric'] + 10*losses['L1 Metric']
    losses['SSL Comb']      =  losses['Texture Comb'] + 0.001 * losses['Color Loss'] + losses['1 - IoU_2D'] + losses['Perceptual Loss'] + losses['Mask Loss']
    return losses

# Comparing orignal network to different types of symmetrized networks (advanced version of "test_symmetrization_3d_results")
# It measures improvements in: different network losses, 3D metrics, new combinations of 3D metrics and network losses

def all_equal(iterable):
    g = itertools.groupby(iterable)
    return next(g, True) and not next(g, False)

#def SymmetrizationTypeNames(symm_data):
def SymmetrizationTypeNames(number_of_slices):
    #number_of_slices =   len(symm_data['V_right_slice']) if 'V_right_slice' in symm_data else 1
    symm_types      = list(itertools.product([1,2,3], repeat = number_of_slices) )
    symm_types      = [0]     + [s[0] if all_equal(s) else s for s in symm_types] #add non-symmetric and shorten  const. symmetrization keys (e.g (1,1) -> 1)
    symm_name_list  = list(itertools.product(['l2r','r2l','cent'],  repeat =  number_of_slices) )
    #symm_names      = ['/'.join(t)  for  t in  symm_name_list] #('l2r', 'l2r') -> 'l2r/l2r'
    symm_names      = ['orig'] +[ t[0] if all_equal(t) else '/'.join(t)  for  t in  symm_name_list]
    symm_names_dict = dict(zip(symm_types,symm_names))
    symm_index_dict = dict( zip(symm_types, range(len(symm_types)) )  ) 
    
    #adding disable symmetrization 
    

    return symm_names, symm_names_dict, symm_index_dict



def process_test_results_pkl_full(symmIndDict, common_pipeline_opt, transf_opt, CAD_path,    pkl_file):
    ############# par startmetric_names
    # Repeated inputs: symmIndDict, transf_opt, CAD_path,  Optional: pipeline_opt-> pipeline_opt_common
    # varying inputs pkl_file, Optional: pass pipeline_opt to deep copy it by executor.map
    # changes: pipeline_opt =  pipeline_opt_common.copy()

    print(f'-I- reading  {pkl_file}')
    # # BackUP pickle files
    # os.system(f'cp {pkl_file} {pkl_file}.backup')  
    # os.system(f'cp {pkl_file} {os.path.splitext(pkl_file)[0]}_3d.pkl')  
    with open(pkl_file, 'rb') as handle:
        results = pickle.load(handle)
    
    symm_type = results['args'].symmetrize  #type per slice s.t. 1->l2r or (1,2)->l2r/r2l
    if type(symm_type) == list:
        symm_type = tuple(symm_type) 
    if not symm_type in symmNameDict:
        print(f'-W- Symmetrization of type {symm_type} is not allowed => skipped' )
        return None, None 
        #continue 
    symm_indx  = symmIndDict[symm_type]    #index inside list of all symmetrization types/names     
    #print(f'(symm_indx, symm_type) = {(symm_indx, symm_type)}')
    #sould be safe for multiprocessing but not for multi-threading !
    pipeline_opt = copy.deepcopy(common_pipeline_opt)

    missing_metrics = ( set(metric_names)-set(results['losses'].keys()) )
    #if  missing_metrics and missing_metrics  <= set(metric_3D_names): #check  if 3d metrics are missing in the pkl 
    if  recompute_3d_metrics or missing_metrics.intersection( set(metric_3D_names) ): #check  if 3d metrics are missing in the pkl 
        print(f'-W- Metrics {missing_metrics} are missing in {pkl_file},\n -I- Recomputing the following metrics: {metric_3D_names}')
        pipeline_opt.symmetrization_type = symm_type
        pipeline_opt.enable_icp = True
        metrics_3D, _ = compute_3d_metrics(os.path.split(pkl_file)[0], CAD_path, transf_opt,pipeline_opt)
        for i in range( len(metric_3D_names) ): #adding these metrics to pickle results
            results['losses'][ metric_3D_names[i] ]  = metrics_3D[:,i]
        print(f'-W-  Adding to  {pkl_file} obtained 3D metrics')
        with open(pkl_file, 'wb') as fd:
            pickle.dump( results, fd)

    #if  missing_metrics and missing_metrics  <= set(metric_3D_names_id): #check  if 3d metrics are missing in the pkl 
    if  recompute_3d_metrics or missing_metrics.intersection( set(metric_3D_names_id)): #check  if 3d metrics are missing in the pkl 
        print(f'-W- Metrics {missing_metrics} are missing in {pkl_file},\n -I- Recomputing the following metrics: {metric_3D_names_id}')
        pipeline_opt.symmetrization_type = symm_type
        pipeline_opt.enable_icp = False
        metrics_3D, _ = compute_3d_metrics(os.path.split(pkl_file)[0], CAD_path, transf_opt,pipeline_opt)
        for i in range( len(metric_3D_names_id) ): #adding these metrics to pickle results
            results['losses'][ metric_3D_names_id[i] ]  = metrics_3D[:,i]
        print(f'-W-  Adding to  {pkl_file} obtained 3D metrics (without ICP)')
        with open(pkl_file, 'wb') as fd:
            pickle.dump( results, fd)
            
    add_new_metric_combinations(results['losses'])

    Shrink =  lambda arr:  arr if len(arr.shape)==1 else  arr[:,0]
    loss_list = [ Shrink(results['losses'][name]) for name in metric_names ] 
    
    metrics = np.stack(loss_list, axis =1)
    ########## end of par function 
    # returns (metrics,symm_indx) -> metrics_list[symm_indx]
    return metrics,symm_indx 

# %%
if __name__ == "__main__":
    #set the right working directorty for notebooks
    if os.path.split(os.getcwd())[1] == 'auxilary':
        os.chdir( os.getcwd() + '/..')
    #  -------  Configurations for short report  ---------------
    # this are tests with 3 slices
    result_path_list = [ \
                    'save/plane_8_MRCNN_small', \
                ]
    # this are tests with 2 slices (top-bottom)                
    # result_path_list = [ \
    #                 #  'save/plane_8_MRCNN_small', \
    #                 # 'save/plane_8_MRCNN_tiny', \
    #                 'save/plane_8_MRCNN__', \
    #                 'save/plane_car_2_PointRend__', \
    #                 'save/bicycle_bus_car_bike_1_PointRend__',\
    #                 'save/bicycle_bus_car_bike_4_PointRend__',\
    #                 #   'save/bicycle_bus_car_bike_1_PointRend_small',\
    #                 #   'save/bicycle_bus_car_bike_4_PointRend_small',\
    #                 #   'save/Pascal3D_12_classes_1_PointRend_small', \
    #                 'save/Pascal3D_12_classes_12_PointRend__', \
    #                 'save/Pascal3D_12_classes_1_PointRend__', \
    #                 #   'save/Pascal3D_12_classes_12_PointRend_small', \
    #                 'save/car_10_MRCNN__',\
    #                 ]
    #-I- For the recent tests in cad_mesh_analysis.ipynb I ran only 
    # 'save/plane_8_MRCNN__,'save/plane_car_2_PointRend__', 'save/Pascal3D_12_classes_12_PointRend__'
    # -I- 'bicycle_bus_car_bike_1_PointRend__' is missing one test_results.pkl for a certain symmetrization type 
    metric_3D_names = ['1-IoU_3D'] #computed at the post-processing 
    metric_3D_names_id = ['1-IoU_3D (id)', 'Hausdorff (id)'] #3D metrics without ICP
    loss_names =    ['L1 Metric', 'FID Metric', '1 - SSIM', 'Texture Comb', 'SSL Comb',   'Total Loss' ] #computin during network test
    # -------------- Quip configrations -----------------------------

    #metric_names = ['Perceptual Loss',  'Color Loss', 'Total Loss', 'Deformation Loss']
    #metric_names = ['Total Loss', 'L1 Metric', 'Perceptual Loss',  'Color Loss',  'Mask Loss']
    

    transf_opt = gu.transformation_options(iter_num=10, sample_num = 2000, voxel_num =100,  enable_rescale = False, verbose =True)
    transf_opt.rewrite = False
    pipeline_opt = gu.pipeline_options()
    CAD_path   = "/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD_preprocessed_tri"


    # symmetrization hss no impact on camera losses becuase camera loss is predicted directly from RGB 
    #loss_names =    ['L1 Metric', 'FID Metric', '1 - SSIM', 'Texture Comb', 'SSL Comb', 'Camera Loss', 'Camera Quat Reg', 'Color Loss',  'Mask Loss' ] #computin during network test
    #loss_names =    ['L1 Metric', 'FID Metric', '1 - SSIM', 'Texture Comb', 'SSL Comb', '1 - IoU_2D',  'Perceptual Loss',  'Color Loss',  'Mask Loss' ] #computin during network test

    metric_names =  loss_names   + metric_3D_names + metric_3D_names_id
    #anchor_metric_indices = [0,1,2,3,4,5, -1,-2,-3] # for each metric m it finds symmetrization combinations per sample that obtains  the highest improvemnt in m 
    anchor_metric_indices = [0,1,2,3, -1,-2] # for each metric m it finds symmetrization combinations per sample that obtains  the highest improvemnt in m 
    #anchor_metric_indices = list(range(len(metric_names)))



    reportOrig = du.MetricReport(metric_names)
    #reportSymm_list = du.MetricReport(loss_metrics) for i in range(0,3)
    reportSymmBestPerMetric       = du.MetricReport(metric_names)  #separate selection of symmetrization type per metric and sample 
    reportSymmBestTypePerMetric   = du.MetricReport(metric_names)

    
    #reportSymmBest                = du.MetricReport(metric_names)   #selecting best symmetrization per sample according to the anchor  metric
    #reportSymmBestType            = du.MetricReport(metric_names)
    reportSymmBest = []
    reportSymmBestType = []
    for _ in range(len(anchor_metric_indices)):
        reportSymmBest.append(du.MetricReport(metric_names) )
        reportSymmBestType.append(du.MetricReport(metric_names) )


    #reportSymmConst               = [du.MetricReport(metric_names), du.MetricReport(metric_names), du.MetricReport(metric_names)] 
    #symmConstNames                = ['Left-to-Right',               'Right-to-Light',              'Center'                     ]
    #symmDict                = {0:'orig', 1:'l2r',  2:'r2l', 3:'cent'}
    
    
    #slice_number = 2 # add max number of horizontal slices from 'result_path_list'
    slice_number = 3 # single slice => contant symmetrization per mesh 
    symmNames,symmNameDict, symmIndDict  = SymmetrizationTypeNames(slice_number)
    symm_num = len(symmNames)
    reportSymmConst                      = [du.MetricReport(metric_names) for _ in range(symm_num-1)] #exclude non_symmetric meshes for  symm_type=symm_index = 0 

    enable_parallel_run  = True # <== set True to compute fast 3D metrics for new tests. Comparing exisining metrics sequentially  runs faster 
    recompute_3d_metrics = False #set it for  tests in wich you need to recompute 3D metrics 
    pd.set_option("display.precision", 5) #number of digits in tables

    diff_symm_pttrn = "_[0-9]"*slice_number

    #"_".join(["[0-9]"]*slice_number)
    for result_path in result_path_list: 
        test_name = result_path.split('/')[-1]
        test_pkls_different_symm =   glob.glob(f'{result_path}-symmetric{diff_symm_pttrn}/*/test_results.pkl')
        test_pkls_same_symm      =   glob.glob(f'{result_path}-symmetric_[0-9]/*/test_results.pkl')
        test_pkls_non_symm       =   glob.glob(f'{result_path}-non_symmetric/*/test_results.pkl')
        test_pkls                =   test_pkls_different_symm + test_pkls_same_symm + test_pkls_non_symm
        #test_pkls = glob.glob(f'{result_path}-*/*/test_results.pkl')
        print(f'test root: {result_path}, pkl test result files: {test_pkls}')
        #metrics_list = [[],[],[],[]] #metrics per each type of symmetrization 
        metrics_list =[ [] for s in range(symm_num)]

        #test_root = os.path.split(test_pkls[0])[0]
        #symmetric_data = np.load(test_root + '/shape_symmetries.npy', allow_pickle = True).item() 
        process_test_results_pkl =  partial(process_test_results_pkl_full, symmIndDict, pipeline_opt, transf_opt, CAD_path)
        
        if enable_parallel_run:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                executor_results = executor.map(process_test_results_pkl,test_pkls)
        else:
            executor_results = map(process_test_results_pkl,test_pkls)


        for metrics, symm_indx  in executor_results:
            metrics_list[symm_indx] = metrics

        reportOrig.add_test(test_name,metrics_list[0])
            
        #compute loss for best symmetrization                
        metric_symm = np.stack(metrics_list[1:],axis=2)
        
        best_symm_per_metric = metric_symm.min(axis=-1)
        best_symm_idx_per_metric   = metric_symm.argmin(axis=-1)
        reportSymmBestTypePerMetric.add_test(test_name,best_symm_idx_per_metric)
        reportSymmBestPerMetric.add_test(test_name, best_symm_per_metric)

        for anchor_i  in range(len(anchor_metric_indices)):
            metric_indx = anchor_metric_indices[anchor_i]

            best_symm_idx = metric_symm[:,metric_indx,:].argmin(axis=-1)
            best_symm      = np.take_along_axis(metric_symm, best_symm_idx[:,np.newaxis,np.newaxis], axis=2)[...,0]
            best_symm_idx_tiles = np.tile(metric_symm[:,0,:].argmin(axis=-1)[:, np.newaxis], (1,len(metric_names)) )
            reportSymmBestType[anchor_i].add_test(test_name,best_symm_idx_tiles)
            reportSymmBest[anchor_i].add_test(test_name, best_symm)
        
        # for symm_type in  range(len(symmConstNames) ):
        #     reportSymmConst[symm_type].add_test(test_name,metrics_list[1+symm_type])

        for symm_only_indx in  range(symm_num-1):
            reportSymmConst[symm_only_indx].add_test(test_name,metrics_list[symm_only_indx+1])
        



    str_list =  []
    
    best_symm_table = reportOrig.compare(reportSymmBestPerMetric, f"\n =================== Improvements achieved by selecting  Best Symmetrization per Sample  per metric (%) =====================")[0]
    #best_symm_table.to_excel('save/best_symmetrization_per_metric.xlsx') #need to "pip install openpyxl"
    #reportOrig.compare(reportSymmBest, f"\n =================== Improvements achieved by selecting  Best Symmetrization per Sample  with respect to {metric_names[anchor_metric_indx]} (%) =====================")[0]

    print('\n\n >>>>>>>>>>>>>>>>>>>>>>>>>>  Improvements achieved by selecting  Best Symmetrization per Sample for a single  metrics  <<<<<<<<<<<<<')
    for anchor_i  in range(len(anchor_metric_indices)):
        reportOrig.compare(reportSymmBest[anchor_i], f"\n =================== Improvements achieved by Best Symmetrization for  {metric_names[anchor_metric_indices[anchor_i]]} (%) =====================")[0]

    #reportOrig.compare(reportSymmLR,   "\n =================== Improvements achieved by constant Left-to-Right Symmetrization  (%) =====================")[0]
    print('\n >>>>>>>>>>>>>>>>>>>>>>>>>>    CONSTANT SYMMETRIZATIONS  <<<<<<<<<<<<<')
    for symm_idx in  range(symm_num-1):
        reportOrig.compare(reportSymmConst[symm_idx],   f"\n =================== Improvements achieved by constant {symmNames[symm_idx+1]} Symmetrization  (%) =====================")[0]

    with open('save/post_processing.pkl', 'wb') as fd: #used in cad_mesh_analysis.ipynb 
        anchor_metric_names = du.List(metric_names)[anchor_metric_indices]
        pickle.dump({'reportSymmBest':reportSymmBest, 'reportSymmBestPerMetric':reportSymmBestPerMetric, 'reportSymmConst':reportSymmConst,\
                     'reportOrig':reportOrig, 'reportSymmBestType':reportSymmBestType, \
                     'anchor_metric_names':anchor_metric_names, 'anchor_metric_indices':anchor_metric_indices, 'metric_names':metric_names, \
                     'result_path_list':result_path_list, 'symmNames':symmNames } , fd)

    print('end')
# %% 