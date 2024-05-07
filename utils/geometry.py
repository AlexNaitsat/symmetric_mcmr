import math
from typing import Union
from importlib_metadata import metadata

import torch
import numpy as np
import igl 
import os
from scipy.spatial.distance import directed_hausdorff
import open3d as o3d
import pyvista as pv


#for mesh upscaling  DR test 
from utils import mesh as mesh_utils
from utils.image import SDR_to_numpy
import cv2
from utils.debug_utils import print_class_attributes


def MeshFace2PolyFace(F):
    face_num, edge_num = F.shape
    return  np.concatenate( (np.repeat(edge_num, face_num)[:, np.newaxis], F), axis = 1 ).flatten()

def VoxelizeMesh(Vt, Ft, opt):
    mesh_t = pv.PolyData(Vt, MeshFace2PolyFace(Ft))

    density = mesh_t.length / opt.voxel_num
    x_min, x_max, y_min, y_max, z_min, z_max = mesh_t.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(x, y, z)
    ugrid = pv.UnstructuredGrid(grid)
    grid_shape = x.shape
    selection_t = ugrid.select_enclosed_points(mesh_t.extract_surface(),
                                        tolerance=0.0,
                                        check_surface=False)
    mask_t = selection_t.point_arrays['SelectedPoints'].view(np.bool)
    mask_t = mask_t.reshape(grid_shape)
    
    return ugrid, mask_t, grid_shape



def OneMinus3DIoU_fast(Vs, Fs, Vt, Ft,metadata, opt):
    iuo_3d = Compute3DIoU_fast(Vs, Fs, Vt, Ft,metadata, opt)
    return 1-iuo_3d


def Compute3DIoU_fast(Vs, Fs, Vt, Ft,metadata, opt):
    
    cad_npy = os.path.splitext(metadata['t_file'])[0] + f"-v{opt.voxel_num}.npy"
    data = np.load(cad_npy, allow_pickle = True).item()
    ugrid      = data['ugrid']
    mask_t     = data['mask_t']
    grid_shape = data['grid_shape']

    # # data from npy should be the same as here:    
    # ugrid, mask_t = VoxelizeMesh(Vt, Ft, opt)
    
    mesh_s = pv.PolyData(Vs, MeshFace2PolyFace(Fs))
    selection_s = ugrid.select_enclosed_points(mesh_s.extract_surface(),
                                            tolerance=0.0,
                                            check_surface=False)

    mask_s = selection_s.point_arrays['SelectedPoints'].view(np.bool)
    mask_s = mask_s.reshape(grid_shape)

    intersect_mask = np.logical_and(mask_s, mask_t)
    return  (intersect_mask.sum()/np.logical_or(mask_s, mask_t).sum()).tolist()

def Compute3DIoU(Vs, Fs, Vt, Ft,  metadata, opt):
    mesh_s = pv.PolyData(Vs, MeshFace2PolyFace(Fs))
    mesh_t = pv.PolyData(Vt, MeshFace2PolyFace(Ft))


    
    #rebuild voxel grid from target bbox and given number of voxels
    density = mesh_t.length / opt.voxel_num
    x_min, x_max, y_min, y_max, z_min, z_max = mesh_t.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(x, y, z)
    ugrid = pv.UnstructuredGrid(grid)
    grid_shape = x.shape
    # get part of the mesh within the mesh's bounding surface.
    
    selection_t = ugrid.select_enclosed_points(mesh_t.extract_surface(),
                                            tolerance=0.0,
                                            check_surface=False)
    selection_s = ugrid.select_enclosed_points(mesh_s.extract_surface(),
                                            tolerance=0.0,
                                            check_surface=False)

    mask_t = selection_t.point_arrays['SelectedPoints'].view(np.bool)
    #mask_t = mask_t.reshape(x.shape)
    mask_t = mask_t.reshape(grid_shape)
    mask_s = selection_s.point_arrays['SelectedPoints'].view(np.bool)
    #mask_s = mask_s.reshape(x.shape)
    mask_s = mask_s.reshape(grid_shape)

    intersect_mask = np.logical_and(mask_s, mask_t)

    # iou_mask = np.zeros_like(x, dtype=int)
    # iou_mask[np.nonzero(intersect_mask)] = 1
    # t_minus_s_mask = np.logical_and(mask_t, np.logical_not(mask_s))
    # iou_mask[np.nonzero(t_minus_s_mask)] = 2
    # s_minus_t_mask = np.logical_and(np.logical_not(mask_t), mask_s)
    # iou_mask[np.nonzero(s_minus_t_mask)] = 3

    return  (intersect_mask.sum()/np.logical_or(mask_s, mask_t).sum()).tolist()


def x_rot(alpha: float,
          clockwise: bool=False,
          pytorch: bool=False
          ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around X axis (default: counter-clockwise).
    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around X axis.
    """
    if pytorch:
        cx = torch.cos(alpha)
        sx = torch.sin(alpha)
    else:
        cx = np.cos(alpha)
        sx = np.sin(alpha)

    if clockwise:
        sx *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([one, zero, zero], dim=1),
                         torch.stack([zero, cx, -sx], dim=1),
                         torch.stack([zero, sx, cx], dim=1)], dim=0)
    else:
        rot = np.asarray([[1., 0., 0.],
                          [0., cx, -sx],
                          [0., sx, cx]], dtype=np.float32)
    return rot


def y_rot(alpha: float,
          clockwise: bool=False,
          pytorch: bool=False
          ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around Y axis (default: counter-clockwise).
    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around Y axis.
    """
    if pytorch:
        cy = torch.cos(alpha)
        sy = torch.sin(alpha)
    else:
        cy = np.cos(alpha)
        sy = np.sin(alpha)

    if clockwise:
        sy *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cy, zero, sy], dim=1),
                         torch.stack([zero, one, zero], dim=1),
                         torch.stack([-sy, zero, cy], dim=1)], dim=0)
    else:
        rot = np.asarray([[cy, 0., sy],
                          [0., 1., 0.],
                          [-sy, 0., cy]], dtype=np.float32)
    return rot


def z_rot(alpha: float,
          clockwise: bool=False,
          pytorch: bool=False
          ) -> Union[np.ndarray, torch.Tensor]:
    """
    Compose a rotation matrix around Z axis (default: counter-clockwise).
    :param alpha: Rotation angle in radians.
    :param clockwise: Default rotation convention is counter-clockwise.
     In case the `clockwise` flag is set, the sign of `sin(alpha)` is reversed
     to rotate clockwise.
    :param pytorch: In case the `pytorch` flag is set, all operation are
     between torch tensors and a torch.Tensor is returned .
    :return rot: Rotation matrix around Z axis.
    """
    if pytorch:
        cz = torch.cos(alpha)
        sz = torch.sin(alpha)
    else:
        cz = np.cos(alpha)
        sz = np.sin(alpha)

    if clockwise:
        sz *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cz, -sz, zero], dim=1),
                         torch.stack([sz, cz, zero], dim=1),
                         torch.stack([zero, zero, one], dim=1)], dim=0)
    else:
        rot = np.asarray([[cz, -sz, 0.],
                          [sz, cz, 0.],
                          [0., 0., 1.]], dtype=np.float32)

    return rot


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Return intrinsics camera matrix with square pixel and no skew.
    :param focal: Focal length
    :param cx: X coordinate of principal point
    :param cy: Y coordinate of principal point
    :return K: intrinsics matrix of shape (3, 3)
    """
    return np.asarray([[fx, 0., cx],
                       [0., fy, cy],
                       [0., 0., 1.]])


def pascal_vpoint_to_extrinsics(az_deg: float,
                                el_deg: float,
                                radius: float):
    """
    Convert Pascal viewpoint to a camera extrinsic matrix which
     we can use to project 3D points from the CAD
    :param az_deg: Angle of rotation around X axis (degrees)
    :param el_deg: Angle of rotation around Y axis (degrees)
    :param radius: Distance from the origin
    :return extrinsic: Extrinsic matrix of shape (4, 4)
    """
    az_ours = np.radians(az_deg - 90)
    el_ours = np.radians(90 - el_deg)

    # Compose the rototranslation for a camera with look-at at the origin
    Rc = z_rot(az_ours) @ y_rot(el_ours)
    Rc[:, 0], Rc[:, 1] = Rc[:, 1].copy(), Rc[:, 0].copy()
    z_dir = Rc[:, -1] / np.linalg.norm(Rc[:, -1])
    Rc[:, -1] *= -1  # right-handed -> left-handed
    t = np.expand_dims(radius * z_dir, axis=-1)

    # Invert camera roto-translation to get the extrinsic
    #  see: http://ksimek.github.io/2012/08/22/extrinsic/
    extrinsic = np.concatenate([Rc.T, -Rc.T @ t], axis=1)
    return extrinsic


def project_points(points_3d: np.array,
                   intrinsic: np.array,
                   extrinsic: np.array,
                   scale: float = 1.) -> np.array:
    """
    Project 3D points in 2D according to pinhole camera model.

    :param points_3d: 3D points to be projected (n_points, 3)
    :param intrinsic: Intrinsics camera matrix
    :param extrinsic: Extrinsics camera matrix
    :param scale: Object scale (default: 1.0)
    :return projected: 2D projected points (n_points, 2)
    """
    n_points = points_3d.shape[0]

    assert points_3d.shape == (n_points, 3)
    assert extrinsic.shape == (3, 4) or extrinsic.shape == (4, 4)
    assert intrinsic.shape == (3, 3)

    if extrinsic.shape == (4, 4):
        if not np.all(extrinsic[-1, :] == np.asarray([0, 0, 0, 1])):
            raise ValueError('Format for extrinsic not valid')
        extrinsic = extrinsic[:3, :]

    points3d_h = np.concatenate([points_3d, np.ones(shape=(n_points, 1))], 1)

    points3d_h[:, :-1] *= scale
    projected = intrinsic @ (extrinsic @ points3d_h.T)
    projected /= projected[2, :]
    projected = projected.T
    return projected[:, :2]


def project_points_tensor(points_3d: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor:
    """
    Project 3D points in 2D according to pinhole camera model.

    :param points_3d: 3D points to be projected (n_points, 3)
    :param intrinsic: Intrinsic camera matrix
    :param extrinsic: Extrinsic camera matrix
    :return projected: 2D projected points (n_points, 2)
    """
    n_points = points_3d.shape[1]

    assert points_3d.shape == (points_3d.shape[0], n_points, 3)
    assert extrinsic.shape[1:] == (3, 4) or extrinsic.shape[1:] == (4, 4)
    assert intrinsic.shape[1:] == (3, 3)

    if extrinsic.shape[1:] == (4, 4):
        if not torch.all(extrinsic[:, -1, :] == torch.FloatTensor([0, 0, 0, 1])):
            raise ValueError('Format for extrinsic not valid')
        extrinsic = extrinsic[:, 3, :]

    points3d_h = torch.cat([points_3d, torch.ones(points_3d.shape[0], n_points, 1).to(points_3d.device)], 2)

    projected = intrinsic @ extrinsic @ points3d_h.transpose(1, 2)
    projected = projected / projected[:, 2, :][:, None, :]
    projected = projected.transpose(1, 2)

    return projected[:, :, :2]


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                      [m01+m10,     m11-m00-m22, 0.0,         0.0],
                      [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                      [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def hamilton_product(qa, qb):
    """Multiply qa by qb.
    Args:
        qa: B X N X 4 quaternions
        qb: B X N X 4 quaternions
    Returns:
        q_mult: B X N X 4
    """
    qa_0 = qa[..., 0]
    qa_1 = qa[..., 1]
    qa_2 = qa[..., 2]
    qa_3 = qa[..., 3]

    qb_0 = qb[..., 0]
    qb_1 = qb[..., 1]
    qb_2 = qb[..., 2]
    qb_3 = qb[..., 3]

    # See https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
    q_mult_0 = qa_0 * qb_0 - qa_1 * qb_1 - qa_2 * qb_2 - qa_3 * qb_3
    q_mult_1 = qa_0 * qb_1 + qa_1 * qb_0 + qa_2 * qb_3 - qa_3 * qb_2
    q_mult_2 = qa_0 * qb_2 - qa_1 * qb_3 + qa_2 * qb_0 + qa_3 * qb_1
    q_mult_3 = qa_0 * qb_3 + qa_1 * qb_2 - qa_2 * qb_1 + qa_3 * qb_0

    return torch.stack([q_mult_0, q_mult_1, q_mult_2, q_mult_3], dim=-1)


def axisangle2quat(axis, angle):
    """
    axis: B x 3: [axis]
    angle: B: [angle]
    returns quaternion: B x 4
    """
    axis = torch.nn.functional.normalize(axis, dim=-1)
    angle = angle.unsqueeze(-1) / 2
    quat = torch.cat([angle.cos(), angle.sin() * axis], dim=-1)
    return quat

#my new 3D utils 
def reflectX(xyz):
    if torch.is_tensor(xyz):
        if len(xyz.shape)==2: #should it be len(xyz.shape)> 2 ??!
            return torch.cat( (-xyz[:,0:1], xyz[:,1:3]), 1) 
        else:
            #return torch.cat( (-xyz[0:1], xyz[1:3]) ) 
            return torch.cat( (-xyz[:,:,0:1], xyz[:,:,1:3]), 2)
    else:   
        if len(xyz.shape)==2:
            return  np.concatenate( (-xyz[:,0:1], xyz[:,1:3]), axis=1) 
        else:
            return  np.concatenate( (-xyz[0:1], xyz[1:3]) ) 

def reflectY(xyz):
    if torch.is_tensor(xyz):
        if len(xyz.shape)==2:  #should it be len(xyz.shape)> 2 ??!
            return torch.cat( (xyz[:,0:1], -xyz[:,1:2], xyz[:,1:3]), 1) 
        else:
            return torch.cat( (xyz[0:1], -xyz[1:2], xyz[1:3]) ) 
    else:   
        if len(xyz.shape)==2:
            return  np.concatenate( (xyz[:,0:1], -xyz[:,1:2], xyz[:,1:3]), axis=1) 
        else:
            return  np.concatenate( (xyz[0:1], -xyz[1:2], xyz[1:3]) ) 

def invert_face_orientation(F):
    if torch.is_tensor(F):
        if len(F)==2:
            return torch.cat( (F[:,0:1], F[:,2:3], F[:,1:2]  ), 1) 
        else:
            return torch.cat( (F[0:1], F[2:3], F[1:2] ) ) 
    else:
        if len(F)==2:
            return  np.concatenate( (F[:,0:1], F[:,2:3], F[:,1:2]  ),  axis=1) 
        else:
            return np.concatenate( (F[0:1], F[2:3], F[1:2] ) ) 




def HaussdorffDistancePairs(Vs,Fs, Vt,Ft):
    s2t = igl.point_mesh_squared_distance(Vs, Vt, Ft)
    s2t_i_max = s2t[0].argmax()
    s2t_dist =  math.sqrt(s2t[0].max())
    t2s = igl.point_mesh_squared_distance(Vt, Vs, Fs)    
    t2s_i_max = t2s[0].argmax()
    t2s_dist =  math.sqrt(t2s[0].max())
    # print(f'max deviation =    { (s2t[0] - np.linalg.norm(Vs-s2t[2], axis = 1)).max()}' )
    # print(f'max deviation =    { (t2s[0] - np.linalg.norm(Vt-t2s[2], axis = 1)).max()}' )
    if s2t_dist > t2s_dist:
        return s2t_dist,  Vs[s2t_i_max,:], s2t[2][s2t_i_max]
    else:
        return t2s_dist,  t2s[2][t2s_i_max], Vt[t2s_i_max,:]

    
    



# def Haussdorff_mesh_distance(Va,Fa, Vb,Fb, subdiv=0, verbose =False):
#     if subdiv > 0:
#      d_hauss = igl.hausdorff(Va,Fa,Va,Fa)

def measure_symmetries(V, F, V_right, V_opposite, opt={}):
    if 'align' in opt.keys() and opt['align']:
        V = np.copy(V) -  np.mean(V,axis=0) #align shape to 0,0,0  

    shape_scale = V[:,0].max() - V[:,0].min()
    
    #Deformation (index) symmetry
    LR_diff_unscaled = np.linalg.norm(V[V_right,:] - reflectX(V[V_opposite[V_right],:]), axis=1)
    LR_diff = LR_diff_unscaled/shape_scale
    LR_mean = (LR_diff.sum()/ V_right.shape[0]).tolist()
    LR_max  = LR_diff.max().tolist()

    #Geometric (Hausdorff) symmetry
    V_ref = reflectX(V)
    H_symm_error = igl.hausdorff(V,F,V_ref,F)/shape_scale

    if 'verbose' in opt.keys() and opt['verbose']:
        print(f'Mean-max Index and geometric Symmetry errors = ({LR_mean:.4f}, {LR_max:.4f},{H_symm_error:.4f})')


    return LR_max, LR_mean, H_symm_error

# validate it
def signed_svd(M):
    U, S, V_T = np.linalg.svd(M- np.mean(M,axis=0))
    if np.linalg.det(V_T) < 0:
        row_index= np.argmax(np.linalg.norm(V_T,axis = 1)) #row with largest norm has non-zero length 
        V_T[row_index,:] = -V_T[row_index,:]
        S[row_index]   = -S[row_index]
    return U, S, V_T  # M =U* np.diag(S)* V_T , where V_T is (positive) rotation 

# used to align principal components of source-target point clouds that were rotated by less than 45 degress 
def signed_sorted_svd(M):
    U, S, V_T = np.linalg.svd(M- np.mean(M,axis=0))
    n = V_T.shape[0] 
    sorted_pos_axes = []
    V_T_sorted = np.zeros_like(V_T)
    S_sorted   = np.zeros_like(S)
    for i in range(n):
        e_i=np.zeros((n,1))
        e_i[i] =1 
        dots = V_T @  e_i
        j = np.argmax(abs(dots))
        V_T_sorted[i,:] = V_T[j,:] * np.sign(dots[j])
        S_sorted[i] = S[j] * np.sign(dots[j])

    return U, S_sorted, V_T_sorted


# ----  geometric source-target transformations
class pipeline_options: 
    def __init__(self, enable_icp=True, verbose = True, symmetrization_type = 0):
            self.enable_icp = enable_icp
            self.symmetrization_type = symmetrization_type
            self.verbose   = verbose
            self.IoU_3D_func = Compute3DIoU_fast 
    def __str__(self):
        return('Pipeline options:\n' + print_class_attributes(self))
            

class transformation_options: 
    def __init__(self, sample_num=1000, iter_num=10,\
                 enable_resample = False, enable_rescale = False,\
                 voxel_num =100, verbose = False):
            self.sample_num = sample_num
            self.iter_num   = iter_num
            self.enable_resample = enable_resample
            self.enable_rescale = enable_rescale
            self.voxel_num = voxel_num
            self.verbose = verbose 
    def report(self):
        print('Transformation options = ')


def align_source2target_PCA(Ver_source,F_source, Ver_target,F_target, sample_num, iter_num,  enable_resample = False):

    cent_source = np.mean(Ver_source,axis=0)
    cent_target = np.mean(Ver_target,axis=0)

    if sample_num is None: 
        U_source, S_source, V_sourceT = signed_sorted_svd(Ver_source) #U_source = Vert_source in the source's SVD coordinates 
        U_target, S_target, V_targetT = signed_sorted_svd(Ver_target) #U_source = Vert_source in the source's SVD coordinates 
    else:
        t_samples = np.unique(np.round(np.linspace(0,Ver_target.shape[0]-1, sample_num)).astype(int))
        s_samples = np.unique(np.round(np.linspace(0,Ver_source.shape[0]-1, sample_num)).astype(int))
        _, _, V_sourceT = signed_sorted_svd(Ver_source[s_samples]) #U_source = Vert_source in the source's SVD coordinates 
        _, _, V_targetT = signed_sorted_svd(Ver_target[t_samples]) #U_source = Vert_source in the source's SVD coordinates 

    trans =  cent_target - cent_source
    Rot =  V_sourceT.T @ V_targetT
    Ver_source_algined =  (Ver_source - cent_source)  @ Rot + trans #rotating source PCA to target PCA
    return Ver_source_algined, Rot, trans
    #print(f'S_source= {S_source}, S_target= {S_target}')

#def align_open3d_PC_ICP(Vs,Fs, Vt,Ft, sample_num, iter_num,  enable_rescale = False, verbose=False): 
def align_open3d_PC_ICP(Vs,Fs, Vt,Ft, opt): 
    source = o3d.geometry.PointCloud()
    source.points  = o3d.utility.Vector3dVector(Vs)
    target = o3d.geometry.PointCloud()
    target.points =  o3d.utility.Vector3dVector(Vt)
    trans_init  = np.eye(4)

    threshold = 0.01 * opt.iter_num # low threshold yelds better results 

    if opt.verbose:
        evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
                                                                threshold, trans_init)
        print(evaluation)

    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,\
                                                o3d.pipelines.registration.TransformationEstimationPointToPoint() )
    source.transform(reg_p2p.transformation)
    Vs_aligned = np.asarray(source.points)
    Rot =  reg_p2p.transformation[0:3,0:3] #not sure !
    t   =  reg_p2p.transformation[0:3,3]

    if opt.enable_rescale:
        Vs_algined_bbox = Vs_aligned.max(0)- Vs_aligned.min(0)
        Vt_bbox = Vt.max(0)- Vt.min(0)
        Vs_aligned *=  (Vt_bbox/Vs_algined_bbox)
    
    return  Vs_aligned, Rot, t

    #print(reg_p2p.transformation)
    #print(f'   Source-target dist before ICP = {max_mean_mesh_distance(Vs,Fs,Vt,Ft)}')
    #print(f'   Source-target dist after  ICP = {max_mean_mesh_distance(Vs_aligned,Fs,Vt,Ft)}')


def align_PC_ICP_init_with_PCA(Vs_,Fs, Vt,Ft, sample_num, iter_num,  enable_resample = False):
    Vs0, Rot0, t0 = align_source2target_PCA(Vs_,Fs, Vt,Ft, sample_num,iter_num, enable_resample)
    Vs, Rot, t    = align_source2target_PCA(Vs0,Fs, Vt,Ft,iter_num, enable_resample)
    return Vs, Rot*Rot0, Rot*t0+t
    


def align_source2target_pc_icp(Vs_,Fs, Vt,Ft, sample_num, iter_num,  enable_resample = False, select_best_iter = True):
    Vs = np.copy(Vs_)
    # if enable_rescale:
    #     Vs_bbox = Vs.max(0)- Vs.min(0)
    #     Vt_bbox = Vt.max(0)- Vt.min(0)
    #     Vs *=  (Vt_bbox/Vs_bbox)
    max_sample_num = min(Vs_.shape[0],Vt.shape[0])
    if sample_num > max_sample_num:
        print( f'Sample only {max_sample_num} out of {sample_num} points (not enough vertices for icp)')
        sample_num = max_sample_num


    Nt = igl.per_vertex_normals(Vt,Ft)
    norm_of_rows = np.linalg.norm(Nt, axis=1)
    Nt = Nt/ norm_of_rows[:, np.newaxis]#looks like nt=Nt= unit normals 

    nan_index   =  np.any(np.isnan(Nt),axis=1)
    if np.any(nan_index):
        Nt[nan_index,:] = Vt[nan_index,:]
        Nt_fixed = Vt[nan_index,] - np.mean(Vt,axis=0)
        Nt_fixed_norm = np.linalg.norm(Nt_fixed, axis=1)
        Nt_fixed = Nt_fixed/Nt_fixed_norm[:, np.newaxis]
        Nt_fixed[np.isclose(Nt_fixed_norm,0),:] = np.array([1,0,0])  #put arbitary unit norm for 0 vertices with nan normals
        Nt[nan_index,:] = Nt_fixed
    
    #nan_index = np.nonzero(np.isnan(nt))
    #norm_of_rows = np.linalg.norm(Nt, axis=1)

    
    t_samples = np.unique(np.round(np.linspace(0,Vt.shape[0]-1, sample_num)).astype(int))
    s_samples = np.unique(np.round(np.linspace(0,Vs.shape[0]-1, sample_num)).astype(int))
    sample_len = max(t_samples.shape[0],s_samples.shape[0])
    t_samples = t_samples[0:sample_len]
    s_samples = s_samples[0:sample_len]
    if enable_resample:
        t_samples = t_samples.tolist()
        s_samples = s_samples.tolist()

    if select_best_iter:
        best_iter = 0
        best_dist  = directed_hausdorff(Vs_[s_samples,:], Vt[t_samples,:])[0] #Fast Hausdorff dist between  point cloud samples
        #best_dist = np.sqrt(igl.point_mesh_squared_distance(Vs_, Vt, Ft)[0]).mean()
        #best_dist = max_mean_mesh_distance(Vs_,Fs, Vt,Ft)[2] #2=mean, mean symmetric distance
        best_result = (Vs_, np.eye(3), np.zeros((3,)) )

    for iter in range(iter_num):
        if enable_resample:
            t_samples = [i+iter if i+iter < sample_len else sample_len-iter-1 for i in t_samples]
            s_samples = [i+iter if i+iter < sample_len else sample_len-iter-1 for i in s_samples]
        Rot, t = igl.rigid_alignment(Vs[s_samples,:],Vt[t_samples,:],Nt[t_samples,:]) 
        Vs = Vs @ Rot.transpose() + t 
        assert not np.any(np.isnan(Vs))
        if select_best_iter:
            #iter_dist =  max_mean_mesh_distance(Vs,Fs, Vt,Ft)[2]
            #iter_dist =   np.sqrt(igl.point_mesh_squared_distance(Vs, Vt, Ft)[0]).mean()
            iter_dist  =  directed_hausdorff(Vs[s_samples,:], Vt[t_samples,:])[0]
            if iter_dist < best_dist:
               best_iter = iter
               best_dist = iter_dist
               best_result = (np.copy(Vs), Rot, t)

    if select_best_iter:
        return best_result
    else: 
        return Vs, Rot, t
    


def align_source2target_with_icp(Vs_,Fs, Vt,Ft, sample_num, iter_num, enable_rescale = True):
    Vs = np.copy(Vs_)
    if enable_rescale:
        Vs_bbox = Vs.max(0)- Vs.min(0)
        Vt_bbox = Vt.max(0)- Vt.min(0)
        Vs *=  (Vt_bbox/Vs_bbox)
    Rot, t = igl.iterative_closest_point(Vs,Fs,Vt,Ft,sample_num,iter_num) 
    Vs = Vs @ Rot.transpose() + t 
    return Vs, Rot, t


#returns in  index 0:  hasudorff dist.  = max{max(source-to-target dist.), max(target-to-source dist) }
#            index 1:  symmetric dist.  = max{mean(source-to-target dist.), mean(target-to-source dist) }
#            index 2:  symmetric dist.+ = max{  source-to-target dist. + target-to-source dist  }
def max_mean_mesh_distance(Vs,Fs, Vt, Ft):
    soucre2target = np.sqrt(igl.point_mesh_squared_distance(Vs, Vt, Ft)[0])
    target2source = np.sqrt(igl.point_mesh_squared_distance(Vt, Vs, Fs)[0])
    #soucre2target_max = soucre2target.max()
    #target2source_max = target2source.max()
    # return (max(soucre2target.max(), target2source.max()),
    return (max(soucre2target.max(), target2source.max()),  \
            max(soucre2target.mean(), target2source.mean()),\
                0.5*(soucre2target.mean()+ target2source.mean()) \
           )

def relative_symmetric_mesh_distances(Vs,Fs, Vt, Ft,metadata=None):
    Vs_size = Vs.max() - Vs.min()
    return (max_mean_mesh_distance(Vs,Fs, Vt, Ft)/Vs_size).tolist()

def tensors2mesh(Vt, Ft, batch = 0):
    if type(Vt).__module__ == np.__name__:
        V = Vt
    else:
        if len(Vt.shape) > 2:
            V = Vt[batch,:].clone().cpu().numpy()
        else:
            V = Vt.cpu().clone().numpy()

    if type(Ft).__module__ == np.__name__:
        F = Ft
    else:
        if len(Ft.shape) > 2:
            F = Ft[batch,:].clone().cpu().numpy()
        else:
            F = Ft.cpu().clone().numpy()
    return V, F

def mesh2tensors(V, F, device = 'cuda',  add_batch =True):
    #Vt = torch.from_numpy(V).unsqueeze(0).to(device)
    #Ft = torch.from_numpy(F).unsqueeze(0).to(device)
    Vt = torch.from_numpy(V).to(device)
    Ft = torch.from_numpy(F).to(device)
    if add_batch:
        Vt = Vt.unsqueeze(0)
        Ft = Ft.unsqueeze(0)
    return Vt, Ft

def centroid_subdivision(V, F):
    nV = V.shape[0]
    nF = F.shape[0]

    F123xyz =  V[F,:]  #Fx3x3 = F*[v1,v2,v3]*[x,y,z] 
    C = F123xyz.mean(axis = 1) #centroid per triangle 
    C_index = np.arange(nV,nV+nF,dtype=int)[:,np.newaxis]
    
    Vsub = np.row_stack((V,C))
    F1 = np.column_stack((C_index, F[:,0:2] ))
    F2 = np.column_stack((C_index, F[:,1:3] ))
    F3 = np.column_stack((C_index, F[:,2],F[:,0] ))

    Fsub =np.row_stack((F1, F2, F3))
    #print('writing ./debug/centroid_subdivision.obj')
    #print('writing ./debug/orginal.obj')
    #igl.write_obj('./debug/centroid_subdivision.obj',Vsub, Fsub)
    #igl.write_obj('./debug/original.obj',V, F)
    return Vsub, Fsub



def subdivide_mesh_wrapper(subdiv_func, V,F, Vinit, Finit, uvimage_pred, sub_mesh_file = 'subdivided_mesh', face_tex_size=6, ):
    V_sub_np, F_sub_np = subdiv_func(*tensors2mesh(V, F))
    V_sub, F_sub = mesh2tensors(V_sub_np, F_sub_np, V.device)


    #Vinit_sub, Finit_sub =  mesh2tensors( *subdiv_func(*tensors2mesh(Vinit, Finit)), Vinit.device, False )
    Vinit_sub, Finit_sub =  subdiv_func(*tensors2mesh(Vinit, Finit))

    #converting "uvimage_pred" texture to per-face texture of subdivided mesh 
    #1. subdivided  UV coorids. = spherical coords. of subdivided init shape 
    uv_sampler = mesh_utils.compute_uvsampler_softras(Vinit_sub, Finit_sub, tex_size=face_tex_size)
    uv_sampler = torch.FloatTensor(uv_sampler).cuda()
    nF = uv_sampler.size(0)
    nT = uv_sampler.size(1)

    #uv_sampler = self.uv_sampler.unsqueeze(0).repeat(feat.shape[0], 1, 1, 1, 1)
    uv_sampler = uv_sampler.unsqueeze(0).repeat(1, 1, 1, 1, 1)  #assumed that feat.shape[0]=1
    uv_sampler = uv_sampler.view(-1, nF, nT * nT, 2)
    tex_pred = torch.nn.functional.grid_sample(uvimage_pred, uv_sampler, align_corners=True)
    tex_pred = tex_pred.view(uvimage_pred.size(0), -1, nF, nT, nT).permute(0, 2, 3, 4, 1)
    tex_pred = tex_pred.unsqueeze(4).expand(tex_pred.shape[0], tex_pred.shape[1], tex_pred.shape[2],
                                                tex_pred.shape[3], nT, tex_pred.shape[4])
    #texture in per-face fromat 
    tex_pred = tex_pred[:, :, :, :, 0, :].view(tex_pred.shape[0], tex_pred.shape[1], nT * nT, 3)

    if sub_mesh_file:
        #UV_sub = mesh_utils.convert_3d_to_uv_coordinates(Vinit_sub)
        UV_sub = (1-mesh_utils.convert_3d_to_uv_coordinates(Vinit_sub))/2
        obj_sub_file      =  f'./debug/{sub_mesh_file}.obj'
        unwrapped_sub_file = f'./debug/{sub_mesh_file}_texture.png'

        print('writing ' + unwrapped_sub_file)
        cv2.imwrite(unwrapped_sub_file, SDR_to_numpy(uvimage_pred))
        print('writing ' + obj_sub_file)
        save_obj_with_uv(obj_sub_file, V_sub_np, F_sub_np, UV_sub, f'{sub_mesh_file}_texture.png')
        

    return V_sub, F_sub, tex_pred 






         

#save ovj file with per vertex UV and additiona metada 
def save_obj_with_uv(obj_file_name,V, F, UV, texture_file = None,  opt = {}, metadata = {}):
    #igl.write_obj(obj_file_name, V, F) #write geometry 
    status  = ''
    if not 'verbose' in opt:
        opt['verbose'] = False
    if not 'invert' in opt:
        opt['invert'] = False
    if texture_file:
        mtl_file       = os.path.splitext( os.path.basename(obj_file_name) )[0] + '.mtl'
        mtl_dir        = os.path.split(obj_file_name)[0]
        mtl_file_path  = os.path.join( mtl_dir, mtl_file)
        with open(mtl_file_path, 'w') as f:
             f.write(f'map_Kd {texture_file}')
        metadata["mtllib"] = mtl_file
        status += f'mtl {mtl_file_path}\n'
        status += f'texture {os.path.join(mtl_dir,texture_file)}\n'


    if not metadata is None: 
        if not "mtllib" in  metadata:
            metadata["mtllib"] = "default.mtl"
            #print("default.mtl")
            default_mtl = os.path.join(os.path.split(obj_file_name)[0], "default.mtl")
                # if not os.exist(default_mtl):
                #     with open(default_mtl, 'w') as f:
                #         f.write(f'map_Kd {texture_file}\n')  

    status += f'obj {obj_file_name}'
    if opt['invert']:
        F = np.stack((F[:,0],F[:,2],F[:,1]),axis = 1)

    with open(obj_file_name, 'w') as f: #write UV 
        for field in metadata.keys():
            f.write(f'{field} {metadata[field]}\n') 
        f.write('\n')

        for vertex in V:
            f.write('v %.8f %.8f %.8f \n' % (vertex[0], vertex[1], vertex[2]) )
        f.write('\n')

        for vertex_uv in UV:
            f.write('vt %.8f %.8f \n' % (vertex_uv[0], vertex_uv[1]) )

        for face in F:
            f.write('f %d/%d %d/%d %d/%d\n' % ( \
                     face[0] + 1,face[0] + 1,   face[1] + 1,face[1] + 1,  face[2] + 1, face[2] + 1))
        f.write('\n')
    if opt['verbose']:
        print(status)
    
#def symmetrize_mesh(V, data, left_to_right = True):
def symmetrize_mesh(V, data, symmetrization_type = 1):
    V_right = data['V_right']
    V_opposite = data['V_opposite']
    V_left     = V_opposite[V_right]
    V_center   = np.nonzero(np.arange(0,V_opposite.shape[0]) == V_opposite)[0]

    if symmetrization_type == 1:
        V_l2r =  np.copy(V)
        V_l2r[V_right,:] =   reflectX(V[V_opposite[V_right],:])
        V_l2r[V_center,0] = 0
        return V_l2r
    elif symmetrization_type == 0:
        V_r2l =  np.copy(V)
        V_r2l[V_left,:] =   reflectX(V[V_opposite[V_left],:] )
        V_r2l[V_center,0] = 0
        return V_r2l
    else:
        V_cent =  np.copy(V)
        V_cent[V_right,:]  =  0.5*V_cent[V_right,:] + 0.5*reflectX(V[V_opposite[V_right],:])
        V_cent[V_left,:]   =  reflectX(V_cent[V_right,:])
        V_cent[V_center,0] = 0
        return V_cent

#for geometric pipeline
def symmetrize_source_mesh(Vs,Fs, Vt,Ft, data, left_to_right = True):
    symmetrization_type =  int(left_to_right)  if type(left_to_right)==bool else left_to_right
    return symmetrize_mesh(Vs, data, symmetrization_type)


#using the same symmetrization implementation like "MCMRNet.symmetrize"
def symmetrize_source_mesh_network(Vs,Fs, Vt,Ft, data, enabled_symm_type):
    return symmetrize_mesh_network(Vs, data, enabled_symm_type)

def symmetrize_mesh_network(V, data, enabled_symm_type=1):
    if  not enabled_symm_type:
        return  V #disabled symmetrization (do I need V.copy())
            
    if type(enabled_symm_type) in  [list, tuple]: #differnt symmetrizations for different mesh parts
        return symmetrize_sliced_mesh(V, data, enabled_symm_type)

    vert = V.copy()
    V_right = data['V_right']
    V_opposite = data['V_opposite']
    V_left     = V_opposite[V_right]
    V_middle   = np.nonzero(np.arange(0,V_opposite.shape[0]) == V_opposite)[0]
    #print(f'const symm type: {enabled_symm_type}')
    symm_coeff = [0.0, 1.0, 0.5] [int(enabled_symm_type)-1] #1-> 0, 2->1, 3->0.5

    vert[V_right,:]  =    symm_coeff*         V[V_right,:]               + (1-symm_coeff)*reflectX(V[V_opposite[V_right],:])
    vert[V_left,:]   =    symm_coeff*reflectX(V[V_opposite[V_left],:])   + (1-symm_coeff)*         V[V_left,:]
    vert[V_middle,0] = 0

    return vert

#sV should be vertices of the initial sphere 
def add_horizontal_slices_to_symmetry(data, sV):
    V_right    = data['V_right']
    V_opposite = data['V_opposite']
    V_left     = V_opposite[V_right] 

    V_right_slice = [[],[]]
    V_left_slice  = [[],[]]
    V_right_slice[0] = V_right[ np.nonzero( sV[V_right,1]  > 0 )]
    V_right_slice[1] = V_right[ np.nonzero( sV[V_right,1] <= 0 )]
    V_left_slice[0]  = V_left[ np.nonzero(  sV[V_left,1]   > 0  ) ]
    V_left_slice[1]  = V_left[ np.nonzero(  sV[V_left,1]  <= 0  ) ]
    
    data['V_right_slice'] = V_right_slice
    data['V_left_slice']  = V_left_slice

    return V_right_slice, V_left_slice

def add_horizontal_slices_to_symmetry_pro(data, sV, slice_num):
    V_right    = data['V_right']
    V_opposite = data['V_opposite']
    V_left     = V_opposite[V_right] 

    slice_range  = np.linspace(sV[:,1].min(),   sV[:,1].max(), slice_num+1)
    

    V_right_slice = [ [] for _ in range(slice_num) ]
    V_left_slice  = [ [] for _ in range(slice_num) ]
    for i in range(slice_num):
        sRY = sV[V_right,1]
        sLY = sV[V_left,1]
        V_right_slice[slice_num-i-1] = V_right[ np.where( (sRY >  slice_range[i]) & (sRY <=  slice_range[i+1])  )] #slice_num-i-1 to get the same slices as in prev funtion for two slices
        V_left_slice[slice_num-i-1]  = V_left[  np.where( (sLY >  slice_range[i]) & (sLY <=  slice_range[i+1])  )] #after tests change it:  [slice_num-i-1] -> [i]
    
    data['V_right_slice'] = V_right_slice
    data['V_left_slice']  = V_left_slice

    return V_right_slice, V_left_slice

#use symmetry slices to get  more symmetrization options 
def symmetrize_sliced_mesh(V, data, enabled_symm_type=(1,1)):
    if  not enabled_symm_type:
        return  V #disabled symmetrization (do I need V.copy())
    vert = V.copy()
    vert = V.copy()
    V_right = data['V_right']
    V_opposite = data['V_opposite'] 
    V_left     = V_opposite[V_right] #left is matched to right, do I need to use "V_opposite" ??
    V_middle   = np.nonzero(np.arange(0,V_opposite.shape[0]) == V_opposite)[0]
    V_right_slice = data['V_right_slice']
    V_left_slice  = data['V_left_slice']
    #print(f'list of  symm types: {enabled_symm_type}')
    symm_coeff = [] # [[0]]*len(self.enable_symmetrization.enable_symmetrization)
    for i in range(len(enabled_symm_type)):
        if not (enabled_symm_type[i]):
            continue
        #symm_coeff.appned(   [0.0, 1.0, 0.5] [int(self.enable_symmetrization[i])-1] )
        symm_coeff = [0.0, 1.0, 0.5] [int(enabled_symm_type[i])-1] #1-> 1, 2->0, 3->0.5
        V_right  = V_right_slice[i]
        V_left   = V_left_slice[i]
        vert[V_right,:]  =    symm_coeff*         V[V_right,:]               + (1-symm_coeff)*reflectX(V[V_opposite[V_right],:])
        vert[V_left,:]   =    symm_coeff*reflectX(V[V_opposite[V_left],:])   + (1-symm_coeff)*         V[V_left,:]

    vert[V_middle,0] = 0

    return vert

