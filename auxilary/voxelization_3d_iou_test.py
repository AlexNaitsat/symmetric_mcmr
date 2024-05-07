import numpy as np
import pyvista as pv
from pyvista import examples
import igl 

## ==> Tested on win10 with "geometry_p37" conda env

#   # Installations in powershell fow win 10
#   # python interpreter=C:\Users\anaitsat\Miniconda3\envs\geometry_p37\python.exe
#   conda create --name geometry_p37 python=3.7 
#   conda activate geometry_p37
#   conda install -c conda-forge pyvista
#   pip install matplotlib
#   conda install -c conda-forge igl


def ObjMesh2PolyData(mesh_file):
    V, F = igl.read_triangle_mesh(mesh_file)
    return  pv.PolyData(V, MeshFace2PolyFace(F)), V, F

def MeshFace2PolyFace(F):
    face_num, edge_num = F.shape
    return  np.concatenate( (np.repeat(edge_num, face_num)[:, np.newaxis], F), axis = 1 ).flatten()

def Compute3DIoU(Vs, Fs, Vt, Ft):
    mesh_s = pv.PolyData(Vs, MeshFace2PolyFace(Fs))
    mesh_t = pv.PolyData(Vt, MeshFace2PolyFace(Ft))


    density = mesh_t.length / 100
    x_min, x_max, y_min, y_max, z_min, z_max = mesh_t.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(x, y, z)
    ugrid = pv.UnstructuredGrid(grid)

    # get part of the mesh within the mesh's bounding surface.
    selection_t = ugrid.select_enclosed_points(mesh_t.extract_surface(),
                                            tolerance=0.0,
                                            check_surface=False)
                                        
    selection_s = ugrid.select_enclosed_points(mesh_s.extract_surface(),
                                            tolerance=0.0,
                                            check_surface=False)


    mask_t = selection_t.point_arrays['SelectedPoints'].view(np.bool)
    mask_t = mask_t.reshape(x.shape)
    mask_s = selection_s.point_arrays['SelectedPoints'].view(np.bool)
    mask_s = mask_s.reshape(x.shape)
    intersect_mask = np.logical_and(mask_s, mask_t)

    # iou_mask = np.zeros_like(x, dtype=int)
    # iou_mask[np.nonzero(intersect_mask)] = 1
    # t_minus_s_mask = np.logical_and(mask_t, np.logical_not(mask_s))
    # iou_mask[np.nonzero(t_minus_s_mask)] = 2
    # s_minus_t_mask = np.logical_and(np.logical_not(mask_t), mask_s)
    # iou_mask[np.nonzero(s_minus_t_mask)] = 3

    return  intersect_mask.sum()/np.logical_or(mask_s, mask_t).sum()

    

#from file 
# rec_file ="car.obj"
rec_file = "car_aligned.obj"
#cad_file = "car_01_clean_solid_voxrem_dec_smooth.obj" #there are not triangle faces !
cad_file = "car_01_clean_solid_voxrem_dec_smooth_symm_tri.obj" # " v.read" loads wrong vertices for this file, while initlization via libigl is correct 



cad_file_list =   [ "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_01_clean_solid_voxrem_dec_smooth_symm_tri.obj", \
                    "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_02_clean_solid_voxrem_dec_smooth_symm_tri.obj", \
                    "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_03_clean_solid_voxrem_dec_smooth_symm_tri.obj", \
                    "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_04_clean_manual_symm_tri.obj", \
                    "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_05_clean_manual_v4_symm_tri.obj", \
                    "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_06_clean_manual_symm_tri.obj", \
                    "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_07_clean_manual_v4_symm_tri.obj", \
                    "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_08_solid_symm_tri.obj", \
                    "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_09_clean_solid_voxrem_dec_symm_tri.obj", \
                    "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_tri\\car\\car_10_clean_manual_v9_symm_tri.obj" ]

rec_file_list =  [ "car_aligned.obj", \
                    "car_aligned.obj", \
                    "car_aligned.obj", \
                    "car_aligned.obj", \
                    "car_aligned.obj", \
                    "car_aligned.obj", \
                    "car_aligned.obj", \
                    "car_aligned.obj", \
                    "car_aligned.obj", \
                    "car_aligned.obj" ]
                    


p = pv.Plotter()
p.add_mesh(ObjMesh2PolyData(rec_file_list[0])[0], show_edges=True)
p.add_title(f"Reconstructed mesh")
p.show()

# mesh = pv.read("car_01_clean_solid_voxrem_dec_smooth_symm_tri.obj")  loads into "mesh.points" wrong coordinates ! (and it also loads normals)
# define pv mesh trough libigl read_triangle_mesh 
load_meshes_direcly = False

#cad_file_list = cad_file_list[0:5]
#file_indices = range(len(cad_file_list))
#file_indices = range(5)    #first half 
file_indices = range(5,10) #secand half 
p = pv.Plotter(shape=(len(file_indices), 3))
iou_score = np.zeros(len(file_indices))
for i in range(len(file_indices)):
    fi = file_indices[i]
    
    rec_file = rec_file_list[fi] 
    cad_file = cad_file_list[fi]

    if load_meshes_direcly:
        rec_mesh = pv.read(rec_file)  #could be wrong for obj files with complex format
        cad_mesh = pv.read(cad_file)
    else: #testing mesh creation from face/vertex arrays
        # rec_V, rec_F = igl.read_triangle_mesh(rec_file)
        # cad_V, cad_F = igl.read_triangle_mesh(cad_file)
        # cad_mesh = pv.PolyData(cad_V, MeshFace2PolyFace(cad_F))
        # rec_mesh = pv.PolyData(rec_V, MeshFace2PolyFace(rec_F))
        cad_mesh, cad_V, cad_F = ObjMesh2PolyData(cad_file)
        rec_mesh, rec_V, rec_F = ObjMesh2PolyData(rec_file)
    cad_mesh
    rec_mesh




    ###############################################################################
    cpos = [
        (7.656346967151718, -9.802071079151158, -11.021236183314311),
        (0.2224512272564101, -0.4594554282112895, 0.5549738359311297),
        (-0.6279216753504941, -0.7513057097368635, 0.20311105371647392),
    ]
    ###############################################################################
    # Create  a voxel grid from mesh interior 
    #voxels = pv.voxelize(surface, density=surface.length / 200, check_surface=False)
    density = cad_mesh.length / 100
    x_min, x_max, y_min, y_max, z_min, z_max = cad_mesh.bounds
    x = np.arange(x_min, x_max, density)
    y = np.arange(y_min, y_max, density)
    z = np.arange(z_min, z_max, density)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(x, y, z)
    ugrid = pv.UnstructuredGrid(grid)

    # get part of the mesh within the mesh's bounding surface.
    cad_selection = ugrid.select_enclosed_points(cad_mesh.extract_surface(),
                                            tolerance=0.0,
                                            check_surface=False)
                                        
    rec_selection = ugrid.select_enclosed_points(rec_mesh.extract_surface(),
                                            tolerance=0.0,
                                            check_surface=False)


    cad_mask = cad_selection.point_arrays['SelectedPoints'].view(np.bool)
    cad_mask = cad_mask.reshape(x.shape)
    p.subplot(i, 0)
    p.add_mesh(grid.points, scalars=cad_mask.astype(int), opacity=0.3, show_scalar_bar=False)
    p.add_title(f"CAD {fi}", font_size=10)
    #pv.plot(grid.points, scalars=cad_mask, opacity=0.3)


    rec_mask = rec_selection.point_arrays['SelectedPoints'].view(np.bool)
    rec_mask = rec_mask.reshape(x.shape)
    #pv.plot(grid.points,   scalars=rec_mask, opacity=0.3)
    p.subplot(i, 1)
    p.add_mesh(grid.points, scalars=rec_mask.astype(int), opacity=0.3, show_scalar_bar=False)
    #p.add_title(f"mesh {fi}", font_size=10)
    p.add_title(f"mesh", font_size=10)


    iou_mask = np.zeros_like(x, dtype=int)
    cad_mesh_intersect_mask = np.logical_and(rec_mask, cad_mask)
    iou_mask[np.nonzero(cad_mesh_intersect_mask)] = 1

    cad_minus_mesh_mask = np.logical_and(cad_mask, np.logical_not(rec_mask))
    iou_mask[np.nonzero(cad_minus_mesh_mask)] = 2

    mesh_minus_cad_mask = np.logical_and(np.logical_not(cad_mask), rec_mask)
    iou_mask[np.nonzero(mesh_minus_cad_mask)] = 3

    # annotations = {
    #     0: "empty voxels",
    #     1: "CAD & mesh",
    #     2: "CAD - mesh",
    #     3: "mesh - CAD",
    # }
    #p = pv.Plotter()
    #sargs = dict(interactive=True, vertical=True, above_label="",n_labels=0)  # Simply make the bar interactive
    #p.add_mesh(grid.points, scalars=iou_mask, annotations = annotations, cmap= ['black', 'red', 'blue', 'green',], opacity=0.3, scalar_bar_args=sargs)
    p.subplot(i, 2)
    sargs = dict(vertical=True, above_label="",n_labels=0)  # Simply make the bar interactive
    p.add_mesh(grid.points, scalars=iou_mask, cmap= ['black', 'red', 'blue', 'green'], opacity=0.3, show_scalar_bar=False)
    
    
    

    iou_score[i] = cad_mesh_intersect_mask.sum()/np.logical_or(rec_mask, cad_mask).sum()
    p.add_title(f"3D IoU = {iou_score[i]:.3f}", font_size=10)

#p.set_background('w', all_renderers=True) #need to change font color too 
#p.add_scalar_bar(vertical=True, above_label="",n_labels=0)
print(f'average 3D IoU={iou_score.mean()}')

iou_3D = np.zeros_like(iou_score)
for i in range(iou_3D.shape[0]):
    rec_V, rec_F = igl.read_triangle_mesh(rec_file_list[file_indices[i]])
    cad_V, cad_F = igl.read_triangle_mesh(cad_file_list[file_indices[i]])
    iou_3D[i] = Compute3DIoU(rec_V,rec_F, cad_V,cad_F)

print(f'average 3D IoU recheck={iou_3D.mean()}')


p.show()

p1 = pv.Plotter()
annotations = {
    0: "empty voxels",
    1: "CAD & mesh",
    2: "CAD - mesh",
    3: "mesh - CAD",
}
sargs = dict(interactive=True, vertical=True, above_label="",n_labels=0)  # Simply make the bar interactive
p1.add_mesh(grid.points, scalars=iou_mask, annotations = annotations, cmap= ['black', 'red', 'blue', 'green',], opacity=0.3, scalar_bar_args=sargs)
#p1.add_title(f"Colorbar and annotations", font_size=12)
p1.show()