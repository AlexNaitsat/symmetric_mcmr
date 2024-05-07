""" 
General utils for meshplot notebook drawing
"""
#from tkinter import N
import numpy as np
import meshplot as mp
from IPython.display import  display, Markdown
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pythreejs as p3s
import copy 

def empty_meshplot(shading={}):
    view  = mp.Viewer(shading)
    display(view._renderer)
    return view

# based on https://github.com/skoch9/meshplot/issues/30  with my additional features 
# (need to rename 'self' input param !)
def add_transparent_mesh(self, v, f, c=None, uv=None, n=None, shading={}, opacity=0.6, texture_data=None):
    # if self is  None:
    #     self = empty_meshplot(shading)
    sh = self._Viewer__get_shading(shading)
    mesh_obj = {}

    #it is a tet
    if v.shape[1] == 3 and f.shape[1] == 4:
        f_tmp = np.ndarray([f.shape[0]*4, 3], dtype=f.dtype)
        for i in range(f.shape[0]):
            f_tmp[i*4+0] = np.array([f[i][1], f[i][0], f[i][2]])
            f_tmp[i*4+1] = np.array([f[i][0], f[i][1], f[i][3]])
            f_tmp[i*4+2] = np.array([f[i][1], f[i][2], f[i][3]])
            f_tmp[i*4+3] = np.array([f[i][2], f[i][0], f[i][3]])
        f = f_tmp

    if v.shape[1] == 2:
        v = np.append(v, np.zeros([v.shape[0], 1]), 1)


    # Type adjustment vertices
    v = v.astype("float32", copy=False)

    # Color setup
    colors, coloring = self._Viewer__get_colors(v, f, c, sh) # "meshplot.Viewer.__get_colors(..)" to access a private member of "Viewer" class

    # Type adjustment faces and colors
    c = colors.astype("float32", copy=False)

    # Material and geometry setup
    ba_dict = {"color": p3s.BufferAttribute(c)}
    if coloring == "FaceColors":
        verts = np.zeros((f.shape[0]*3, 3), dtype="float32")
        for ii in range(f.shape[0]):
            #print(ii*3, f[ii])
            verts[ii*3] = v[f[ii,0]]
            verts[ii*3+1] = v[f[ii,1]]
            verts[ii*3+2] = v[f[ii,2]]
        v = verts
    else:
        f = f.astype("uint32", copy=False).ravel()
        ba_dict["index"] = p3s.BufferAttribute(f, normalized=False)

    ba_dict["position"] = p3s.BufferAttribute(v, normalized=False)

    if uv is not None:
        uv = (uv - np.min(uv)) / (np.max(uv) - np.min(uv))
        # tex = p3s.DataTexture(data=texture_data, format="RGBFormat", type="FloatType")
        material = p3s.MeshStandardMaterial(map=texture_data, reflectivity=sh["reflectivity"], side=sh["side"],
                roughness=sh["roughness"], metalness=sh["metalness"], flatShading=sh["flat"],
                polygonOffset=True, polygonOffsetFactor= 1, polygonOffsetUnits=5)
        ba_dict["uv"] = p3s.BufferAttribute(uv.astype("float32", copy=False))
    else:
        material = p3s.MeshStandardMaterial(vertexColors=coloring, reflectivity=sh["reflectivity"],
                    side=sh["side"], roughness=sh["roughness"], metalness=sh["metalness"], 
                                            opacity=opacity, transparent=True,alphaTest=opacity*0.99,
                                            blending='CustomBlending',depthWrite=False,
                    flatShading=True)

    if type(n) != type(None) and coloring == "VertexColors":
        ba_dict["normal"] = p3s.BufferAttribute(n.astype("float32", copy=False), normalized=True)

    geometry = p3s.BufferGeometry(attributes=ba_dict)

    if coloring == "VertexColors" and type(n) == type(None):
        geometry.exec_three_obj_method('computeVertexNormals')
    elif coloring == "FaceColors" and type(n) == type(None):
        geometry.exec_three_obj_method('computeFaceNormals')

    # Mesh setup
    mesh = p3s.Mesh(geometry=geometry, material=material)

    # Wireframe setup
    if sh['wireframe']:
        wf_geometry = p3s.WireframeGeometry(mesh.geometry) # WireframeGeometry
        wf_material = p3s.LineBasicMaterial(color=sh["wire_color"], linewidth=sh["wire_width"],\
                                            opacity=opacity, transparent=True,alphaTest=opacity*0.99 )
        wireframe = p3s.LineSegments(wf_geometry, wf_material)
        mesh.add(wireframe)
        mesh_obj["wireframe"] = wireframe

    # Object setup
    mesh_obj["max"] = np.max(v, axis=0)
    mesh_obj["min"] = np.min(v, axis=0)
    mesh_obj["geometry"] = geometry
    mesh_obj["mesh"] = mesh
    mesh_obj["material"] = material
    mesh_obj["type"] = "Mesh"
    mesh_obj["shading"] = sh
    mesh_obj["coloring"] = coloring

    return self._Viewer__add_object(mesh_obj)



def color4plot(rgb_col,  vert_or_num):
    if isinstance(vert_or_num, np.ndarray):
        num = vert_or_num.shape[0]
    else:
        num = vert_or_num
    return np.repeat(np.array([rgb_col]), num, axis = 0)

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(int(255*r), int(255*g), int(255*b))

#set copy_elements = True if there are meshes visualized more than once inside a table
def MeshList2Table(mesh_list, nrow, ncol, copy_elements=False):
    mesh_table = []
    #mesh_table =  [[] for _ in range(nrow)]
    i =0
    for ri in range(nrow):
        mesh_table.append([[] for _ in range(ncol)])
    for ri in range(nrow):
        for ci in range(ncol):
            if i >= len(mesh_list):
                i += 1
                continue 
            mesh_table[ri][ci] = copy.deepcopy(mesh_list[i]) if  copy_elements else mesh_list[i] 
            i += 1
    return mesh_table 

#convert all auxilary mesh data into numpy arrays of the vertex shape 
def canonical_mesh_list(mesh_list_):
    mesh_list = [copy.deepcopy(el) for el in mesh_list_]
    V  = mesh_list[0]
    #F  = mesh_list[1]
    C  = mesh_list[2]
    nC = len(C) if type(C)==list else C.shape[0]
    if nC != V.shape[0]:
        C = color4plot(C,V)
        mesh_list[2] = C

    if len(mesh_list) >3 : #point data
        P  = mesh_list[3]
        if is_vertex_index_list(P):
            P = V[P,:]
            mesh_list[3] = P 

        cP = mesh_list[4]
        ncP = len(cP) if type(cP)==list else cP.shape[0]
        if ncP != P.shape[0]:
            cP = color4plot(cP,P)
            mesh_list[4] = cP
    return mesh_list

#Combine two meshes with plot data  into a single mesh with a plot data (for running 'DrawMeshTable' with two meshes in a single cell)
def StackMeshes(meshLists_):
    meshLists = [canonical_mesh_list(mesh_list) for mesh_list in meshLists_]
    nparam0 = len(meshLists[0])
    nparam1 = len(meshLists[1])
    V0 = meshLists[0][0]
    F0 = meshLists[0][1]
    C0 = meshLists[0][2]

    V1  = meshLists[1][0]
    F1  = meshLists[1][1]
    C1  = meshLists[1][2]

    V = np.row_stack((V0,V1))
    F = np.row_stack((F0,F1+V0.shape[0]))
    C = np.row_stack((C0,C1))
    mesh_stack = [V,F,C]

    if nparam0 > 3 or nparam1 > 3: #stack point data 
        if nparam0 > 3: #stack point data 
            P0  = meshLists[0][3]
            cP0 = meshLists[0][4]
            P  = P0
            cP = cP0
        if nparam1 > 3:
            P1  = meshLists[1][3]
            cP1 = meshLists[1][4]
            P  = P1
            cP = cP1
        if nparam0 > 3 and  nparam1 > 3:
            P = np.row_stack ( (P0,  P1) )
            cP = np.row_stack( (cP0,cP1) )
        mesh_stack.append(P)
        mesh_stack.append(cP)
  
    return mesh_stack

def DrawMeshTable(plot_handle, mesh_table, col_shift=1.1, row_shift=1.1, \
                  mesh_shading=dict(wireframe=True, line_color= [0.1,0.1,0.1] ),\
                  point_shading=dict(point_size= 0.05) ):
    nrow = len(mesh_table)
    ncol = len(mesh_table[0])
    col_cent = np.zeros(ncol) 
    col_width = np.zeros(ncol) 
    row_cent = np.zeros(nrow) 
    row_height = np.zeros(nrow) 
    

    bbox  = np.zeros((nrow,ncol,3,2))
    for ri in range(nrow):
        for ci in range(ncol):
            curr_mesh = mesh_table[ri][ci]
            if not curr_mesh:
                continue 
            V = curr_mesh[0]
            bbox[ri,ci,:,0] = V.min(axis=0)
            bbox[ri,ci,:,1]  = V.max(axis=0)
            V -= 0.5*(bbox[ri,ci,:,0] + bbox[ri,ci,:,1]) 
            if len(curr_mesh) > 3 and not( is_vertex_index_list(curr_mesh[3]) ): #3D point data
                P = curr_mesh[3]
                P -= 0.5*(bbox[ri,ci,:,0] + bbox[ri,ci,:,1]) 

    for ri in range(nrow):
        row_cent[ri]   = 0.5*(bbox[ri,:,1,0] + bbox[ri,:,1,1]).max()
        row_height[ri] =     (bbox[ri,:,1,1] - bbox[ri,:,1,0]).max()

    for ci in range(ncol):
        col_cent[ci]   = 0.5*(bbox[:,ci,0,0]  + bbox[:,ci,0,1]).max()
        col_width[ci]  =     (bbox[:,ci,0,1]  - bbox[:,ci,0,0]).max()

    col_left = np.cumsum(col_width  * col_shift)
    col_left = np.concatenate(([0],col_left))

    row_top  = np.cumsum(row_height * row_shift)
    row_top = np.concatenate(([0],row_top))

    mesh_list = []
    for ri in range(nrow):
        for ci in range(ncol):
            if not mesh_table[ri][ci]:
                continue 
            curr_mesh = mesh_table[ri][ci]
            V = curr_mesh[0]
            V[:,0] += col_left[ci]
            V[:,1] += row_top[ri]
            if len(curr_mesh) > 3 and not( is_vertex_index_list(curr_mesh[3]) ): #3D point data
                P = curr_mesh[3]
                P[:,0] += col_left[ci]
                P[:,1] += row_top[ri]
            mesh_list.append(curr_mesh)

    return DrawMeshes(plot_handle,mesh_list,(0,0,0),mesh_shading,point_shading)


def is_vertex_index_list(L):
    return (type(L) is list) or  (L.dtype == int and (len(L.shape) < 2 or L.shape[1] == 1))

def edge_path(P):
    nP =P.shape[0] 
    edge_indices = [ [i, i+1] for i in range(nP-1) ]
    return np.array(edge_indices,dtype=np.int)

def DrawMeshes(plot_handle, mesh_list, shifts=(1.4,0,0), mesh_shading=dict(wireframe=True, line_color= [0.1,0.1,0.1] ), \
               point_shading=dict(point_size= 0.05) ):
    if not type(shifts) is list:
        shifts = [shifts] * (len(mesh_list) -1)
    total_shift = np.array([0.0, 0.0, 0.0])
    plot_handle_ = [plot_handle]
    connect_points = 'connect_points' in  point_shading.keys() and point_shading['connect_points'] 
    for i in range(len(mesh_list)):
        V = mesh_list[i][0] 
        #print(f'total_shift={total_shift}, V_bbox = {V.max(0)- V.min(0)}')
        #margin = (V.max(0)- V.min(0)) * shifts[i]
        #V =  V + total_shift
        #total_shift = (V.max(0)- V.min(0)) * shifts[i]
        

        F = mesh_list[i][1]
        C = mesh_list[i][2]

        #total_shift = (V.max(0)- V.min(0)) * shifts[i]
        if (type(C) is list) or (C.shape[0] != V.shape[0]):
           C = color4plot(C,V)
        
        is_transparent = False
        alpha = 1.0
        if C.shape[1] ==4:
            alpha = C[0,3]
            C = C[:, 0:3]
            if alpha < 1:
                is_transparent = True

        if type(mesh_shading) is dict:
            current_mesh_shading = mesh_shading #common shaidng for all meshes 
        else:
            current_mesh_shading = mesh_shading[i] #separate shading dictionaries for each meah 

        if plot_handle_[0] is None:
            plot_handle_[0]  = empty_meshplot(current_mesh_shading)#mp.Viewer(mesh_shading)
        
        if is_transparent:
            add_transparent_mesh(plot_handle_[0]  , V+total_shift, F, c=C,  shading=current_mesh_shading, opacity=alpha )
        else: 
            plot_handle_[0].add_mesh(V+total_shift ,F,C, shading = current_mesh_shading)
      
        #draw highlighted points 
        if len(mesh_list[i]) > 3:
            #if (type(mesh_list[i][3]) is list) or  (  mesh_list[i][3].dtype == int and (len(mesh_list[i][3].shape) < 2 or mesh_list[i][3].shape[1] == 1)):
            if is_vertex_index_list(mesh_list[i][3]):
                P = V[mesh_list[i][3],: ] #convert vertex indices to 3D points 
            else:
                P = mesh_list[i][3]
            Cp =  mesh_list[i][4]
            if (type(Cp) is list) or (Cp.shape[0] != P.shape[0]):
                Cp = color4plot(Cp,P)
            plot_handle_[0].add_points(P+total_shift, c = Cp, shading = point_shading)           
            if connect_points: 
                plot_handle_[0].add_edges(P+total_shift, edge_path(P),  shading=point_shading)           

        if i < len(mesh_list)-1:
            total_shift += (V.max(0)- V.min(0)) * shifts[i]
    return plot_handle_[0]

def draw_bboxes(p, vert):
    v_box, v_box_half, f_box = get_bboxes(vert)
    
    id1 = p.add_edges(v_box, f_box, shading={"line_color": "black"})
    id2 = p.add_edges(v_box_half, f_box, shading={"line_color": "black",'line_width':0.3})
    id3 = p.add_edges(v_box, np.array([[0, 1]],dtype=np.int), shading={"line_color": "red",'line_width':1})
    id4 = p.add_edges(v_box, np.array([[0, 3]],dtype=np.int), shading={"line_color": "green",'line_width':1})
    id5 = p.add_edges(v_box, np.array([[0, 4]],dtype=np.int), shading={"line_color": "blue",'line_width':1})
    
    return [id1,id2,id3,id4,id5]

sphere_plot_ids = []
def visualize_sphere(p, V, selected_indices, selected_col):
    res = 0
    #print(f'(V,F): {(V.shape,F.shape)}')
    for i in sphere_data.keys():
        #print((sphere_data[i]['V'].shape, sphere_data[i]['F'].shape))
        if sphere_data[i]['V'].shape[0] ==V.shape[0] and sphere_data[i]['F'].shape[0] == F.shape[0] and  not np.any(sphere_data[i]['F'] - F):
            print(f"resolution level={i}")
            res = i
            break

    S = sphere_data[res]
    sphere_size = S['V'][:,0].max() - S['V'][:,0].min()
    SV = (S['V']/sphere_size)*shape_scale + np.array( [V[:,0].max() + shape_scale, 0, 0]) #normlaize to mesh size and shift left 
    if reflect_sphere_y:
        SV[:,1] = -SV[:,1]
    sphere_plot_ids.append(p.add_mesh(SV,S['F'], shading = shading_dic))
    #selected_vert = np.stack( (SV[vr_max,:],SV[vl_max,:]), axis =1 ).transpose()
    selected_vert = SV[selected_indices,:]
    sphere_plot_ids.append( p.add_points(selected_vert, c =selected_col, shading={"point_size": 0.15}) )
    sphere_plot_ids.extend(draw_bboxes(p, SV))
    return sphere_plot_ids

def get_bboxes(vert):
    m = np.min(vert, axis=0)
    ma = np.max(vert, axis=0)
    ma_half = np.max(vert, axis=0)
    ma_half[0] = (m[0]+ma_half[0])/2.0

    # Corners of the bounding box
    v_box = np.array([[m[0], m[1], m[2]], [ma[0], m[1], m[2]], [ma[0], ma[1], m[2]], [m[0], ma[1], m[2]],
                    [m[0], m[1], ma[2]], [ma[0], m[1], ma[2]], [ma[0], ma[1], ma[2]], [m[0], ma[1], ma[2]]])


    v_box_half = np.array([[m[0], m[1], m[2]], [ma_half[0], m[1], m[2]], [ma_half[0], ma_half[1], m[2]], [m[0], ma_half[1], m[2]],
                    [m[0], m[1], ma_half[2]], [ma_half[0], m[1], ma_half[2]], [ma_half[0], ma_half[1], ma_half[2]], [m[0], ma_half[1], ma_half[2]]])                  

    # Edges of the bounding box
    f_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], 
                      [7, 4], [0, 4], [1, 5], [2, 6], [7, 3]], dtype=np.int)
                      
    return v_box, v_box_half, f_box

#help functions
def reflectX(xyz):
    if len(xyz.shape)==2:
        return  np.concatenate( (-xyz[:,0:1], xyz[:,1:3]), axis=1) 
    else:
        return  np.concatenate( (-xyz[0:1], xyz[1:3]) ) 

def Out(str, size = 3):
    display(Markdown(f'{"#"*size} {str}'))



def get_fitted_sphere(V, sphere_data):
    res = 0
    #print(f'(V,F): {(V.shape,F.shape)}')
    for i in sphere_data.keys():
        #print((sphere_data[i]['V'].shape, sphere_data[i]['F'].shape))
        if sphere_data[i]['V'].shape[0] ==V.shape[0] and sphere_data[i]['F'].shape[0] == F.shape[0] and  not np.any(sphere_data[i]['F'] - F):
            print(f"resolution level={i}")
            res = i
            break

    S = sphere_data[res]
    sphere_size = S['V'][:,0].max() - S['V'][:,0].min()
    shape_scale = V[:,0].max() - V[:,0].min()
    #SV = (S['V']/sphere_size)*shape_scale + np.array( [V[:,0].max() + shape_scale, 0, 0]) #normlaize to mesh size and shift left 
    SV = (S['V']/sphere_size)*shape_scale
    SF = S['F'] #done change it !
    #if reflect_sphere_y:
    if sphere_data['reflect_y']:
        SV[:,1] = -SV[:,1]
    return SV, SF #SV=copy, SF=original data- do not change it!


def visualize_symmetry_details(V,F, symmetry_data, sphere_data):
    #LR_max    = symmetry_data['LR_max']
    V_right    = symmetry_data['V_right']
    V_opposite = symmetry_data['V_opposite']
    #V_right   = symmetry_data['V_left']
    V_left     = V_opposite[V_right]
    V_center   = np.nonzero(np.arange(0, V.shape[0]) == V_opposite)[0]
    print(V_center)

    SV, SF = get_fitted_sphere(V, sphere_data)

    V_l2r =  np.copy(V)
    V_l2r -= np.mean(V_l2r,axis=0) #align shape to 0,0,0  
    V_l2r[V_right,:] = reflectX(V[V_opposite[V_right],:])
    V_l2r_bbox     = V_l2r.max(0)- V_l2r.min(0)

    V_l2r_cent = np.copy(V_l2r)
    V_l2r_cent -= np.mean(V_l2r_cent,axis=0) #align shape to 0,0,0  
    V_l2r_cent[V_center,0] = 0


    V_r2l =  np.copy(V)
    V_l2r -= np.mean(V_r2l,axis=0) #align shape to 0,0,0  
    V_r2l[V_left,:] =   gu.reflectX(V[V_opposite[V_left],:] )
    V_r2l_bbox     = V_r2l.max(0)- V_r2l.min(0)

    col = color4plot([0,0,1],V)
    r = [1,0,0]
    col[V_left] = np.array([0,1,0])
    ph = DrawMeshes(None,  [ [SV, SF, col, V_center,r],  [V,F,col, V_center,r], \
                           [V_l2r,F,[0,1,0], V_center,r], [V_l2r_cent,F,[0,1,0], V_center,r], \
                           [V_r2l,F,[0,0,1], V_center,r ] ] \
                    )
    return ph

"""
cmap =  matplotlib.colors.Colormap or its name, nband = number of colors in the cmap -> numpay array of RGA colors in [0,1] range
use it to convert  matplotlib colormaps to meshplot colors (based on https://gist.github.com/salotz/4f585aac1adb6b14305c)
"""
def cmap_to_numpy(cmap, nband, add_alpha=False):
    if type(cmap) is str:
       cmap = plt.get_cmap(cmap)
       h = .5 / nband
       if add_alpha:
            return cmap( np.linspace( h, 1 - h, nband ))
       else:
           return cmap( np.linspace( h, 1 - h, nband ))[:,0:3]

def weighted_color_sum(colors, weights):
    col_arr = np.array(colors) if type(colors)==list else colors
    if type (weights) in [float, int]:
        weights = [weights] * col_arr.shape[0]
    w_arr = np.array(weights) if type(weights)==list else weights
    return w_arr @ col_arr 


           
#class to simplify mesh colorization for meshplot package (see usage examples in "notebook/cad_mesh_analysis.ipynb")
class ColorMapGradient:
    def __init__(self, rgb_list,nbin=10):
        self.color_gradinet(rgb_list,nbin)

    def color_gradinet(self, rgb_list,nbin):
        self.nbins = nbin if type(nbin) == list else [nbin]*len(rgb_list)
        col_segments = []
        for i in range(len(rgb_list)-1): 
            col_segments.append( np.linspace(rgb_list[i], rgb_list[i+1], self.nbins[i]) )
        
        self.color_array = np.concatenate(col_segments) 
        self.nbins = self.color_array.shape[0]-1
        return self.color_array, self.nbins
    
    #array_values = list of 1D arrays of the same length  of matrix of these 1D  row arrays
    def set_bins(self, array_list):
        #print(f'is array_list list or tuple = {type(array_list) in [list, tuple]}')
        array_mat = np.concatenate(array_list) if type(array_list) in [list, tuple]  else  array_list
        self.bins = np.linspace( array_mat.min(),  array_mat.max(),self.nbins)
    
    def colorize(self, array_list, common_range=True):
        color_list = []
        for arr in array_list:
            #print(f'arr.shape={arr.shape}, bin.shape={self.bins.shape}')
            if not common_range:
                self.set_bins(arr)
            color_list.append( self.color_array[np.digitize(arr,self.bins),:] )
        return color_list
    
    def draw_color_bar(self, label="", title="", orientation="vertical"): 
        color =  LinearSegmentedColormap.from_list(title, self.color_array.tolist(), self.nbins+1)
        f1, ax1 = plt.subplots()
        plt.axis('off')
        s = ax1.scatter(x=np.zeros( self.nbins), y=np.zeros( self.nbins), c=100*self.bins, cmap=color)
        f1.colorbar(s, label=label, orientation=orientation, ax = ax1)
# # Test open3d and meshplot visulization utils 
# if __name__ == "__main__":  
#     import igl 
#     import open3d as o3d
#     from scipy.spatial.transform import Rotation as R
#     import copy

#     # mp.Viewer.add_edges
#     # mp.Viewer.add_mesh
#     #sett = {"width": 1000, "height": 600, "antialias": True, "scale": 1.5, "background": "#ffffff","fov": 30}
#     #p = mp.Viewer(sett)
#     #mp.Viewer.add_text(p,"text")
#     obj_file = '/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD_vox_tri/car/car_06_clean_manual_symm_cub_tri.obj'
#     import igl 
#     Vobj, Fobj = igl.read_triangle_mesh(obj_file)
#     mesh_shading=dict(wireframe=True, line_color= [0.1,0.1,0.1] )
#     mp.plot(Vobj,Fobj,  color4plot([1,1,0], Vobj), shading = mesh_shading)
#     print('main script')


#     def draw_registration_result(source, target, transformation):
#         source_temp = copy.deepcopy(source)
#         target_temp = copy.deepcopy(target)
#         source_temp.paint_uniform_color([1, 0.706, 0])
#         target_temp.paint_uniform_color([0, 0.651, 0.929])
#         source_temp.transform(transformation)
#         o3d.visualization.draw_geometries([source_temp, target_temp])
#     V1, F1 = igl.read_triangle_mesh('/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD/car/01.off')
#     V2, F2 = igl.read_triangle_mesh('/home/ubuntu/research/datasets/Pascal3D+_release1.1/CAD/car/02.off')
#     Rmat = (R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])).as_matrix()
#     V2rot = np.matmul(V2,Rmat)
#     source = o3d.geometry.PointCloud()
#     source.points  = o3d.utility.Vector3dVector(V1)
#     target = o3d.geometry.PointCloud()
#     target.points =  o3d.utility.Vector3dVector(V2rot)
#     trans_init  = np.eye(4)
#     threshold = 0.02
#     reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,\
#                                                 o3d.pipelines.registration.TransformationEstimationPointToPoint() )
#     source.transform(reg_p2p.transformation)
