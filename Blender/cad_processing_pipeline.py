import bpy
import os 
import glob


def delete_meshes():
    for obj in bpy.data.objects:
        if obj.name != "Light" and  obj.name != "Camera":
            obj.select_set(True)
    bpy.ops.object.delete()

def get_first_mesh():
    for obj in bpy.data.objects:
        if obj.name != "Light" and  obj.name != "Camera":
            return obj.name
            #obj.select_set(True) #doesn't work 
            #print(f'selected {obj.name}')

def activate_first_mesh():
    for obj in bpy.data.objects:
        if obj.name != "Light" and  obj.name != "Camera":
            obj.select_set(True) #doesn't work 
            bpy.context.view_layer.objects.active = obj
            return obj.name

# geometry processing operations 
#def cleanup_trianlgle_mesh(obj_size):
#    print('cleanup_trianlgle_mesh')
#    #bpy.ops.object.mode_set(mode = 'EDIT')
#    #bpy.ops.mesh.delete_loose() #usually do nothing and switch to object mode
#    
#    bpy.ops.object.mode_set(mode = 'EDIT')
#    bpy.ops.mesh.select_all(action='SELECT')
#    bpy.ops.mesh.dissolve_degenerate()
#    bpy.ops.mesh.dissolve_limited() #generate non triangle emshes by mesuring nerboiring faces 
#    bpy.ops.mesh.remove_doubles() #merge by distance 
#    bpy.ops.mesh.normals_make_consistent(inside=False)
#    bpy.ops.mesh.fill_holes()
#    #back to triangle mesh 
#    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    
    
def cleanup_trianlgle_mesh(obj_size):
    print('cleanup_trianlgle_mesh aaa')
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.dissolve_degenerate()
    bpy.ops.mesh.dissolve_limited() #generate non triangle emshes by mesuring nerboiring faces 
    bpy.ops.mesh.dissolve_degenerate()
    bpy.ops.mesh.remove_doubles() #merge by distance 
    bpy.ops.mesh.fill_holes()
    #back to triangle mesh 
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    #bpy.context.tool_settings.mesh_select_mode = (False, True, False)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.editmode_toggle()
    #bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)

def solidify_mesh(obj_size,voxelSize):
    print('solidify_mesh')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.modifier_add(type='SOLIDIFY')
    bpy.context.object.modifiers["Solidify"].thickness = obj_size*voxelSize+eps #=voxel size
    bpy.ops.object.modifier_apply(modifier="Solidify")
                
def voxel_remesh(obj_size,voxelSize):
    print('voxel_remesh')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.modifier_add(type='REMESH')
    bpy.context.object.modifiers["Remesh"].voxel_size = obj_size*voxelSize
    bpy.ops.object.modifier_apply(modifier="Remesh")
    
    
def decimate_mesh(obj_size, decimateRatio, minFaceNum = None):
    if not (minFaceNum is None):
        face_num = len(bpy.context.selected_objects[0].data.polygons)
        if face_num < minFaceNum:
            print(f'Skip decimatation because face num < {minFaceNum}')
            return 
        elif face_num*decimateRatio < minFaceNum:
             decimateRatio = minFaceNum/face_num  
             print(f'ratio reset to {decimateRatio} to get {minFaceNum} faces' )
            
            
    print('decimate_mesh')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.modifier_add(type='DECIMATE')
    bpy.context.object.modifiers["Decimate"].ratio = decimateRatio
    bpy.ops.object.modifier_apply(modifier="Decimate")


def smooth_mesh(obj_size, coeff=0.5):
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.vertices_smooth(factor=coeff)

      
def triangulate_mesh(obj_size): 
    print('triangulate_mesh')
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    
def symmetrize_mesh(obj_size): 
    bpy.ops.object.mode_set(mode = 'EDIT')
    print('symmetry snap and symmetrization')
    bpy.ops.mesh.symmetry_snap()
    bpy.ops.mesh.symmetrize()


def cubify_mesh(obj_size, octree_depth=7, scale=0.99, remove_disconnected = False):
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.modifier_add(type='REMESH')
    bpy.context.object.modifiers["Remesh"].mode = 'BLOCKS'
    bpy.context.object.modifiers["Remesh"].octree_depth = octree_depth
    bpy.context.object.modifiers["Remesh"].scale = scale
    bpy.context.object.modifiers["Remesh"].use_remove_disconnected = remove_disconnected

    bpy.ops.object.modifier_apply(modifier="Remesh")





def run_pipeline(cad_path, cad_classes, save_path, pipeline):
    for cad_class in cad_classes:
        print("Class " + cad_class)
        #glob.glob("C:\\Users\\anaitsat\\datasets\\CAD_obj\\car\\car_??.obj")
        input_path = os.path.join(cad_path, cad_class)
        out_path   = os.path.join(save_path, cad_class)
        
        obj_files  = glob.glob( os.path.join(input_path, cad_class+"_*??.obj"))
            
        for obj_file in obj_files:
            file_name = os.path.split(obj_file)[-1].split(".")[0]
            print("  cleaning " + file_name)
            delete_meshes()
            #file_name =  'car_01'
            #cad_class = 'car'
            input_path = os.path.join(CAD_path, cad_class)
            print(input_path)
            #load mesh 
            #imported_object = bpy.ops.import_scene.obj(filepath= input_path + file_name +'.obj')
            imported_object = bpy.ops.import_scene.obj(filepath= os.path.join(input_path, file_name +'.obj') )
            obj_name=activate_first_mesh()
            bpy.ops.object.mode_set(mode = 'OBJECT')
            #print(f'select {obj_name}')
            dim_x =  bpy.data.objects[obj_name].dimensions.x
            dim_y =  bpy.data.objects[obj_name].dimensions.y
            dim_z =  bpy.data.objects[obj_name].dimensions.z
            obj_size = max(dim_x,dim_y,dim_z)
            print(f'    object dimensions={(dim_x,dim_y,dim_z)}, size = {obj_size}')
            suffix = ""
            
            for cmd in pipeline:
                func = cmd[0]
                suffix +=  cmd[1]
                to_obj = cmd[2]
                arg = cmd[3:]
                print((func,obj_size,arg))
                func(obj_size, *arg)
                if to_obj:
                    bpy.ops.export_scene.obj(filepath= os.path.join(out_path, file_name + suffix + '.obj') ) #save

    bpy.ops.object.mode_set(mode = 'OBJECT')
    

if __name__ == "__main__":   
    MinFaceNumForDecimate = 1500
    DecimateRatio = 0.1
    eps           = 1e-4
    VoxelSize     = 0.009
    EnableMeshCleanUp = True
    EnableSolidify    = True 
    EnableVoxelRemesh = True 
    EnableDecimate    = True 

#    #============================================================================================
#    #processing original cad models 
#    CAD_path = "C:\\Users\\anaitsat\\datasets\\CAD_obj"
#    #CAD_classes = ['aeroplane','bicycle', 'boat',  'bottle',  'bus',  'car',  'chair',  'diningtable',  'motorbike',  'sofa',  'train',  'tvmonitor']

#    CAD_classes = ['car']
#    Pipeline   =   [ (cleanup_trianlgle_mesh, "_clean", True), \
#                     (solidify_mesh, "_solid", False, VoxelSize),  \
#                     (triangulate_mesh, "", False ),     \
#                     (voxel_remesh, "_voxrem", True,  VoxelSize),   \
#                     (decimate_mesh, "_dec", True,    DecimateRatio, MinFaceNumForDecimate), \
#                     (smooth_mesh, "_smooth", True) \
#                   ]
#    run_pipeline(CAD_path, CAD_classes, Pipeline)
#    
#    # ========================================================================================
    # processing CAD models after previous pipeline + some manual clean-up as follows:
    # symmetrizing meshes 
    # cubification and triangulation
    
    CAD_path = "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed"
    out_path = "C:\\Users\\anaitsat\\datasets\\CAD_preprocessed_out"
    
    CAD_classes = ['car']
    Pipeline   =   [ (symmetrize_mesh, "_symm", True), \
                     (cubify_mesh, "_cub", True, 7, 0.99, False), \
                     (triangulate_mesh, "_tri", True), \
                   ]
    run_pipeline(CAD_path, CAD_classes, out_path, Pipeline)
    

#    CAD_classes = ['bicycle']
#    VoxelSize     = 0.01
#    Pipeline   =   [ (cleanup_trianlgle_mesh, "_clean", True), \
#                     (solidify_mesh, "_solid", False, VoxelSize),  \
#                     (triangulate_mesh, "", False ),     \
#                     (voxel_remesh, "_voxrem", True,  VoxelSize),   \
#                     (decimate_mesh, "_dec", True,    DecimateRatio, 5000), \
#                     (smooth_mesh, "_smooth", True) \
#                   ]

#    run_pipeline(CAD_path, CAD_classes, Pipeline)