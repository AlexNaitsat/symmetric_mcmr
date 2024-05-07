from sys import exec_prefix
import torch
import numpy as np
#from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import os
from tabulate import tabulate
import copy
import pandas as pd

#############################################################################
#               Textual  utils  
#############################################################################
def output_network_dict(D, prefix=''):
    if prefix =='':
        print(f'{prefix}{type(D)}')
    prefix = prefix + '  '
    for name in D.keys():
        D_value = D[name]
        if isinstance(D_value, list):
            print(f'{prefix}[{name}]  :  {type(D_value)}')
            output_tensor_list(D_value, prefix)
        elif isinstance(D_value, dict):
            print(f'{prefix}[{name}]  :  {type(D_value)}')
            output_network_dict(D_value, prefix)
        elif hasattr(D_value, 'shape'):
            print(f'{prefix}[{name}]  : {D_value.shape}')
        else:
            print(f'{prefix}[{name}]  :  {type(D_value)}')

def output_tensor_list(L, prefix=''):
    if prefix == '':
        print(f'{prefix}{type(L)}')
    prefix = prefix + '     '
    for indx in range(0, len(L)):
        if isinstance(L[indx], list):
            print(f'{prefix}[{indx}]  :  {type(L[indx])}')
            output_tensor_list(L[indx], prefix)
        elif isinstance(L[indx], dict):
            print(f'{prefix}[{indx}]  :  {type(L[indx])}')
            output_network_dict(L[indx], prefix)
        elif  hasattr(L[indx], 'shape'):
            print(f'{prefix}[{indx}]  :  {L[indx].shape}')
        else:
            print(f'{prefix}[{indx}]  :  {type(L[indx])}')

def string_to_path(file_name, folder, ext):
    if not os.path.split(file_name)[0]:
        file_name  = os.path.join(folder, file_name)
    if not os.path.splitext(file_name)[-1]:
        file_name  =  file_name + '.' + ext
    return file_name


#############################################################################
#               Image   utils  
#############################################################################
def tensor_to_batch(T, prefix, to_normalize=False):
    import cv2 as cv
    if torch.is_tensor(T):
        T = T.detach().cpu().numpy()
    if to_normalize:
        maxT =np.max(T)
        if maxT != 0: T= T/maxT
    
    if len(T.shape)==3: #batchless image 
        T = np.expand_dims(T, 0)

    batch_n = T.shape[0]
    for b in range(0,batch_n):
        image_array = (np.transpose(T[b,:,:,:], [1, 2, 0]) if  T.shape[1] <= 3 else T[b,:,:,:]) #C*H*W=> H*W*C
        if image_array.shape[2] ==2: #B color  channel for 2-chanlle images (e.g uv map)
            image_array =  np.concatenate( (image_array, image_array[:,:,0][:,:,np.newaxis]), axis= 2)

        if  np.max(image_array) < 1.0001:
            image_array = image_array.astype('float32')*255 #convert to RGB range
        image_file_name = './debug/' + prefix + '_batch_' + str(b) +'.png'
        print('writing ' + image_file_name)
        cv.imwrite(image_file_name, image_array)

def draw_multiview_tensor_with_batchs(Tlist,  prefix, to_normalize=False, scale=1):
    import cv2 as cv
    view_num = len(Tlist)
    batch_n = Tlist[0].shape[0]

    if len(Tlist[0].shape)<4:
        Tlist = [T.unsqueeze(1) for T in Tlist]

    H  = Tlist[0].shape[2]
    W  = Tlist[0].shape[3]
    mv_images_per_batch = [np.zeros((H,W*view_num,3)).astype('float32')]*batch_n
    for b in range(0, batch_n):
        for v in range(0, view_num):
            T = Tlist[v]
            if torch.is_tensor(T):
                T = T.detach().cpu().numpy()
                image_array = np.transpose(T[b,:,:,:], [1, 2, 0])
                if to_normalize:
                    m = np.min(image_array)
                    image_array = (image_array -m)/(np.max(image_array) - m)
                else:
                    image_array = image_array*scale
                if  np.max(image_array) < 1.0001:
                    image_array = image_array.astype('float32')*255 #convert to RGB range
                mv_images_per_batch[b][:, W*v:W*(v+1),:] = image_array
        image_file_name = './debug/' + prefix + '_batch_' + str(b) + '.png'
        print('writing ' + image_file_name)
        cv.imwrite(image_file_name, mv_images_per_batch[b])



#Added some visualization utils from DenseMatching
def mask2RGB(mask_orig, rgb):
    if len(mask_orig.shape) == 2: 
        mask  = mask_orig[:,:,np.newaxis]
    else:
         mask = mask_orig[:,:,0]
    return  mask * rgb

def torch_image_to_numpy(torch_image):
    image = torch_image.detach().cpu().numpy()
    if len(torch_image.shape) ==  4:
        image = image[0,:] #remove batch dimension
    if len(torch_image.shape) ==  2:
        image = image[:,:,np.newaxis]
    if image.shape[0] <= 3:  #move color dimension to the end
        image = np.transpose(image, (1,2,0))
    
    if image.shape[2] ==2: #add B color for 2-channel images (e.g UV maps)
        image =  np.concatenate( (image, image[:,:,0][:,:,np.newaxis]), axis= 2)
    return image
        


def horizontal_pictures_with_title(file_name, image_dic, sup_title=None, size=None, font_size = 24, batch = 0):
    
    file_name =  string_to_path(file_name,'./debug','png') #complete shortened file paths

    image_num = len(image_dic.keys())
    fig, axis = plt.subplots(1, image_num, figsize = (30,30))
    i =0
    for name, images in image_dic.items():
        if batch is None:
            image  = images
        else:
            image = images[batch]
        if torch.is_tensor(image):
            image = torch_image_to_numpy(image)
        axis[i].imshow(image)
        axis[i].set_title(name, fontsize=font_size)
        i = i + 1


    if not sup_title is None:
        fig.suptitle(sup_title, fontsize=font_size + 4)

    if size is None:
       size =[10. *image_num, 8.]
    fig.set_size_inches(size)

    fig.savefig(file_name,bbox_inches='tight')
    print('writing ' + file_name)




def picture_table_with_title(file_name, image_dic_list, sup_title=None, size=None, font_size = 24, batch = 0,fig = None, axis = None):

    #file_name =  string_to_path(file_name,'./debug','png') #complete shortened file paths

    row_num = len(image_dic_list)
    col_num = len(image_dic_list[0].keys())

    if fig is None or axis is None:
        fig, axis = plt.subplots(row_num, col_num, figsize = (30,30))
    r =0
    for image_dic in image_dic_list:
        i =0
        for name, images in image_dic.items():
            #print(name + ', r=' + str(r))
            if images is None:
                continue
            elif type(images) == str:
                images = plt.imread(images)
            if batch is None:
                image  = images
            else:
                image = images[batch]
            if torch.is_tensor(image):
                image = torch_image_to_numpy(image)
            axis[r, i].imshow(image)
            axis[r, i].set_title(name, fontsize=font_size)
            i = i + 1

        if not sup_title is None:
            fig.suptitle(sup_title, fontsize=font_size + 4)
        r =r+1
    if size is None:
        size =[10. *col_num, 10.*row_num]
        fig.set_size_inches(size)

    fig.tight_layout()
    fig.savefig(file_name,bbox_inches='tight')
    print('writing ' + file_name)

def normalize_image(image):
    if len(image.shape) > 3:
        norm_image =  np.zeros(image.shape)
        for b in range(0, image.shape[0]):
            norm_image[b,:] = normalize_bachless_image(image[b,:])
        return norm_image
    else:
        return normalize_bachless_image(image)

    
def normalize_bachless_image(image):
    if torch.is_tensor(image):
        image = torch_image_to_numpy(image)
    m = np.min(image)
    M = np.max(image)
    return  (image - m)/(M-m) 

def normalize_torch_image(image,batch_dim = None):
    if batch_dim is None:
        m = torch.min(image).detach()
        M = torch.max(image).detach()
        return  (image - m)/(M-m)

########################################################################
# image/ tensor utils related to Pytorch 3D
#######################################################################

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
    png_file = None
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
    
    if png_file:
       fig.savefig(png_file)
       print(f'Saved as {png_file}')



########################################################################
#                Tensor utils 
########################################################################
def shift_tensor(T, xy_shifts): #T = B*C*H*W
    if xy_shifts[1] == 0:
        return torch.roll(T, shifts=xy_shifts[0] , dims=-2)
    elif xy_shifts[0] == 0:
        return torch.roll(T, shifts=xy_shifts[1] , dims=-1)
    else:
        return torch.roll(torch.roll(T, shifts=xy_shifts[0] , dims=-2), shifts=xy_shifts[1] , dims=-1)


def add_zeros_to_tensor(input, axis, trailing_zeros=1, leading_zeros=0, normalize = False):
    pad_flag = 2*[0]*len(input.shape)
    pad_flag[2*axis+1] = leading_zeros #even indices for leading zeros instead of odd to match .reversed order in "pad"
    pad_flag[2*axis]   = trailing_zeros
    pad_flag.reverse()

    if normalize:
        return torch.nn.functional.normalize(torch.nn.functional.pad(input, pad_flag), dim = axis)
    else:
        return torch.nn.functional.pad(input, pad_flag)


############################################
#    mat loader
############################################


# based on https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
import scipy.io as spio
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

## string, prints and tables
def out(str, str_list = None): 
    print(str)
    if not str_list is None:
        str_list.append(str)

def dict2table(table_dict, caption = None, columns= None):
    table_list = [v.tolist() if isinstance(v,np.ndarray)  else v for k, v in table_dict.items()]
    if not columns is None:
         table_list.insert(0, columns)
    
    table = tabulate(table_list)
    if not caption is None:
        print(f'{caption}\n{table} ')

    return table_list, table 

#### Reports
class MetricReport:
    #def __init__(self, metric_names_or_dict, path2name_func = lambda path: path):
    def __init__(self, metric_names_or_dict):
        if type(metric_names_or_dict)==list: #intializing from scratch with given list of metric names 
            metric_names = metric_names_or_dict
            self.metric_names = metric_names 
            self.metric_num = len(metric_names)
            self.result_dict = {}
            self.table_rows = [['test'] + self.metric_names]
        elif type(metric_names_or_dict)==dict: #intializing with loaded dictionary 
            self.result_dict = metric_names_or_dict
            first_test = list(self.result_dict.keys())[0]
            self.metric_names = list(self.result_dict[first_test].keys())
            self.metric_num = len(self.metric_names)
            self.recompute_table()
        else:
            raise Exception(f'Unsupported type  {type} for 1st argument of  MetricReport  initialization')
        self.metrics_list = []
        self.metrics_dict = {}
        #self.path2name_func  = path2name_func        

    def add_test(self, test_name, metrics_np):
    # def add_test(self, test_path, metrics_np):
    #     test_name = self.path2name_func(test_path)
        if not test_name in self.result_dict:
            self.result_dict[test_name] = dict(zip(self.metric_names,[None]*self.metric_num))
        mean_metrics =  metrics_np.mean(axis=0).tolist()
        self.metrics_list.append(metrics_np) #Fo I need it ??
        self.metrics_dict[test_name] = metrics_np

        for i in range(self.metric_num):
            self.result_dict[test_name][self.metric_names[i]] = mean_metrics[i]
        self.table_rows.append( [test_name] + mean_metrics )
        


    def recompute_table(self):
        first_rows  =  [['test'] + self.metric_names]
        other_rows = [ [test] + list(self.result_dict[test].values()) for test in self.result_dict.keys()]
        self.table_rows =  first_rows + other_rows
        return self.table_rows 

    def get_table(self, caption=None):
        table_str= tabulate(self.table_rows) 
        if not caption is None:
            table_str = caption + '\n' + table_str
            print(table_str)
        return table_str, self.table_rows,   np.array(self.table_rows)

    def get_pandas_table(self, caption=None):
        table_pd = self.DataFrame()
        if not caption is None:
            print(caption)
            print(table_pd)
        return table_pd, self.table_rows,   np.array(self.table_rows)    
   
    def DataFrame(self):
        return pd.DataFrame(self.table_rows[1:], columns =  self.table_rows[0])

    #here mean improvements is: (mean(source)- mean(target))/mean(source)
    #but it should be         : mean((source- target)/source )
    def compare_means(self, reportTarget, caption = None, in_percents =  True):
        reportImprov = copy.deepcopy(reportTarget)
        multiplier   = (100 if in_percents else 1)
        for t in self.result_dict.keys():
            for m in self.metric_names:
                target_metric = reportTarget.result_dict[t][m]
                source_metric = self.result_dict[t][m]
                #print(f'{(t,m)}=>{(source_metric,target_metric)}')
                reportImprov.result_dict[t][m] = ((source_metric - target_metric)/source_metric) * multiplier

        reportImprov.recompute_table()
        return reportImprov.get_table(caption)

    def compare(self, reportTarget, caption = None, in_percents =  True, as_pandas = True):
        #reportImprov = copy.deepcopy(reportTarget)
        reportImprov = MetricReport(self.metric_names)
        multiplier   = (100 if in_percents else 1)
        for t in self.result_dict.keys():
                target_metric_arr = reportTarget.metrics_dict[t]
                source_metric_arr = self.metrics_dict[t]
                reportImprov.add_test(t,  multiplier *(source_metric_arr - target_metric_arr)/source_metric_arr)
        return  reportImprov.get_pandas_table(caption)  if  as_pandas else reportImprov.get_table(caption) 

##################################################
# List utils 
##################################################
def is_iterable(object):
    try:
        iter(object)
        return True 
    except TypeError:
        return False


# Merges sublists/tuple/arrays, 
# e.g for inputs  [ 1, [2,3]] ,  [1, (2,3)] ,  [1, np.array([2,3])]  return [1,2,3]
def flatten_list(L):
    if  not  is_iterable(L):
        return L
    else:
        Lf =[]
        for i in L:
            if is_iterable(i):
                [Lf.append(j) for j in i]
            else:
                Lf.append(i)
        return Lf

#list wrapper class to allow indexing with lists and numpy arrayes
# E.g.
#  L =List(['a', 'b', 'c','d']) 
#  print(  ( L[[0,1,2]]), L[np.array([0,1,2])]) ) )
class List(list):
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)): return list.__getitem__(self, keys)
        if isinstance(keys,np.ndarray):
            keys = keys.tolist()
        return [self[k] for k in keys]

################################
## Class list utils 
def dict_to_table(d, verbose = True):
    str = print("{:<8} {:<15}".format('Label','Number'))

    #print("{:<8} {:<15} {:<10}".format('Key','Label','Number'))
    for k, v in d.iteritems():
        label, num = v
        #print("{:<8} {:<15} {:<10}".format(k, label, num))
        str += '\n' + "{:<8} {:<15}".format(label, num)
        #print("{:<8} {:<15}".format(label, num))
    if verbose:
        print(str)
    return str


def print_class_attributes(C, verbose=True):
    return  dict_to_table(C.__dict__,verbose)
