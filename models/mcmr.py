import meshzoo
import torch
import torch.nn as nn
import torchvision
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes

from models.deepsdf import DeepSDFDecoder
from models.net_blocks import net_init, conv2d, decoder2d, fc
from models.spade_decoder import SPADEGenerator_noSPADENorm
from utils import mesh
import os
import numpy as np
from utils.geometry import measure_symmetries as gu_measure_symmetries
from utils.geometry import reflectX

class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4, pretrain=True):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrain)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, enc_flat_type='view', batch_norm=True, pretrain=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=n_blocks, pretrain=pretrain)
        self.enc_conv1 = conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        self.out_shape = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)

        # using fc (after forward flattening), avg_pool over spatial features or leave as it is
        self.enc_flat_type = enc_flat_type
        if self.enc_flat_type == 'fc':
            self.enc_feat_flat = fc(batch_norm=True, nc_inp=self.out_shape, nc_out=nz_feat)
            self.out_shape_flat = nz_feat
        elif self.enc_flat_type == 'avg':
            self.enc_feat_flat = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.out_shape_flat = 256
        elif self.enc_flat_type == 'view':
            self.out_shape_flat = self.out_shape
        else:
            raise ValueError

        net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv.forward(img)

        feat = self.enc_conv1(resnet_feat)

        if self.enc_flat_type == 'fc':
            feat_flat = feat.view(img.size(0), -1)
            feat_flat = self.enc_feat_flat(feat_flat)
        elif self.enc_flat_type == 'avg':
            feat_flat = self.enc_feat_flat(feat).squeeze(-1).squeeze(-1)
        else:
            feat_flat = feat.view(img.size(0), -1)

        return feat, feat_flat


class TexturePredictor(nn.Module):
    """
    Outputs mesh texture
    """
    def __init__(self, nc_init=256, n_upconv=6, upconv_mode='bilinear', decoder_name='SPADE', norm_mode='batch'):
        super(TexturePredictor, self).__init__()
        self.decoder_name = decoder_name

        if self.decoder_name == 'SPADE':
            self.decoder = SPADEGenerator_noSPADENorm(nc_init, n_upconv, nc_out=3, predict_flow=False,
                                                      upsampling_mode=upconv_mode, norm_mode=norm_mode)
        elif self.decoder_name == 'upsample':
            self.decoder = decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=3, use_deconv=False,
                                     upconv_mode=upconv_mode)
        elif self.decoder_name == 'deconv':
            self.decoder = decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=3, use_deconv=True,
                                     upconv_mode=upconv_mode)
        else:
            raise ValueError

    def set_uv_sampler(self, verts, faces, tex_size):
        uv_sampler = mesh.compute_uvsampler_softras(verts, faces, tex_size=tex_size)
        uv_sampler = torch.FloatTensor(uv_sampler).cuda()
        self.F = uv_sampler.size(0)
        self.T = uv_sampler.size(1)
        self.uv_sampler = uv_sampler  # F x T x T x 2

    def forward(self, feat):
        uvimage_pred = self.decoder.forward(feat)
        if self.decoder_name != 'SPADE':
            uvimage_pred = torch.tanh(uvimage_pred)
        # #example of how to upload uvimage_pred from image file 
        # import cv2 
        # loaded_unwrapp_np = cv2.cvtColor(cv2.imread('save/car_10_MRCNN_new/car_small-qualitative/unwrapped/car_small_edited_0000_v2.png'), cv2.COLOR_RGB2BGR)/255.0
        # if loaded_unwrapp_np.shape != (256,256):
        #     loaded_unwrapp_np =  cv2.resize(loaded_unwrapp_np, (256,256), interpolation = cv2.INTER_AREA)
        # loaded_unwrapp = torch.from_numpy(loaded_unwrapp_np).permute(2,0,1).unsqueeze(0)
        # import utils.debug_utils as du
        # du.tensor_to_batch(loaded_unwrapp ,'loaded_unwrapp')
        # uvimage_pred.copy_(loaded_unwrapp)
    
        self.uvimage_pred = uvimage_pred

        # B x F' x T x T x 2
        uv_sampler = self.uv_sampler.unsqueeze(0).repeat(feat.shape[0], 1, 1, 1, 1)
        # B x F x T x T x 2 --> B x F x T*T x 2
        uv_sampler = uv_sampler.view(-1, self.F, self.T * self.T, 2)
        tex_pred = torch.nn.functional.grid_sample(uvimage_pred, uv_sampler, align_corners=True)
        tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)
        tex_pred = tex_pred.unsqueeze(4).expand(tex_pred.shape[0], tex_pred.shape[1], tex_pred.shape[2],
                                                tex_pred.shape[3], self.T, tex_pred.shape[4])
        tex_pred = tex_pred[:, :, :, :, 0, :].view(tex_pred.shape[0], tex_pred.shape[1], self.T * self.T, 3)

        return tex_pred.contiguous()


class MCMRNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=256, num_classes=12, texture_type='surface', enable_shape_symmetrization = False):
        super(MCMRNet, self).__init__()

        self.input_shape = input_shape  # input_shape is H x W of the image
        self.nz_feat = nz_feat
        self.num_classes = num_classes
        self.texture_type = texture_type

        self.opts = opts
        self.disable_resnet_pretrain = opts.disable_resnet_pretrain
        self.use_learned_class = opts.use_learned_class

        assert self.texture_type in ('surface', 'vertex')

        self.enable_symmetrization = enable_shape_symmetrization
        if self.enable_symmetrization:
            print( '===============>  Enable Symmetrization of mean shapes and deformations <===============')
            print(f'                  symmetrization type = {self.enable_symmetrization}'  )
        self.align_middle_vertices_in_symmetrization = True 
        #load data for symmetrization files generated by 'notebook/symmetry_computation.ipynd'
        if os.path.exists('./auxilary/icoshpere_meshes.npy'):
            self.symmetry_data =  np.load('./auxilary/icoshpere_meshes.npy', allow_pickle = True).item()
        #support only one subdivison during the training/test
        if os.path.exists('./auxilary/icoshpere_mesh_subdivision_1.npy'):
            self.symmetry_subdiv_data =  [np.load('./auxilary/icoshpere_mesh_subdivision_1.npy', allow_pickle = True).item()]

            

        
        #############################
        # MEAN SHAPE INITIALIZATION #
        #############################
        self.mean_shape_subdivision_num = 0 # how much times predicred mean shapes  were subdivided 
        verts_np, faces_np = meshzoo.icosa_sphere(n=self.opts.subdivide)
        verts_np = verts_np * 0.25  # sphere vertices in [-0.25, 0.25]
       
        verts = torch.from_numpy(verts_np).float()
        faces = torch.from_numpy(faces_np).long()

        self.icosa_sphere_v = verts.detach().clone()
        self.icosa_sphere_f = faces.detach().clone()

        if self.opts.fixed_meanshape:
            self.mean_v = nn.Parameter(verts.unsqueeze(0).repeat(self.num_classes, 1, 1), requires_grad=False)
        else:
            self.mean_v = nn.Parameter(verts.unsqueeze(0).repeat(self.num_classes, 1, 1), requires_grad=True)

        self.faces = nn.Parameter(faces, requires_grad=False)

        ############
        # NETWORKS #
        ############
        # ResNet18 feature extractor
        self.encoder = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat, enc_flat_type=self.opts.enc_flat_type,
                               pretrain=not self.disable_resnet_pretrain)

        # shape deformation predictor
        # DeepSDF: default parameters are as in the DeepSDF original experiments
        output_size = 3
        self.shape_predictor = DeepSDFDecoder(latent_size=self.encoder.out_shape_flat + self.num_classes,
                                              output_size=output_size,
                                              dims=[512, 512, 512, 512],
                                              dropout=[0, 1, 2, 3],
                                              dropout_prob=0.2,
                                              norm_layers=[0, 1, 2, 3],
                                              latent_in=[2]
                                              )

        # camera rototranslation predictor
        self.camera_predictor = torch.nn.Sequential(torch.nn.Dropout(p=0.5),
                                                    fc(batch_norm=True, nc_inp=self.encoder.out_shape_flat, nc_out=64),
                                                    # torch.nn.Dropout(p=0.2),
                                                    torch.nn.Linear(64, 7))

        # meanshape classifier
        if self.use_learned_class:
            self.class_classifier = nn.Sequential(
                # torch.nn.Dropout(p=0.5),
                fc(batch_norm=True, nc_inp=self.encoder.out_shape_flat, nc_out=64),
                nn.Linear(64, self.num_classes),
                nn.Softmax(dim=-1)
            )

        # texture predictor
        if self.texture_type == 'surface':
            self.albedo_predictor = TexturePredictor(n_upconv=6,
                                                     upconv_mode=self.opts.upconv_mode,
                                                     decoder_name=self.opts.decoder_name,
                                                     norm_mode=self.opts.norm_mode)
            self.albedo_predictor.set_uv_sampler(self.icosa_sphere_v.detach().cpu().numpy(),
                                                 self.icosa_sphere_f.detach().cpu().numpy(),
                                                 self.opts.tex_size)  # shape size doesn't affect this
            net_init(self.albedo_predictor)
        elif self.texture_type == 'vertex':
            self.albedo_predictor = DeepSDFDecoder(latent_size=self.encoder.out_shape_flat + self.num_classes,
                                                   output_size=3,
                                                   dims=[512, 512, 512, 512],
                                                   dropout=[0, 1, 2, 3],
                                                   dropout_prob=0.2,
                                                   norm_layers=[0, 1, 2, 3],
                                                   latent_in=[2]
                                                   )

    def forward(self, img, mean_shape):
        img_feat, img_feat_flat = self.encoder.forward(img)

        if self.use_learned_class:
            class_pred = self.class_classifier(img_feat_flat) #weights of meanshapes 
            mean_shape = (mean_shape * class_pred.unsqueeze(-1).unsqueeze(-1)).sum(1) 
        else:
            class_pred = None

        latent = torch.cat([img_feat_flat, class_pred], dim=1)
        deform_pred_ = self.shape_predictor.forward(latent, mean_shape)

        if self.enable_symmetrization:
            deform_pred = self.symmetrize(deform_pred_)
        else:
            deform_pred = deform_pred_

        camera_pred = self.camera_predictor.forward(img_feat_flat)
        quat_pred = camera_pred[:, :4]
        quat_pred = torch.tanh(quat_pred)
        quat_pred_norm = torch.nn.functional.normalize(quat_pred)
        trans_pred = camera_pred[:, 4:]

        if self.texture_type == 'surface':
            albedo_pred = self.albedo_predictor.forward(img_feat)
        elif self.texture_type == 'vertex':
            latent = torch.cat([img_feat_flat, class_pred], dim=1)
            albedo_pred = self.albedo_predictor.forward(latent, mean_shape + deform_pred)

        return deform_pred, quat_pred, quat_pred_norm, trans_pred, albedo_pred, class_pred, mean_shape

    def subdivide_mesh(self):
        new_mean_v = []
        subdivider = None
        for i, mean_verts in enumerate(self.get_mean_shape()):
            old_mesh = Meshes(verts=[mean_verts.detach().clone()], faces=[self.faces.detach().clone()])
            if subdivider is None:
                subdivider = SubdivideMeshes(old_mesh)
            new_mesh = subdivider(old_mesh)  # type: Meshes
            new_verts = new_mesh.verts_packed()
            new_mean_v += [new_verts]

        new_faces = new_mesh.faces_packed()

        self.mean_v = nn.Parameter(torch.stack(new_mean_v))
        self.faces = nn.Parameter(new_faces, requires_grad=False)

        # same for icosa_sphere
        old_mesh = Meshes(verts=[self.icosa_sphere_v], faces=[self.icosa_sphere_f])
        subdivider = SubdivideMeshes(old_mesh)
        new_mesh = subdivider(old_mesh)  # type: Meshes
        new_verts = new_mesh.verts_packed()
        new_faces = new_mesh.faces_packed()

        self.icosa_sphere_v = new_verts
        self.icosa_sphere_f = new_faces

        self.mean_shape_subdivision_num = self.mean_shape_subdivision_num + 1 

        return self.mean_v

    def symmetrize_mesh(self, verts):
        """
        Assumes vertices are arranged as [indep, left] (the code is incomplete, I'm using my code instead)
        """
        num_indep = self.indep_idx.shape[0]
        indep = verts[..., :num_indep, :]
        left = verts[..., num_indep:, :]
        right = verts[..., num_indep:, :] * torch.tensor([-1, 1, 1], dtype=verts.dtype,
                                                         device=verts.device).view((1,) * (verts.dim() - 1) + (3,))
        ilr = torch.cat([indep, left, right], dim=-2)
        assert (self.ilr_idx_inv.max() < ilr.shape[-2]), f'idx ({self.ilr_idx_inv.max()}) >= dim{-2} of {ilr.shape}'
        verts_full = torch.index_select(ilr, -2, self.ilr_idx_inv.to(ilr.device))

        return verts_full

    def get_mean_shape(self):
        if self.enable_symmetrization:
            return self.symmetrize(self.mean_v)
        else:
            return self.mean_v

    def get_symmetry_sliced_indices(self):
        data =[]
        if self.mean_shape_subdivision_num > 0: 
            data = self.symmetry_subdiv_data[self.mean_shape_subdivision_num-1][self.opts.subdivide]
        else:
            data = self.symmetry_data[self.opts.subdivide]
        
        return data['V_right_slice'], data['V_left_slice']

    def get_symmetry_indices(self):
        data =[]
        if self.mean_shape_subdivision_num > 0: 
            data = self.symmetry_subdiv_data[self.mean_shape_subdivision_num-1][self.opts.subdivide]
        else:
            data = self.symmetry_data[self.opts.subdivide]
            
        return data['V_right'], data['V_left'],data['V_opposite'],  data['V_middle']
    
    # # Add "vert[:, center_indices,0] =0" like in auxilary.test_postprocessing.compute_additional_test_metrics !!!
    def symmetrize_simple(self, vert_not_sym):
        vert = vert_not_sym.clone()
        V_right, V_left, V_opposite, V_middle = self.get_symmetry_indices()
        LR_indices = V_right if self.enable_symmetrization == 1 else V_left
        vert[:,LR_indices,:] =   vert[:,V_opposite[LR_indices],:]
        vert[:,LR_indices,0] =  -vert[:,V_opposite[LR_indices],0]
        if self.align_middle_vertices_in_symmetrization:
              vert[:,V_middle,0] = 0 #without it predicted deformation has non-zero hasudrorff symmetric error 
        return vert
    
    #new version that supports 3 types of symmetrization
    def symmetrize(self, V):
        if type(self.enable_symmetrization)==list: #run symmetrization with vertical slices 
            return self.symmetrize_pro(V)
        
        vert = V.clone()
        V_right, V_left, V_opposite, V_middle = self.get_symmetry_indices()
        #V_left_match = V_opposite[V_right] #matched 

        symm_coeff = [0.0, 1.0, 0.5] [int(self.enable_symmetrization)-1] #1-> 1, 2->0, 3->0.5

        vert[:,V_right,:]  =    symm_coeff*         V[:,V_right,:]               + (1-symm_coeff)*reflectX(V[:,V_opposite[V_right],:])
        vert[:,V_left,:]   =    symm_coeff*reflectX(V[:,V_opposite[V_left],:])   + (1-symm_coeff)*         V[:,V_left,:]

        vert[:,V_middle,0] = 0

        return vert
    
    #Uses vertical slices to get a larger set of possible mesh symmetrizations
    def symmetrize_pro(self, V):
        vert = V.clone()
        
        _, _, V_opposite, V_middle  = self.get_symmetry_indices()
        V_right_slice, V_left_slice = self.get_symmetry_sliced_indices()
        
        
        #symm_coeff = [0.0, 1.0, 0.5] [int(self.enable_symmetrization)-1] #1-> 1, 2->0, 3->0.5
        symm_coeff = [] # [[0]]*len(self.enable_symmetrization.enable_symmetrization)
        for i in range(len(self.enable_symmetrization)):
            if not self.enable_symmetrization[i]:
                continue
            symm_coeff = [0.0, 1.0, 0.5] [int(self.enable_symmetrization[i])-1] #1-> 1, 2->0, 3->0.5
            V_right  = V_right_slice[i]
            V_left   = V_left_slice[i]
            vert[:,V_right,:]  =    symm_coeff*         V[:,V_right,:]               + (1-symm_coeff)*reflectX(V[:,V_opposite[V_right],:])
            vert[:,V_left,:]   =    symm_coeff*reflectX(V[:,V_opposite[V_left],:])   + (1-symm_coeff)*         V[:,V_left,:]

  

        vert[:,V_middle,0] = 0

        return vert


    def compute_symmetry_errors(self, vert):
        V_right, V_left, V_opposite, _ = self.get_symmetry_indices()
        symm_erros = []
        for i in range(vert.shape[0]):
            V, F  = self.tensor_mesh_to_numpy(vert[i,:], self.faces)
            symm_erros.append(gu_measure_symmetries(V, F, V_right, V_opposite))
        return symm_erros
    
    def tensor_mesh_to_numpy(self, t_vert, t_face):
        if torch.is_tensor(t_vert):
            #V = t_vert.squeeze(0).cpu().detach().numpy()
            V = t_vert.squeeze(0).cpu().detach().numpy()
        else:
            V  = t_vert
        if torch.is_tensor(t_face):
            F = t_face.squeeze(0).cpu().detach().numpy()
        else:
             F = t_face
        return V, F



