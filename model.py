from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
import math
 
from layers import GBottleneck, adjacency_from_faces


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512 -> b x 64 x 2 x 2 x 2
            # Output: b x 32 x 32 x 32
            pass
            # TODO:
            # self.linear1 = nn.Linear(512, 128*2*2*2)

            self.decoder = nn.Sequential(
                torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, bias=False, padding=1), #shape: 32 x 4 x 4 x 4
                torch.nn.BatchNorm3d(64),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, bias=False, padding=1), #shape: 32 x 4 x 4 x 4
                torch.nn.BatchNorm3d(32),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, bias=False, padding=1), # shape: 16 x 8 x 8 x 8
                torch.nn.BatchNorm3d(16),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, bias=False, padding=1), #shape: 8 x 16 x 16 x 16
                torch.nn.BatchNorm3d(8),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, bias=False, padding=1), #shape: 4 x 32 x 32 x 32
                torch.nn.BatchNorm3d(4),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose3d(4, 1, kernel_size=1, bias=False),
            )
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.decoder = nn.Sequential(
                torch.nn.Linear(512, 512),
                torch.nn.LayerNorm(512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.LayerNorm(256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.LayerNorm(128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 3)
            )
            self.linear1 = torch.nn.Linear(2, 512)

            sqrt_n = int(math.sqrt(self.n_point + 0.1))
            grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, steps=sqrt_n), torch.linspace(0, 1, steps=sqrt_n), indexing="ij")
            self.grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            # TODO:
            # self.decoder 
            faces = mesh_pred.faces_list()[0]
            num_verts = mesh_pred.verts_list()[0].shape[0]
            adj = adjacency_from_faces(faces, num_verts)
            deg = adj.sum(1)
            deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
            adj_norm = deg_inv_sqrt @ adj @ deg_inv_sqrt

            self.n_point = num_verts
            self.decoder = GBottleneck(6, 515, 128, 3, adj_norm, activation=True).to(device=self.device)

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            # encoded_feat = self.linear1(encoded_feat)
            encoded_feat = encoded_feat.reshape(B, 64, 2, 2, 2)
            voxels_pred = self.decoder(encoded_feat)
            return voxels_pred

        elif args.type == "point":
            # TODO:
            # pointclouds_pred = 
            x = torch.zeros(B, self.n_point, 2)
            D = encoded_feat.shape[1]

            if args.eval:
                x = self.grid.unsqueeze(0).repeat(B, 1, 1).to(device=args.device)
            else:
                x = torch.rand(B, self.n_point, 2).to(device=args.device)
            
            x = self.linear1(x)+encoded_feat.unsqueeze(1).expand(B, self.n_point, D)
            pointclouds_pred = self.decoder(x.reshape(-1, D))
            pointclouds_pred = pointclouds_pred.reshape(B, -1, 3)

            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            D = encoded_feat.shape[1]
            encoded_feat = encoded_feat.unsqueeze(1).expand(B, self.n_point, D)
            verts = self.mesh_pred.verts_padded()
            deform_vertices_pred, _ = self.decoder(torch.cat([encoded_feat, verts], dim=2))          
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))

            return  mesh_pred