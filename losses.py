import torch
from pytorch3d.ops import knn_points

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	criterion = torch.nn.BCEWithLogitsLoss()
	loss = criterion(voxel_src, voxel_tgt)

	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	dist_src, _, _ = knn_points(point_cloud_src, point_cloud_tgt, return_nn=True)
	dist_tgt, _, _ = knn_points(point_cloud_tgt, point_cloud_src, return_nn=True)

	loss_chamfer = (torch.mean(dist_src) + torch.mean(dist_tgt)) / 2

	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss

	L = mesh_src.laplacian_packed()
	LV = L @ mesh_src.verts_packed()
	LV = torch.norm(LV, dim=1)
	loss_laplacian = torch.mean(LV ** 2)
	
	return loss_laplacian