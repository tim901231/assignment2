import torch
from pytorch3d.structures import Meshes

# 1. Define the 4 vertices (nodes) of the tetrahedron.
# We place one at the origin and the others along the axes.
verts = torch.tensor([
    [0., 0., 0.],  # Vertex 0
    [1., 0., 0.],  # Vertex 1
    [0., 1., 0.],  # Vertex 2
    [0., 0., 1.]   # Vertex 3
], dtype=torch.float32)

# 2. Define the 4 triangular faces using the indices of the vertices.
faces = torch.tensor([
    [0, 2, 1],  # Base triangle
    [0, 1, 3],  # Side triangle
    [0, 3, 2],  # Side triangle
    [1, 2, 3]   # Top triangle
], dtype=torch.int64)

# 3. Create a Meshes object.
# The inputs must be lists, so we wrap our tensors in [].
mesh = Meshes(verts=[verts], faces=[faces])

# 4. Compute the packed cotangent Laplacian.
L = mesh.laplacian_packed()

# 5. For a small matrix, it's easier to inspect the dense version.
L_dense = L.to_dense()

print("Full Dense Laplacian Matrix (4x4):")
print(L_dense)


# 6. We can also compute the Laplacian coordinates.
# This shows the "smoothness vector" for each vertex.
laplacian_coords = L @ verts
print("\nLaplacian Coordinates (L @ V):")
print(laplacian_coords)