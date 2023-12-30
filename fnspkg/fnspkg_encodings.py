import itertools
from typing import Literal, Sequence
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.field_components.encodings import Encoding

class FNspkgEncoding(Encoding):
    """Learned K-Planes encoding

    A plane encoding supporting both 3D and 4D coordinates. With 3D coordinates this is similar to
    :class:`TriplaneEncoding`. With 4D coordinates, the encoding at point ``[i,j,k,q]`` is
    a n-dimensional vector computed as the elementwise product of 6 n-dimensional vectors at
    ``planes[i,j]``, ``planes[i,k]``, ``planes[i,q]``, ``planes[j,k]``, ``planes[j,q]``,
    ``planes[k,q]``.

    Unlike :class:`TriplaneEncoding` this class supports different resolution along each axis.

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of the grid. Can be a sequence of 3 or 4 integers.
        num_components: The number of scalar planes to use (ie: output feature size)
        init_a: The lower-bound of the uniform distribution used to initialize the spatial planes
        init_b: The upper-bound of the uniform distribution used to initialize the spatial planes
        reduce: Whether to use the element-wise product of the planes or the sum
    """

    def __init__(
        self,
        resolution: Sequence[int] = (128, 128, 128),
        num_components: int = 64,
        init_a: float = 0.1,
        init_b: float = 0.5,
        reduce: Literal["sum", "product"] = "product",
    ) -> None:
        super().__init__(in_dim=len(resolution))

        self.resolution = resolution
        self.num_components = num_components
        self.reduce = reduce
        if self.in_dim not in {3, 4}:
            raise ValueError(
                f"The dimension of coordinates must be either 3 (static scenes) "
                f"or 4 (dynamic scenes). Found resolution with {self.in_dim} dimensions."
            )
        has_time_planes = self.in_dim == 4

        self.coo_combs = list(itertools.combinations(range(self.in_dim), 2))
        # Unlike the Triplane encoding, we use a parameter list instead of batching all planes
        # together to support uneven resolutions (especially useful for time).
        # Dynamic models (in_dim == 4) will have 6 planes:
        # (y, x), (z, x), (t, x), (z, y), (t, y), (t, z)
        # static models (in_dim == 3) will only have the 1st, 2nd and 4th planes.
        self.plane_coefs = nn.ParameterList()
        for coo_comb in self.coo_combs:
            new_plane_coef = nn.Parameter(
                torch.empty([self.num_components] + [self.resolution[cc] for cc in coo_comb[::-1]])
            )
            if has_time_planes and 3 in coo_comb:  # Time planes initialized to 1
                nn.init.ones_(new_plane_coef)
            else:
                nn.init.uniform_(new_plane_coef, a=init_a, b=init_b)
            self.plane_coefs.append(new_plane_coef)

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs input_dim"]) -> Float[Tensor, "*bs output_dim"]:
        """Sample features from this encoder. Expects ``in_tensor`` to be in range [-1, 1]"""
        original_shape = in_tensor.shape

        assert any(self.coo_combs)
        output = 1.0 if self.reduce == "product" else 0.0  # identity for corresponding op
        for ci, coo_comb in enumerate(self.coo_combs):
            grid = self.plane_coefs[ci].unsqueeze(0)  # [1, feature_dim, reso1, reso2]
            coords = in_tensor[..., coo_comb].view(1, 1, -1, 2)  # [1, 1, flattened_bs, 2]
            interp = F.grid_sample(
                grid, coords, align_corners=True, padding_mode="border"
            )  # [1, output_dim, 1, flattened_bs]
            interp = interp.view(self.num_components, -1).T  # [flattened_bs, output_dim]
            if self.reduce == "product":
                output = output * interp
            else:
                output = output + interp

        # Typing: output gets converted to a tensor after the first iteration of the loop
        assert isinstance(output, Tensor)
        return output.reshape(*original_shape[:-1], self.num_components)



