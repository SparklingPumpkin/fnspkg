
from typing import Dict, Iterable, List, Optional, Tuple, Sequence
from torchtyping import TensorType

import torch
from rich.console import Console
from torch import nn, Tensor

from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from .fnspkg_encodings import FNspkgEncoding  # !

from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions


############
import tinycudann as tcnn
##########






# 美化
CONSOLE = Console(width=120)

def interpolate_ms_features(
        pts: torch.Tensor,
        grid_encodings: Iterable[FNspkgEncoding],  # !
        concat_features: bool,
) -> torch.Tensor:
    """interpolate_multi-scale_features

    Args:
        pts: 需要查询的点 points
        grid_encodings: 要查询的网格编码
        concat_features: bool, 不同尺度上的特征是否要被串联

    Returns:
        特征向量
    """

    multi_scale_interp = [] if concat_features else 0.0
    for grid in grid_encodings:
        grid_features = grid(pts)

        if concat_features:
            multi_scale_interp.append(grid_features)
        else:
            multi_scale_interp = multi_scale_interp + grid_features

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)

    return multi_scale_interp




class FNspkgField(Field):
    """fnspkg field

    Args:
        aabb: scene aabb bounds 的参数
        num_images: dataset 图片 num
        num_layers: 隐藏层 num
        hidden_dim: 隐藏层 dim
        geo_feat_dim: 几何特征维度 如位置、方向
        num_levels: 哈希表用于对应 base mlp 的层数 -- instantngp 的方法
        base_res: 哈希表用于对应 base mlp 的 基础分辨率 -- instantngp
        max_res: 哈希表用于对应 base mlp 的 最大分辨率 -- instantngp
        log2_hashmap_size: 哈希表用于对应 base mlp 的 大小 (用2的对数表示) -- instantngp
        num_layers_color: 颜色 net 的隐藏层 num
        features_per_level: 每级 哈希网格 的特征数
        hidden_dim_color: 颜色 net 的隐藏层 dim
        appearance_embedding_dim: 外观 embedding dim
        use_average_appearance_embedding: 是否使用 均值外观 embedding 或 0
        spatial_distortion: 空间形变类型
        implementation: 实现方式, 分为 tcnn-tinycudann 和 torch

        其中 外观 embedding 指的是颜色/ 纹理等

        ###
        concat_across_scales: 是否在不同尺度上串联特征
        multiscale_res: 多尺度网格分辨率
        grid_base_resolution: Base grid resolution
        grid_feature_dim: Dimension of feature vectors stored in grid
    """

    aabb: TensorType  # 边界

    def __init__(
            # 超参数设定
            self,
            aabb: TensorType,
            num_images: int,
            # max_res: int = 2048,
            # num_levels: int = 16,
            # log2_hashmap_size: int = 19,

            geo_feat_dim: int = 15,
            # base_res: int = 16,
            appearance_embedding_dim: int = 32,
            spatial_distortion: Optional[SpatialDistortion] = None,
            use_average_appearance_embedding: bool = False,

            # k-planes (不使用线性解码器)
            concat_across_scales: bool = True,
            multiscale_res: Sequence[int] = (1, 2, 4),
            grid_base_resolution: Sequence[int] = (128, 128, 128),
            grid_feature_dim: int = 32,
    ) -> None:
        super().__init__()
        # register_buffer() 将 aabb 注册为一个缓冲区。这意味着 aabb 会被作为模型的
        # 一部分进行保存和加载，但不会被视为模型的参数，因此在训练期间不会被更新. 后同
        self.register_buffer("aabb", aabb)
        # self.register_buffer("max_res", torch.tensor(max_res))
        # self.register_buffer("num_levels", torch.tensor(num_levels))
        # self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.num_images = num_images
        # self.base_res = base_res
        self.geo_feat_dim = geo_feat_dim
        self.appearance_embedding_dim = appearance_embedding_dim
        self.spatial_distortion = spatial_distortion
        self.use_average_appearance_embedding = use_average_appearance_embedding

        # 初定义
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.step = 0  # 初始化步数为0

        # k-planes
        self.geo_feat_dim = geo_feat_dim
        self.concat_across_scales = concat_across_scales
        self.grid_base_resolution = list(grid_base_resolution)
        # 初定义
        self.has_time_planes = len(grid_base_resolution) > 3  # 用于确定是否存在时间维度

        # -----------条件定义----------- #
        # k-planes
        # 初始化 planes -- 多尺度网格分辨率
        self.grids = nn.ModuleList()
        for res in multiscale_res:
            # Resolution fix: multi-res only on spatial planes
            resolution = [r * res for r in self.grid_base_resolution[:3]] + self.grid_base_resolution[3:]
            self.grids.append(FNspkgEncoding(resolution, grid_feature_dim))  # !
        self.feature_dim = (
            grid_feature_dim * len(multiscale_res) if self.concat_across_scales
            else grid_feature_dim
        )

        # 初始化解码网络, 神经网络架构
        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=self.geo_feat_dim + 1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )
        in_dim_color = (
                self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim
        )
        self.color_net = tcnn.Network(
            n_input_dims=in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        '''计算密度'''
        # 确定光线在场景中的中心位置 (视锥体中心), 根据设定是否考虑偏移, 返回 xyz
        if self.spatial_distortion is not None:
            # 处理空间缩放
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = positions / 2  # !k-planes-change from [-2, 2] to [-1, 1]
        else:
            # From [0, 1] to [-1, 1]
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb) * 2.0 - 1  # !k-planes-change

        # k-planes 时间维度
        if self.has_time_planes:
            assert ray_samples.times is not None, "Initialized model with time-planes, but no time data is given"
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = ray_samples.times * 2.0 - 1.0
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        # 展平张量 用于输入 MLP
        positions_flat = positions.view(-1, positions.shape[-1])

        # k-planes 特征插值 以及输出
        features = interpolate_ms_features(
            positions_flat, grid_encodings=self.grids, concat_features=self.concat_across_scales
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device, requires_grad=True)
        features = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        features, density_before_activation = torch.split(features, [self.geo_feat_dim, 1], dim=-1)

        # h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        # density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        self._density_before_activation = density_before_activation

        density = trunc_exp(density_before_activation.to(positions) - 1)

        return density, features

    def get_outputs(
            self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        '''get_outputs函数是返回自定义数据时需要实现的函数, 此处是输出 RGB'''

        assert density_embedding is not None

        output_shape = ray_samples.frustums.shape
        directions = ray_samples.frustums.directions.reshape(-1, 3)

        if 0:  # !self.linear_decoder
            color_features = [density_embedding]
        else:
            directions = get_normalized_directions(directions)
            d = self.direction_encoding(directions)
            color_features = [d, density_embedding.view(-1, self.geo_feat_dim)]

        if self.appearance_embedding_dim > 0:
            if self.training:
                assert ray_samples.camera_indices is not None
                camera_indices = ray_samples.camera_indices.squeeze()
                embedded_appearance = self.appearance_ambedding(camera_indices)
            elif self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_ambedding.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*output_shape, self.appearance_embedding_dim),
                    device=directions.device,
                )

            if not 0:  # self.linear_decoder
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)
        if 0: #  self.linear_decoder
            basis_input = directions
            if self.appearance_ambedding_dim > 0:
                basis_input = torch.cat([directions, embedded_appearance], dim=-1)
            basis_values = self.color_basis(basis_input)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(basis_input.shape[0], 3, -1)  # [batch, color_feature_len, 3]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = torch.sigmoid(rgb).view(*output_shape, -1).to(directions)
        else:
            rgb = self.color_net(color_features).view(*output_shape, -1)

        return {FieldHeadNames.RGB: rgb}




class FNspkgDensityField(Field):
    """
        轻量化密度求解模型
    """

    def __init__(
        self,
        aabb: TensorType,
        resolution: List[int],
        num_output_coords: int,
        spatial_distortion: Optional[SpatialDistortion] = None,
        # linear_decoder: bool = False,
    ):
        super().__init__()

        self.register_buffer("aabb", aabb)

        self.spatial_distortion = spatial_distortion
        self.has_time_planes = len(resolution) > 3
        self.feature_dim = num_output_coords
        # self.linear_decoder = linear_decoder

        self.grids = FNspkgEncoding(resolution, num_output_coords, init_a=0.1, init_b=0.15)

        self.sigma_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "None" if 0 else "ReLU",  # self.linear_decoder
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        CONSOLE.log(f"Initialized KPlaneDensityField. with time-planes={self.has_time_planes} - resolution={resolution}")

    # pylint: disable=arguments-differ
    def density_fn(self, positions: TensorType["bs":..., 3], times: Optional[TensorType["bs", 1]] = None) -> TensorType["bs":..., 1]:

        if times is not None and (len(positions.shape) == 3 and len(times.shape) == 2):
            # position is [ray, sample, 3]; times is [ray, 1]
            times = times[:, None]  # RaySamples can handle the shape
        # Need to figure out a better way to descibe positions with a ray.
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times,
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, None]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = positions / 2  # from [-2, 2] to [-1, 1]
        else:
            # From [0, 1] to [-1, 1]
            positions = SceneBox.get_normalized_positions(positions, self.aabb) * 2.0 - 1.0

        if self.has_time_planes:
            assert ray_samples.times is not None, "Initialized model with time-planes, but no time data is given"
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = ray_samples.times * 2.0 - 1.0
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        positions_flat = positions.view(-1, positions.shape[-1])
        features = interpolate_ms_features(
            positions_flat, grid_encodings=[self.grids], concat_features=False
        )
        if len(features) < 1:
            features = torch.zeros((0, 1), device=features.device, requires_grad=True)
        density_before_activation = self.sigma_net(features).view(*ray_samples.frustums.shape, -1)
        density = trunc_exp(density_before_activation.to(positions) - 1)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> dict:
        return {}




















