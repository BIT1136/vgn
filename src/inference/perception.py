import numpy as np
import open3d as o3d

from .utils import map_cloud_to_grid


class UniformTSDFVolume:
    def __init__(self, length, resolution):
        self.length = length#TSDF体积的边长,单位m
        self.resolution = resolution#TSDF体积的分辨率
        self.voxel_size = self.length / self.resolution
        self.sdf_trunc = 4 * self.voxel_size
        self.o3dvol = o3d.pipelines.integration.UniformTSDFVolume(
            length=self.length,
            resolution=self.resolution,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

    # def reset(self):
    #     self.o3dvol.reset()

    def integrate(self, depth_img, intrinsic, extrinsic):
        """
        Args:
            depth_img: 单位m
        """
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.empty_like(depth_img)),
            o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic.width,
            height=intrinsic.height,
            fx=intrinsic.fx,
            fy=intrinsic.fy,
            cx=intrinsic.cx,
            cy=intrinsic.cy,
        )
        extrinsic = extrinsic.as_matrix()
        self.o3dvol.integrate(rgbd, intrinsic, extrinsic)

    def get_scene_cloud(self):
        return self.o3dvol.extract_point_cloud()
    #文档:提取具有法线的点云
    #物体表面的点(更精细),不包括桌面

    def get_map_cloud(self):
        return self.o3dvol.extract_voxel_point_cloud()
    #文档:调试函数将体素数据提取到点云中
    #每个有值的体素一个点

    def get_grid(self):
        map_cloud = self.get_map_cloud()
        points = np.asarray(map_cloud.points)
        distances = np.asarray(map_cloud.colors)[:, [0]]
        return map_cloud_to_grid(self.voxel_size, points, distances,self.resolution)


def create_tsdf(size, resolution, imgs, intrinsic, views):
    tsdf = UniformTSDFVolume(size, resolution)
    for img, view in zip(imgs, views):
        tsdf.integrate(img, intrinsic, view.inv())
    return tsdf
