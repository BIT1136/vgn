
import numpy as np
import sys
sys.path.append('src')
import pickle
import matplotlib.pyplot as plt
import rospy
import open3d as o3d

import vgn
from inference.detection import VGN, select_local_maxima
from inference.perception import UniformTSDFVolume
from inference.rviz import Visualizer
from robot_helpers.perception import CameraIntrinsic
from inference.utils import *
from sensor_msgs.msg import Image

def to_u8(array: np.ndarray, max=255):
    array = array - array.min()
    if array.max() == 0:
        return array.astype(np.uint8)
    return (array * (max / array.max())).astype(np.uint8)

def pub_scene_cloud():
    scene_cloud = tsdf.get_scene_cloud()
    points = np.asarray(scene_cloud.points)
    msg = to_cloud_msg("object_base", points)
    spub.publish(msg)

def pub_map_cloud():
    map_cloud = tsdf.get_map_cloud()
    points = np.asarray(map_cloud.points)
    distances = np.asarray(map_cloud.colors)[:, [0]]
    msg = to_cloud_msg("object_base", points, distances=distances)
    mpub.publish(msg)
    
rospy.init_node("pub",disable_signals=True)

mpub=rospy.Publisher("map_cloud", PointCloud2, queue_size=1)
spub=rospy.Publisher("scene_cloud", PointCloud2, queue_size=1)
fpub=rospy.Publisher("fixed_depth", Image, queue_size=1)

volume_size = 0.5
tsdf_resolution = 40
tsdf = UniformTSDFVolume(volume_size, tsdf_resolution)
intrinsic=CameraIntrinsic(640, 480, 554, 554, 320, 240)
vis=Visualizer()
vgn = VGN("models/vgn_conv.pth")

for i in range(1):
    # a=input(f"{i}>")
    with open(f'extrinsic{i}.ext', 'rb') as f:
        extrinsic = pickle.load(f)
    with open(f'depth{i}.npy', 'rb') as f:
        depth=np.load(f)
    depth=depth.astype(np.float32)/1000
    u8depth=to_u8(depth)
    fpub.publish(ros_numpy.msgify(Image,u8depth,encoding='mono8'))
    print(extrinsic.as_matrix())
    tsdf.integrate(depth, intrinsic, extrinsic)
    pub_map_cloud()
    pub_scene_cloud()
    voxel_size=volume_size/tsdf_resolution
    tsdf_grid=tsdf.get_grid()
    out = vgn.predict(tsdf_grid)
    grasps, qualities = select_local_maxima(voxel_size, out, threshold=0.9)
    vis.grasps("object_base", grasps, qualities)
    for grasp,qual in zip(grasps,qualities):
        print(grasp.pose,qual)
        print(extrinsic.inv()*grasp.pose)
    


# print(qualities)

# input()

# print("Extract a triangle mesh from the volume and visualize it.")
# mesh = tsdf.o3dvol.extract_triangle_mesh()
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], front=[0.5297, -0.1873, -0.8272],
#                                   lookat=[2.0712, 2.0312, 1.7251],
#                                   up=[-0.0558, -0.9809, 0.1864], zoom=0.47)

# ev = o3d.visualization.ExternalVisualizer(address='tcp://192.168.3.5:51454',timeout=1000)
# ev.set(mesh)

# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(tsdf_grid, edgecolor='k')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# ax.set_xlim(10,30)
# ax.set_ylim(0,20)
# ax.set_zlim(0,20)

# plt.show()

