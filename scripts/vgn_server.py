#!/root/mambaforge/envs/vgn/bin/python

import numpy as np
import time

import rospy
from ros_numpy import numpify

import vgn
from inference.detection import VGN, select_local_maxima
from inference.rviz import Visualizer
from inference import utils
from robot_helpers.ros import conversions
from inference.perception import UniformTSDFVolume
from robot_helpers.ros import tf
from robot_helpers import spatial

from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Empty
from vgn.srv import PredictGrasps, PredictGraspsResponse


class VGNServer:
    def __init__(self):
        self.get_ros_params()
        self.tsdf = UniformTSDFVolume(self.length, self.resolution)
        self.vis = Visualizer()
        self.vis.roi(self.base_frame_id, self.length)

        self.vgn = VGN("../models/vgn_conv.pth")
        tf.init()

        try:
            msg = rospy.wait_for_message(self.info_topic, CameraInfo, 1)
        except Exception as e:
            msg = CameraInfo()
            msg.K = [
                554.382,
                0.0,
                320.0,
                0.0,
                554.382,
                240.0,
                0.0,
                0.0,
                1.0,
            ]
            msg.width, msg.height = 640, 480
            rospy.logwarn(f"{e}, 使用默认相机参数")
        self.intrinsic = conversions.from_camera_info_msg(msg)

        rospy.Service(f"{rospy.get_name()}/reset_map", Empty, self.reset)
        rospy.Service(f"{rospy.get_name()}/predict_grasps", PredictGrasps, self.predict_grasps)
        rospy.Subscriber(self.depth_topic, Image, self.callback)

        rospy.loginfo(f"{rospy.get_name()}节点就绪")

    def get_ros_params(self):
        self.length = rospy.get_param("~length", 0.3)
        self.resolution = rospy.get_param("~resolution", 40)

        self.cam_frame_id = rospy.get_param("~camera/frame_id", "depth")
        self.base_frame_id = rospy.get_param("~frame_id", "object_base")

        self.info_topic = rospy.get_param(
            "~camera/info_topic", "/d435/camera/depth/camera_info"
        )
        self.depth_topic = rospy.get_param(
            "~camera/depth_topic", "/d435/camera/depth/image_convert"
        )

    def reset(self, msg):
        rospy.loginfo("重置tsdf")
        self.tsdf = UniformTSDFVolume(self.length, self.resolution)
        self.vis.clear()
        return []

    def callback(self, msg):
        extrinsic: spatial.Transform = tf.lookup(
            self.cam_frame_id, self.base_frame_id, msg.header.stamp, rospy.Duration(0.1)
        )  # 从tsdf基坐标到相机坐标的变换
        rospy.loginfo(f"进行整合,外参:{extrinsic}")
        depth = numpify(msg).astype(np.float32) / 1000
        self.tsdf.integrate(depth, self.intrinsic, extrinsic)

        scene_cloud = self.tsdf.get_scene_cloud()
        points = np.asarray(scene_cloud.points)
        self.vis.scene_cloud(self.base_frame_id, points)

        map_cloud = self.tsdf.get_map_cloud()
        points = np.asarray(map_cloud.points)
        distances = np.asarray(map_cloud.colors)[:, [0]]
        self.vis.map_cloud(self.base_frame_id, points, distances)

    def predict_grasps(self, msg):
        rospy.loginfo("开始推理")
        t_start = time.perf_counter()
        # 创建TSDF网格
        voxel_size = self.length / self.resolution
        tsdf_grid = self.tsdf.get_grid()

        out = self.vgn.predict(tsdf_grid)
        grasps, qualities = select_local_maxima(voxel_size, out, threshold=0.9)

        idx = np.argsort(qualities)[::-1]
        grasps = grasps[idx]
        qualities = qualities[idx]

        cam_transform = tf.lookup(self.cam_frame_id,self.base_frame_id)
        for g in grasps:
            g.pose = cam_transform * g.pose

        t_end = time.perf_counter()
        rospy.loginfo(f"推理完成,耗时: {(t_end - t_start)*1000:.2f}ms")
        if len(grasps) == 0:
            raise rospy.ServiceException("未检测到抓取")
        else:
            rospy.loginfo(f"检测到 {len(grasps)} 个抓取候选: {[str(g) for g in grasps]}")

        self.vis.quality(self.base_frame_id, voxel_size, out.qual)
        self.vis.grasps(self.cam_frame_id, grasps, qualities)

        res = PredictGraspsResponse()
        res.grasps = [utils.to_grasp_config_msg(g) for g in grasps]
        return res


if __name__ == "__main__":
    rospy.init_node("vgn_server")
    VGNServer()
    rospy.spin()
