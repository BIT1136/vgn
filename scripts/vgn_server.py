#!/root/mambaforge/envs/vgn/bin/python

from pathlib import Path
import rospy
from ros_numpy import numpify
import cv_bridge

from inference.detection import VGN, select_local_maxima
from inference.rviz import Visualizer
from inference.utils import *
from robot_helpers.ros.conversions import *
from inference.perception import UniformTSDFVolume
from robot_helpers.ros import tf

from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_srvs.srv import Empty
from vgn.srv import PredictGrasps,PredictGraspsResponse


class VGNServer:
    def __init__(self):
        self.frame = rospy.get_param("~frame_id")
        self.vgn = VGN("src/vgn/models/vgn_conv.pth")
        rospy.Service("predict_grasps", PredictGrasps, self.predict_grasps)
        self.length = rospy.get_param("~length")
        self.resolution = rospy.get_param("~resolution")
        rospy.Service("reset_map", Empty, self.reset)
        self.reset()
        self.depth_scale = rospy.get_param("~depth_scale")
        tf.init()
        self.cam_frame_id = rospy.get_param("~camera/frame_id")
        self.frame_id = rospy.get_param("~frame_id")
        msg = rospy.wait_for_message("camera/info_topic", CameraInfo)
        self.intrinsic = from_camera_info_msg(msg)
        self.cv_bridge = cv_bridge.CvBridge()

        self.scene_cloud_pub = rospy.Publisher("scene_cloud", PointCloud2, queue_size=1)
        self.map_cloud_pub = rospy.Publisher("map_cloud", PointCloud2, queue_size=1)
        rospy.Subscriber("depth", Image, self.callback)

        self.vis = Visualizer()
        rospy.loginfo("VGN server ready")

    def reset(self):
        self.tsdf = UniformTSDFVolume(self.length, self.resolution)
        return

    def callback(self, msg):
        depth=numpify(msg).astype(np.float32)*self.depth_scale
        extrinsic = tf.lookup(
            self.cam_frame_id, self.frame_id, msg.header.stamp, rospy.Duration(0.1)
        )
        self.tsdf.integrate(depth, self.intrinsic, extrinsic)

        scene_cloud = self.tsdf.get_scene_cloud()
        points = np.asarray(scene_cloud.points)
        msg = to_cloud_msg(self.frame_id, points)
        self.scene_cloud_pub.publish(msg)

        map_cloud = self.tsdf.get_map_cloud()
        points = np.asarray(map_cloud.points)
        distances = np.asarray(map_cloud.colors)[:, [0]]
        msg = to_cloud_msg(self.frame_id, points, distances=distances)
        self.map_cloud_pub.publish(msg)

    def predict_grasps(self, req):
        # Construct the input grid
        voxel_size = req.voxel_size
        points, distances = from_cloud_msg(req.map_cloud)
        tsdf_grid = map_cloud_to_grid(voxel_size, points, distances)

        # Compute grasps
        out = self.vgn.predict(tsdf_grid)
        grasps, qualities = select_local_maxima(voxel_size, out, threshold=0.9)

        # Visualize detections
        self.vis.grasps(self.frame, grasps, qualities)

        # Construct the response message
        res = PredictGraspsResponse()
        res.grasps = [to_grasp_config_msg(g, q) for g, q in zip(grasps, qualities)]
        return res


if __name__ == "__main__":
    rospy.init_node("vgn_server")
    VGNServer()
    rospy.spin()
