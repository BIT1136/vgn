import vgn
from scipy.spatial.transform import Rotation#不知道为什么可以避免链接器错误

import rospy
import ros_numpy

import numpy as np
import pickle

from sensor_msgs.msg import Image

from robot_helpers.ros import tf

i=0

def callback(msg):
    global i
    nmsg=ros_numpy.numpify(msg)
    print(nmsg.shape)
    # np.save(f"depth{i}.npy",nmsg)
    extrinsic = tf.lookup("depth", "object_base",timeout=rospy.Duration(0))
    print(extrinsic)
    # with open(f'_extrinsic{i}.ext', 'wb') as f:
    #     pickle.dump(extrinsic, f)
    i+=1

rospy.init_node("get_msg")
tf.init()
rospy.Subscriber("/d435/camera/depth/image_convert", Image, callback)
print("ready")
rospy.spin()
