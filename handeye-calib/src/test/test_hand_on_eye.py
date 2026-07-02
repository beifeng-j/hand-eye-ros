#!/usr/bin/env python3
# coding: utf-8
"""
手眼标定测试程序，眼在手上
输入：
1.aruco码在相机坐标系下的坐标 camera_frame->aruco_marker_frame
2.手眼标定结果base_link->camera_frame
输出：
1.aruco码在机械臂基坐标下的位置,base_link->aruco_maker_frame

单独测试代码：
1.rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 1 base_link link7_name
2.rosrun tf2_ros static_transform_publisher 0 0 2 0 0 0 1 camera_frame aruco_marker_frame
3.roslaunch handeye-calib test_hand_on_eye_calib.launch

"""
import rospy
import tf
import transforms3d as tfs
from tf2_msgs.msg import TFMessage
import geometry_msgs.msg
import sys
import numpy as np
import math
import json
  

if __name__ == '__main__':
    rospy.init_node('test_hand_on_eye')   
    base_link = rospy.get_param("/test_hand_on_eye/base_link")
    end_link = rospy.get_param("/test_hand_on_eye/end_link")
    camera_link = rospy.get_param("/test_hand_on_eye/camera_link")
    marker_link = rospy.get_param("/test_hand_on_eye/marker_link")

    end_link2camera_link = rospy.get_param("/test_hand_on_eye/end_link2camera_link")
    end_link2camera_link = json.loads(end_link2camera_link.replace("'",'"'))

    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()  

    rate = rospy.Rate(20.0)
    count = 0
    while not rospy.is_shutdown():
        try:
            # 1. 获取手眼矩阵并发布TF
            (trans2, rot2) = end_link2camera_link['t'], end_link2camera_link['r']
            br.sendTransform(trans2, rot2, rospy.Time.now(), camera_link, end_link)
            count += 1
            
            # 2. 每计数20次，查找并打印base_link到marker_link的变换
            if count > 20:
                # 查找TF变换（rospy.Time(0)表示取最新可用的变换）
                (trans1, rot1) = listener.lookupTransform(base_link, marker_link, rospy.Time(0))
                print("result:%s->%s, %s,%s" % (base_link, marker_link, trans1, rot1))
                (trans2, rot2) = listener.lookupTransform(base_link, end_link, rospy.Time(0))
                print("result:%s->%s, %s,%s" % (base_link, end_link, trans2, rot2))
                (trans3, rot3) = listener.lookupTransform(base_link, camera_link, rospy.Time(0))
                print("result:%s->%s, %s,%s" % (base_link, camera_link, trans3, rot3))
                (trans4, rot4) = listener.lookupTransform(camera_link, marker_link, rospy.Time(0))
                print("result:%s->%s, %s,%s" % (camera_link, marker_link, trans4, rot4))
                count = 0
                
        # 捕获TF相关异常并打印详细信息
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as tf_e:
            # 打印TF异常类型+具体信息，不终止程序
            print(f"【TF异常】类型: {type(tf_e).__name__}, 信息: {str(tf_e)}")
            continue
        
        # 捕获所有其他致命异常（如KeyError/NameError等）并打印，可选是否终止程序
        except Exception as e:
            # 打印完整报错栈（包含出错行号），方便定位
            print(f"【致命异常】类型: {type(e).__name__}, 信息: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整的报错堆栈（含行号）
            # 可选：若想遇到致命异常终止程序，取消下面这行注释
            # rospy.signal_shutdown(f"程序因致命异常终止: {str(e)}")
            # break

        rate.sleep()