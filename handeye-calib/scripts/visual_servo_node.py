#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import geometry_msgs.msg
import numpy as np
from scipy.spatial.transform import Rotation as R

class HandEyeCalibrationNode:
    def __init__(self):
        rospy.init_node('hand_eye_calibration_controller')
        
        # 使用您提供的具体手眼标定参数（眼在手上），即 摄像头 相对于 末端 的位姿：T_end->cam
        # 如需在参数服务器中覆盖，请设置 ~handeye_{x,y,z,rx,ry,rz}
        hx = rospy.get_param('~handeye_x', -0.0868662)
        hy = rospy.get_param('~handeye_y', 0.0099093)
        hz = rospy.get_param('~handeye_z', 0.0416863)
        hrx = rospy.get_param('~handeye_rx_deg', -23.4421)
        hry = rospy.get_param('~handeye_ry_deg', 2.21775)
        hrz = rospy.get_param('~handeye_rz_deg', -85.4754)

        self.T_end_to_cam = self.create_transform_matrix(
            x=hx, y=hy, z=hz,
            rx=hrx, ry=hry, rz=hrz
        )
        
        # TF广播器和监听器
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 订阅标定板位姿
        self.marker_pose = None
        self.pose_sub = rospy.Subscriber('/aruco_single/pose', geometry_msgs.msg.PoseStamped, self.marker_pose_callback)
        
        # 发布“标定板在基座坐标系下”的位姿
        self.marker_in_base_pub = rospy.Publisher('/target_pose_base_frame', geometry_msgs.msg.PoseStamped, queue_size=10)
        
        # 坐标系名称 - 请根据您的实际设置修改
        self.base_frame = rospy.get_param('~base_frame', 'base_link')  # 机械臂基座坐标系
        self.end_effector_frame = rospy.get_param('~ee_frame', 'link6')  # 机械臂末端坐标系
        self.marker_child_frame = rospy.get_param('~marker_child_frame', 'marker_in_base')
        
        rospy.loginfo("手眼标定控制节点已启动")
        rospy.loginfo(f"基座坐标系: {self.base_frame}")
        rospy.loginfo(f"末端坐标系: {self.end_effector_frame}")
        rospy.loginfo("模式: 眼在手上 (T_end->cam 已加载)")
        
    def create_transform_matrix(self, x, y, z, rx, ry, rz, degrees=True, euler_seq='xyz'):
        """根据平移和欧拉角创建4x4变换矩阵

        参数:
        - x, y, z: 平移(米)
        - rx, ry, rz: 欧拉角(默认单位为度)，顺序由 euler_seq 指定，默认 'xyz' (roll, pitch, yaw)
        - degrees: True 表示角度为度，False 表示弧度
        - euler_seq: 欧拉角顺序，默认 'xyz'，按常用 RPY 约定
        """
        T = np.eye(4)

        # 设置平移
        T[0:3, 3] = [x, y, z]

        # 设置旋转（欧拉角 -> 旋转矩阵）
        rotation = R.from_euler(euler_seq, [rx, ry, rz], degrees=degrees)
        T[0:3, 0:3] = rotation.as_matrix()

        # 记录信息，便于核对
        rospy.loginfo("创建的手眼变换矩阵：")
        rospy.loginfo(f"平移: [{x:.6f}, {y:.6f}, {z:.6f}]")
        rospy.loginfo(f"欧拉角({euler_seq}, degrees={degrees}): [{rx:.6f}, {ry:.6f}, {rz:.6f}]")
        quat = rotation.as_quat()  # [qx, qy, qz, qw]
        rospy.loginfo(f"对应四元数[x,y,z,w]: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
        try:
            matrix_str = "\n".join(
                ["    " + "  ".join([f"{v: .6f}" for v in row]) for row in T]
            )
            rospy.loginfo("T_base_to_marker 4x4矩阵:\n" + matrix_str)
        except Exception as e:
            rospy.logwarn(f"矩阵输出失败: {e}")

        return T
    
    def pose_to_matrix(self, pose_msg):
        """将Pose消息转换为4x4变换矩阵"""
        T = np.eye(4)
        
        # 设置平移
        T[0:3, 3] = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
        
        # 设置旋转
        quat = [
            pose_msg.orientation.x,
            pose_msg.orientation.y, 
            pose_msg.orientation.z,
            pose_msg.orientation.w
        ]
        rotation = R.from_quat(quat)
        T[0:3, 0:3] = rotation.as_matrix()
        
        return T
    
    def matrix_to_pose(self, T):
        """将4x4变换矩阵转换为Pose消息"""
        rotation = R.from_matrix(T[0:3, 0:3])
        quat = rotation.as_quat()  # [qx, qy, qz, qw]
        position = T[0:3, 3]
        
        pose = geometry_msgs.msg.Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        return pose
    
    def transform_to_matrix(self, transform):
        """将geometry_msgs/Transform转换为4x4矩阵"""
        T = np.eye(4)
        T[0:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]
        quat = [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        ]
        rotation = R.from_quat(quat)
        T[0:3, 0:3] = rotation.as_matrix()
        return T
    
    def get_transform(self, target_frame, source_frame):
        """获取两个坐标系之间的变换"""
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
            return self.transform_to_matrix(transform.transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"无法获取从 {source_frame} 到 {target_frame} 的变换: {e}")
            return None
    
    def marker_pose_callback(self, msg):
        """标定板位姿回调函数"""
        self.marker_pose = msg
        self.calculate_marker_pose_in_base()
    
    def calculate_marker_pose_in_base(self):
        """计算并发布：标定板在基座坐标系下的位姿

        眼在手上：已知
        - T_base->end（来自TF）
        - T_end->cam（手眼标定结果）
        - T_cam->marker（相机检测结果 /aruco_single/pose）

        则：T_base->marker = T_base->end * T_end->cam * T_cam->marker
        """
        if self.marker_pose is None:
            return
        
        try:
            # 获取基座到末端的变换
            T_base_to_end = self.get_transform(self.base_frame, self.end_effector_frame)
            if T_base_to_end is None:
                rospy.logwarn("无法获取基座到末端的变换，请检查TF树")
                return
            
            # 将标定板位姿（相机坐标系下）转换为变换矩阵：T_cam->marker
            T_cam_to_marker = self.pose_to_matrix(self.marker_pose.pose)
            
            rospy.loginfo_throttle(5, f"检测到标定板（相机系）位姿: "
                                      f"[{self.marker_pose.pose.position.x:.3f}, "
                                      f"{self.marker_pose.pose.position.y:.3f}, "
                                      f"{self.marker_pose.pose.position.z:.3f}]")

            # 计算基座到标定板的变换
            T_base_to_marker = T_base_to_end @ self.T_end_to_cam @ T_cam_to_marker

            # 创建位姿消息（在基座坐标系中）
            marker_in_base = geometry_msgs.msg.PoseStamped()
            marker_in_base.header.stamp = rospy.Time.now()
            marker_in_base.header.frame_id = self.base_frame

            # 将变换矩阵转换为位姿
            marker_in_base.pose = self.matrix_to_pose(T_base_to_marker)

            # 发布标定板位姿
            self.marker_in_base_pub.publish(marker_in_base)

            # 广播TF变换用于可视化
            self.broadcast_marker_tf(marker_in_base)

            # 详细输出：平移、四元数、欧拉角(度)
            pos = marker_in_base.pose.position
            quat = [
                marker_in_base.pose.orientation.x,
                marker_in_base.pose.orientation.y,
                marker_in_base.pose.orientation.z,
                marker_in_base.pose.orientation.w,
            ]
            rpy_deg = R.from_quat(quat).as_euler('xyz', degrees=True)

            rospy.loginfo(
                (
                    "标定板->基座 位姿(基座系):\n"
                    f"  平移 [m]    : x={pos.x:.4f}, y={pos.y:.4f}, z={pos.z:.4f}\n"
                    f"  四元数 [x,y,z,w]: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]\n"
                    f"  欧拉角[deg] (xyz/RPY): roll={rpy_deg[0]:.2f}, pitch={rpy_deg[1]:.2f}, yaw={rpy_deg[2]:.2f}"
                )
            )
            
        except Exception as e:
            rospy.logerr(f"计算标定板在基座系位姿时出错: {e}")
    
    def broadcast_marker_tf(self, pose_msg):
        """广播：基座系 -> 标定板 的TF变换"""
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = pose_msg.header.stamp
        t.header.frame_id = pose_msg.header.frame_id
        t.child_frame_id = self.marker_child_frame
        
        t.transform.translation.x = pose_msg.pose.position.x
        t.transform.translation.y = pose_msg.pose.position.y
        t.transform.translation.z = pose_msg.pose.position.z
        
        t.transform.rotation = pose_msg.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)
    
    def run(self):
        """主循环"""
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        node = HandEyeCalibrationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass