import sys
import os

# Add the parent directory of src to sys.path
# 把当前脚本的上一级目录加入 Python 搜索路径
# 目的：可以直接 import Robotic_Arm.xxx
# 常用于项目型代码而不是单文件脚本
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Robotic_Arm.rm_robot_interface import *    # 导入机械臂 SDK 的 Python 接口（运动、夹爪、正运动学等）
from avp_stream import VisionProStreamer # 1600HZ   # Vision Pro 数据流接口 ，提供手腕位姿、捏合距离等实时数据
import numpy as np
from scipy.spatial.transform import Rotation    # 姿态表示转换（旋转矩阵 / 欧拉角 / 四元数）
import multiprocessing  # 多进程（夹爪独立控制）
import time # 控制频率与延时补偿



# Define a simple Low-Pass Filter class
class LowPassFilter:
    def __init__(self, alpha, initial_value):
        """        
        :param alpha: 平滑系数,越大 → 跟随越快,越小 → 越平滑
        :param initial_value: 说明
        """
        self.alpha = alpha
        self.filtered_value = np.array(initial_value, dtype=np.float64)     # 初始输出设为机械臂初始位置

    def filter(self, new_value):
        """
        核心滤波公式: y_t = α·x_t + (1−α)·y_{t−1}
        作用：抑制 VisionPro 手部抖动，避免机械臂震荡
        
        :param new_value: 需要滤波的值
        """
        self.filtered_value = self.alpha * np.array(new_value, dtype=np.float64) + (1 - self.alpha) * self.filtered_value
        return self.filtered_value


# gripper control subprocess
# 定义一个一阶指数低通滤波器
# 用于平滑手部 → 机械臂的位移映射
def gripper_control_proc(gripper_state, robot_controller):
    """
    独立进程：只负责夹爪
    避免夹爪指令阻塞主控制循环
    :param gripper_state: 说明
    :param robot_controller: 说明
    """
    # last_state = None
    # 持续监听共享变量
    while True:
        if gripper_state.value == 1:     # 1 → 闭合夹爪（抓取）
            robot_controller.rm_set_gripper_pick(500, 200, True, 10)    # 设置抓取力度、速度等
        else:   # 张开夹爪
            robot_controller.rm_set_gripper_release(500, True, 10)  # 这里用的是进程间共享内存（multiprocessing.Value）
        # last_state = gripper_state.value

def main():
    # time.sleep(6)
    # Create a robot arm controller instance and connect to the robot arm
    # robot_controller = RobotArmController("10.20.46.135", 8080, 3)

    # ----------------创建机械臂对象-----------------
    # 使用三线程模式（通信 / 运动 / 状态）
    robot_ip = "10.20.46.135"
    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    # 连接真实机械臂
    handle = robot.rm_create_robot_arm(robot_ip, 8080)
    print("机械臂ID：", handle.id)

    # ------------visionpro------------
    # 建立 VisionPro 手部追踪数据流
    visionpro_ip = "10.12.102.110" 
    vp_pose_scale_x = 0.8
    vp_pose_scale_y = 0.8
    vp_pose_scale_z = 0.8
    # vp = VisionProStreamer(ip=visionpro_ip, record=True, frequency=40)
    vp = VisionProStreamer(ip=visionpro_ip, record=True)
    # 等待数据稳定（非常重要）
    time.sleep(1)   # 暂停0.5秒，等待VisionPro数据稳定

    # Rotatation matrix for vp2robot
    # R_Vp2Robot = np.array([
    #     [0, 0, 1],
    #     [1, 0, 0],
    #     [0, -1, 0] 
    #     ])
    # ----------------坐标系转换矩阵----------------
    # Rotatation matrix for vp2robot
    # VisionPro 坐标系 → 机械臂坐标系
    # 用于姿态变换：
    R_Vp2Robot = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1] 
        ])



    # Get API version
    print("\nAPI Version: ", rm_api_version(), "\n")

    # Perform movej motion
    # 机械臂回到一个安全、舒适的初始姿态
    #robot.rm_movej([0.113,-8.014,0.185,87.687,-0.004,64.609,-0.008], v=20, r=0, connect=0, block=1)
    robot.rm_movej([0,-45,0,75,0,50,-30], v=20, r=0, connect=0, block=1)
    robot.rm_movej([0,15,0,105,0,-30,-30], v=20, r=0, connect=0, block=1)

    # robot arm joint2pose solver
    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_75_E
    force_type = rm_force_type_e.RM_MODEL_RM_B_E
    algo_handle = Algo(arm_model, force_type)

    # 读取当前关节角
    joint_angle = robot.rm_get_joint_degree()[1]
    print(joint_angle)

    # 正运动学
    # 得到 TCP 的：
    # 位置 (x,y,z)
    # 姿态（四元数）
    robot_initial_pose = algo_handle.rm_algo_forward_kinematics(joint_angle, 0)
    print("robot_initial_pose: ", robot_initial_pose)

    # 保存“零参考位姿”
    robot_initial_pos = robot_initial_pose[:3] # 机械臂初始位置
    # wxyz to xyzw
    robot_initial_xyzw =  robot_initial_pose[3:]
    robot_initial_xyzw = [robot_initial_xyzw[1], robot_initial_xyzw[2], robot_initial_xyzw[3], robot_initial_xyzw[0]]
    # robot_initial_rot_matrix = Rotation.from_euler('xyz', robot_initial_pose[3:]).as_matrix() # 机械臂初始旋转矩阵
    robot_initial_rot_matrix = Rotation.from_quat(robot_initial_xyzw).as_matrix() # 机械臂初始旋转矩阵

    # get hand initial pose
    # --------------VisionPro 初始手势标定--------------
    # 获取手腕 4×4 齐次变换矩阵
    base_hand_pose = vp.latest["left_wrist"].squeeze(0)
    print("the shape of base_ee_pose", base_hand_pose.shape)
    # 作为 遥操作的零位参考
    # 后续所有手部运动都是 相对位移 / 相对旋转
    initial_hand_xyz = base_hand_pose[:3, 3]
    initial_hand_rotation_matrix = base_hand_pose[:3, :3]  #  for the after rotation transfor

    # gripper control
    # -------------启动夹爪子进程-------------
    # 共享变量：夹爪开 / 合
    gripper_state = multiprocessing.Value('i', 0)  # 0: open, 1: close
    gripper_proc = multiprocessing.Process(target=gripper_control_proc, args=(gripper_state, robot))
    gripper_proc.daemon = True 
    # 后台独立运行
    gripper_proc.start()

    # filter for position
    # 位移低通滤波器
    filter_alpha_pose = 0.7 # For position (x, y, z)
    lpf_pose = LowPassFilter(filter_alpha_pose, robot_initial_pos)

    # 控制参数
    # 机械臂控制频率 30Hz
    target_frequency = 30.0  # 例如30Hz
    control_period = 1.0 / target_frequency  # 每次循环的目标周期

    # 主控制循环（核心）
    while True:
        loop_start_time = time.perf_counter()  # 记录循环开始时间

        # update visionpro data
        # 1. 读取 VisionPro 手部数据
        # 得到 手部相对位移
        hand_pose_origin = vp.latest["left_wrist"].squeeze(0)
        hand_xyz = (hand_pose_origin[:3, 3] - initial_hand_xyz)
        hand_rotation_matrix = hand_pose_origin[:3, :3]

        # 2. 姿态相对变化计算
        R_rel_vp = np.dot(hand_rotation_matrix, np.linalg.inv(initial_hand_rotation_matrix))    # 计算手部在 VP 坐标系中的相对旋转
        R_rel_robot = np.dot(R_Vp2Robot, np.dot(R_rel_vp, R_Vp2Robot.T))    # 将相对旋转转换到机械臂坐标系
        #target_rot_matrix = np.dot(robot_initial_rot_matrix, R_rel_robot)   # 将转换后的相对旋转应用到机械臂初始旋转
        target_rot_matrix = np.dot(R_rel_robot, robot_initial_rot_matrix)

        # 3. 位移映射与缩放
        d_pos_raw = hand_xyz[:3]
        d_pos_scaled = np.array([
            d_pos_raw[1] * vp_pose_scale_x,
            d_pos_raw[0] * vp_pose_scale_y * -1,
            d_pos_raw[2] * vp_pose_scale_z
        ])

        # 4. 姿态修正（经验 trick）
        ee_quat_target = Rotation.from_matrix(target_rot_matrix).as_quat()  # 转换为四元数
        # 调整四元数顺序从 (x, y, z, w) 到 (w, x, y, z)
        ee_quat_target = np.array([ee_quat_target[3], ee_quat_target[0], ee_quat_target[1], ee_quat_target[2]])
        # position increment
        # 5. 位置平滑 + 目标位姿
        ee_pos_target = robot_initial_pos + d_pos_scaled

        # targetpose filter
        filter_ee_pos_target = lpf_pose.filter(ee_pos_target)
        target_pose = np.hstack([filter_ee_pos_target, ee_quat_target]) # test for quat
        print("pose_array: ",target_pose)

        # execute motion
        # 6. 发送控制指令
        # 实时笛卡尔空间控制机械臂末端
        robot.rm_movep_canfd(target_pose, False, 1, 80)

        # 7. 手势 → 夹爪
        # 捏合 → 抓取
        # 张开 → 释放
        close_gripper = vp
        close_gripper = vp.latest["left_pinch_distance"] < 0.03
        gripper_state.value = 1 if close_gripper else 0

        # 计算实际耗时并补偿延迟
        loop_duration = time.perf_counter() - loop_start_time # 250-1000HZ
        # 8. 控制频率补偿
        # 保证稳定 30Hz 控制
        sleep_time = max(0.0, control_period - loop_duration)
        time.sleep(sleep_time)  # 精确等待剩余时间
        actual_freq = 1.0 / (loop_duration + sleep_time)
        #print(loop_duration)
        #print(f"Actual frequency: {actual_freq:.2f}Hz")

if __name__ == "__main__":
    main()
