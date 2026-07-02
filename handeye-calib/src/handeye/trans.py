import csv
from scipy.spatial.transform import Rotation as R

# 你的原始数据文件
input_file = '/home/ds/g1/songling_ws/src/handeye-calib/src/handeye-calib/config/base_hand_on_eye_test_data.csv'
# 生成的新数据文件
output_file = '/home/ds/g1/songling_ws/src/handeye-calib/src/handeye-calib/config/output_6d2.csv'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        
        if parts[0] == 'eye':
            # 提取平移和四元数 (ROS 标准顺序通常是 qx, qy, qz, qw)
            x, y, z = parts[1:4]
            qx, qy, qz, qw = parts[4:8]
            
            # 使用 scipy 将四元数转为欧拉角
            rot = R.from_quat([float(qx), float(qy), float(qz), float(qw)])
            rx, ry, rz = rot.as_euler('xyz', degrees=True)
            
            # 拼接成 6 元数格式，平移保持原样(m)，欧拉角保留3位小数
            new_line = f"eye,{x},{y},{z},{rx:.3f},{ry:.3f},{rz:.3f}\n"
            f_out.write(new_line)
            
        elif parts[0] == 'hand':
            # 将 hand 的前三个数值(平移)从 mm 转为 m
            x_m = float(parts[1]) / 1000.0
            y_m = float(parts[2]) / 1000.0
            z_m = float(parts[3]) / 1000.0
            
            # 后三个数值(欧拉角)保持不变
            rx, ry, rz = parts[4], parts[5], parts[6]
            
            # 拼接成新的 hand 数据，平移保留6位小数以免丢失精度
            new_line = f"hand,{x_m:.6f},{y_m:.6f},{z_m:.6f},{rx},{ry},{rz}\n"
            f_out.write(new_line)

print(f"转换完成！结果已保存到 {output_file}")
