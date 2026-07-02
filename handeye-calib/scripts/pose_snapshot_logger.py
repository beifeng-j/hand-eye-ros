#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import datetime as dt
import threading
import rospy
from geometry_msgs.msg import PoseStamped

# Simple keypress listener using termios (Linux)
class KeyListener:
    def __enter__(self):
        import termios, tty
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        import termios
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def getch(self, timeout=0.1):
        import select
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if r:
            return sys.stdin.read(1)
        return None


class SnapshotLogger:
    def __init__(self, aruco_topic, end_topic, out_csv):
        self.aruco_topic = aruco_topic
        self.end_topic = end_topic
        self.out_csv = out_csv

        self.lock = threading.Lock()
        self.last_aruco = None  # type: PoseStamped
        self.last_end = None    # type: PoseStamped
        self.sample_idx = 0

        # Prepare CSV
        os.makedirs(os.path.dirname(self.out_csv) or '.', exist_ok=True)
        self.fh = open(self.out_csv, 'w', newline='')
        self.writer = csv.writer(self.fh)
        self.writer.writerow([
            'sample_idx', 'wall_time',
            'aruco_seq', 'aruco_secs', 'aruco_nsecs', 'aruco_frame',
            'aruco_px', 'aruco_py', 'aruco_pz', 'aruco_ox', 'aruco_oy', 'aruco_oz', 'aruco_ow',
            'end_seq', 'end_secs', 'end_nsecs', 'end_frame',
            'end_px', 'end_py', 'end_pz', 'end_ox', 'end_oy', 'end_oz', 'end_ow'
        ])
        self.fh.flush()

        # ROS subscribers
        self.sub1 = rospy.Subscriber(self.aruco_topic, PoseStamped, self._cb_aruco, queue_size=10)
        self.sub2 = rospy.Subscriber(self.end_topic, PoseStamped, self._cb_end, queue_size=10)

    def close(self):
        try:
            self.fh.flush()
            self.fh.close()
        except Exception:
            pass

    def _cb_aruco(self, msg: PoseStamped):
        with self.lock:
            self.last_aruco = msg

    def _cb_end(self, msg: PoseStamped):
        with self.lock:
            self.last_end = msg

    def snapshot(self):
        with self.lock:
            a = self.last_aruco
            e = self.last_end

        if a is None or e is None:
            rospy.logwarn_throttle(2.0, '等待消息中：aruco=%s, end=%s', a is not None, e is not None)
            return False

        self.sample_idx += 1
        now = dt.datetime.now().isoformat(timespec='seconds')
        row = [
            self.sample_idx, now,
            a.header.seq, a.header.stamp.secs, a.header.stamp.nsecs, a.header.frame_id,
            a.pose.position.x, a.pose.position.y, a.pose.position.z,
            a.pose.orientation.x, a.pose.orientation.y, a.pose.orientation.z, a.pose.orientation.w,
            e.header.seq, e.header.stamp.secs, e.header.stamp.nsecs, e.header.frame_id,
            e.pose.position.x, e.pose.position.y, e.pose.position.z,
            e.pose.orientation.x, e.pose.orientation.y, e.pose.orientation.z, e.pose.orientation.w,
        ]
        self.writer.writerow(row)
        self.fh.flush()
        rospy.loginfo('已记录第 %d 条 → %s', self.sample_idx, self.out_csv)
        return True


def default_csv_path():
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(os.getcwd(), 'logs', f'pose_pairs_{stamp}.csv')


def main():
    parser = argparse.ArgumentParser(description='按 r 记录一次两个PoseStamped话题到同一CSV，按 q 退出')
    parser.add_argument('--aruco_topic', default='/aruco_single/pose')
    parser.add_argument('--end_topic', default='/end_pose')
    parser.add_argument('--out', default=default_csv_path(), help='输出CSV路径，默认 logs/pose_pairs_YYYYmmdd_HHMMSS.csv')
    args, unknown = parser.parse_known_args()

    rospy.init_node('pose_snapshot_logger', anonymous=True)
    logger = SnapshotLogger(args.aruco_topic, args.end_topic, args.out)

    print('\n=== 交互说明 ===')
    print('r: 记录一次两路PoseStamped快照到 CSV')
    print('q: 退出')
    print(f'输出文件: {args.out}')
    print('================\n')

    try:
        rate = rospy.Rate(50)
        with KeyListener() as kl:
            while not rospy.is_shutdown():
                ch = kl.getch(timeout=0.1)
                if ch is not None:
                    if ch.lower() == 'r':
                        logger.snapshot()
                    elif ch.lower() == 'q':
                        print('收到 q，退出...')
                        break
                rate.sleep()
    finally:
        logger.close()

    rospy.signal_shutdown('user_quit')


if __name__ == '__main__':
    main()
