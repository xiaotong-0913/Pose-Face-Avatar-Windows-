import argparse
import time
import csv
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# COCO 17 关键点名称，按模型输出顺序
KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

def is_camera(src_str: str) -> bool:
    # "0","1" 视为摄像头
    return src_str.isdigit()

def open_writer(csv_path: str):
    if not csv_path:
        return None, None
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    f = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    header = ["frame","person"]
    for n in KP_NAMES:
        header += [f"{n}_x", f"{n}_y", f"{n}_c"]
    w.writerow(header)
    return f, w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 表示摄像头，也可填写视频/图片路径")
    ap.add_argument("--model", default="yolov8n-pose.pt", help="Ultralytics pose 模型")
    ap.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    ap.add_argument("--width", type=int, default=1280, help="摄像头宽")
    ap.add_argument("--height", type=int, default=720, help="摄像头高")
    ap.add_argument("--save", type=str, default="", help="将关键点保存到 CSV 的路径，如 poses.csv")
    ap.add_argument("--save_vid", type=str, default="", help="保存带骨架的视频，如 out.mp4")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model).to(device)

    src_is_cam = is_camera(str(args.source))
    cap = cv2.VideoCapture(int(args.source)) if src_is_cam else cv2.VideoCapture(args.source)
    if src_is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("无法打开视频源")
        return

    # 初始化视频写入
    ret, frame0 = cap.read()
    if not ret:
        print("无法读取首帧")
        return
    h, w = frame0.shape[:2]
    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    fps_cap = fps_cap if fps_cap and fps_cap > 1 else 30
    out_vid = None
    if args.save_vid:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(args.save_vid, fourcc, fps_cap, (w, h))

    # 初始化 CSV
    csv_file, writer = open_writer(args.save)

    frame_idx = 0
    # 先把首帧放回处理
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        # 推理
        results = model.predict(frame, conf=args.conf, verbose=False)
        res = results[0]

        # 可视化
        annotated = res.plot()

        # FPS
        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(
            annotated, f"FPS {fps:.1f} | {device}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # 写 CSV
        if writer and res.keypoints is not None:
            kps = res.keypoints  # ultralytics.engine.results.Keypoints
            xy = kps.xy  # (n, 17, 2) Tensor
            conf = kps.conf  # (n, 17) or (n, 17, 1)
            if xy is not None:
                if hasattr(xy, "cpu"):
                    xy = xy.cpu().numpy()
                else:
                    xy = np.asarray(xy)
                if conf is not None:
                    if hasattr(conf, "cpu"):
                        conf = conf.cpu().numpy()
                    else:
                        conf = np.asarray(conf)
                    conf = conf.reshape(conf.shape[0], conf.shape[1])  # 保证 (n,17)
                else:
                    conf = np.zeros((xy.shape[0], xy.shape[1]), dtype=np.float32)

                num_person = xy.shape[0]
                for pid in range(num_person):
                    row = [frame_idx, pid]
                    for j in range(xy.shape[1]):
                        x, y = float(xy[pid, j, 0]), float(xy[pid, j, 1])
                        c = float(conf[pid, j])
                        row += [x, y, c]
                    writer.writerow(row)

        # 显示与保存视频
        cv2.imshow("Pose", annotated)
        if out_vid is not None:
            out_vid.write(annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if out_vid is not None:
        out_vid.release()
    if csv_file is not None:
        csv_file.close()
        print(f"关键点已保存到: {args.save}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
