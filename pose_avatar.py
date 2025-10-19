import argparse
import math
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# COCO 关键点索引
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_EL=7; R_EL=8; L_WR=9; R_WR=10
L_HIP=11; R_HIP=12; L_KNEE=13; R_KNEE=14; L_ANK=15; R_ANK=16
KP_NUM = 17

def pick_main_person(xy, conf):
    """选择平均置信度最高的人"""
    if xy is None or len(xy)==0: return None
    scores = conf.mean(axis=1) if conf is not None else np.ones(len(xy))
    idx = int(scores.argmax())
    return xy[idx], (conf[idx] if conf is not None else np.ones(xy.shape[1]))

def ema(prev, curr, alpha=0.7):
    if prev is None: return curr
    return alpha*prev + (1-alpha)*curr

def pt_ok(c, thr=0.35):
    return c is not None and c > thr

def draw_limb(img, p1, p2, color, thickness=18):
    if p1 is None or p2 is None: return
    p1 = (int(p1[0]), int(p1[1])); p2 = (int(p2[0]), int(p2[1]))
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.circle(img, p1, thickness//2, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, thickness//2, color, -1, cv2.LINE_AA)

def draw_torso(img, l_sh, r_sh, r_hip, l_hip, color):
    pts = []
    for p in [l_sh, r_sh, r_hip, l_hip]:
        if p is None: return
        pts.append([int(p[0]), int(p[1])])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color, lineType=cv2.LINE_AA)

def safe_pt(xy, conf, idx, thr=0.35):
    if xy is None: return None
    if conf is None: return tuple(xy[idx])
    return tuple(xy[idx]) if conf[idx] >= thr else None

def normalize_to_canvas(kp_xy, kp_conf, W, H):
    """把骨架平移缩放到画布中心，保持稳定大小"""
    if kp_xy is None: return None, None
    # 根节点与尺度
    roots = []
    for a,b in [(L_HIP,R_HIP),(L_SH,R_SH)]:
        pa = kp_xy[a] if kp_conf[a]>0 else None
        pb = kp_xy[b] if kp_conf[b]>0 else None
        if pa is not None and pb is not None:
            roots.append((pa+pb)/2)
    if roots:
        root = np.mean(roots, axis=0)
    else:
        root = kp_xy[NOSE]
    # 肩宽
    if kp_conf[L_SH]>0 and kp_conf[R_SH]>0:
        shoulder_w = np.linalg.norm(kp_xy[L_SH]-kp_xy[R_SH])
    elif kp_conf[L_HIP]>0 and kp_conf[R_HIP]>0:
        shoulder_w = np.linalg.norm(kp_xy[L_HIP]-kp_xy[R_HIP]) * 1.2
    else:
        shoulder_w = 200.0
    target = min(W,H) * 0.30
    s = target / max(shoulder_w, 1.0)

    center = np.array([W/2, H/2], dtype=np.float32)
    xy_norm = (kp_xy - root) * s + center
    return xy_norm, kp_conf

def draw_head(img, xy, conf):
    # 头中心与半径
    if conf[L_EYE]>0 and conf[R_EYE]>0:
        eye_c = (xy[L_EYE] + xy[R_EYE]) / 2
        radius = int(max(12, np.linalg.norm(xy[L_EYE]-xy[R_EYE]) * 0.9))
        center = (int(eye_c[0]), int(eye_c[1]-radius*0.3))
    else:
        nose = xy[NOSE] if conf[NOSE]>0 else None
        if nose is None: return
        center = (int(nose[0]), int(nose[1]-25))
        radius = 25
    # 脸
    cv2.circle(img, center, int(radius*1.2), (255, 220, 180), -1, cv2.LINE_AA)
    # 眼睛
    eye_off = int(radius*0.35)
    eye_r = max(2, int(radius*0.12))
    cv2.circle(img, (center[0]-eye_off, center[1]-eye_off//2), eye_r, (0,0,0), -1, cv2.LINE_AA)
    cv2.circle(img, (center[0]+eye_off, center[1]-eye_off//2), eye_r, (0,0,0), -1, cv2.LINE_AA)
    # 嘴
    cv2.ellipse(img, (center[0], center[1]+eye_off//2), (eye_off, eye_r), 0, 10, 170, (0,0,0), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 摄像头或视频路径")
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--bg", choices=["white","black"], default="white")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model).to(device)

    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    if not cap.isOpened():
        print("无法打开视频源"); return

    prev_xy = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        H, W = frame.shape[:2]
        # 仅绘制卡通，背景纯色
        bg_color = (255,255,255) if args.bg=="white" else (0,0,0)
        canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)

        t0 = time.time()
        res = model.predict(frame, conf=args.conf, verbose=False)[0]

        xy = res.keypoints.xy if res.keypoints is not None else None
        conf = res.keypoints.conf if res.keypoints is not None else None
        if xy is not None:
            if hasattr(xy, "cpu"): xy = xy.cpu().numpy()
            if hasattr(conf, "cpu"): conf = conf.cpu().numpy()
            person = pick_main_person(xy, conf)
        else:
            person = None

        if person is None:
            cv2.imshow("Avatar", canvas)
            if cv2.waitKey(1)&0xFF in [27, ord('q')]: break
            continue

        kp_xy, kp_conf = person
        kp_xy, kp_conf = normalize_to_canvas(kp_xy, kp_conf, W, H)
        kp_xy = ema(prev_xy, kp_xy, alpha=0.7); prev_xy = kp_xy

        # 取点（不足则 None）
        pts = [tuple(kp_xy[i]) if kp_conf[i]>0 else None for i in range(KP_NUM)]

        # 躯干
        draw_torso(canvas, pts[L_SH], pts[R_SH], pts[R_HIP], pts[L_HIP], (200, 160, 255))
        # 四肢
        draw_limb(canvas, pts[L_SH], pts[L_EL], (140, 80, 200))
        draw_limb(canvas, pts[L_EL], pts[L_WR], (140, 80, 200))
        draw_limb(canvas, pts[R_SH], pts[R_EL], (140, 80, 200))
        draw_limb(canvas, pts[R_EL], pts[R_WR], (140, 80, 200))
        draw_limb(canvas, pts[L_HIP], pts[L_KNEE], (80, 160, 240))
        draw_limb(canvas, pts[L_KNEE], pts[L_ANK], (80, 160, 240))
        draw_limb(canvas, pts[R_HIP], pts[R_KNEE], (80, 160, 240))
        draw_limb(canvas, pts[R_KNEE], pts[R_ANK], (80, 160, 240))
        # 颈部与头
        if pts[L_SH] and pts[R_SH]:
            mid_sh = ((pts[L_SH][0]+pts[R_SH][0])//2, (pts[L_SH][1]+pts[R_SH][1])//2)
            if pts[NOSE]:
                draw_limb(canvas, mid_sh, pts[NOSE], (200, 160, 255), thickness=14)
        draw_head(canvas, kp_xy, kp_conf)

        fps = 1.0 / max(time.time()-t0, 1e-6)
        cv2.putText(canvas, f"FPS {fps:.1f} | {device}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0) if args.bg=="white" else (255,255,255), 2)

        cv2.imshow("Avatar", canvas)
        if cv2.waitKey(1)&0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
