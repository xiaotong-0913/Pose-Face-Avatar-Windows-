import os, time, math, urllib.request, argparse
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker_v2_with_blendshapes/float16/latest/"
             "face_landmarker_v2_with_blendshapes.task")
MODEL_PATH = os.path.join("models", "face_landmarker_v2_with_blendshapes.task")

def ensure_model(path=MODEL_PATH, url=MODEL_URL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("下载模型中...")
        urllib.request.urlretrieve(url, path)
    return path

def ema(prev, cur, a=0.7):
    if prev is None: return cur
    return prev*a + cur*(1-a)

def get_bs_dict(result):
    d = {}
    if result.face_blendshapes:
        for c in result.face_blendshapes[0].categories:
            d[c.category_name] = float(c.score)
    return d

def draw_eye(img, cx, cy, r, open01):
    # open01: 0=闭眼, 1=睁大
    open01 = float(np.clip(open01, 0.05, 1.0))
    rx = int(r*0.55)
    ry = max(2, int(r*0.45*open01))
    center = (int(cx), int(cy))
    cv2.ellipse(img, center, (rx, ry), 0, 0, 360, (255,255,255), -1, cv2.LINE_AA)
    cv2.ellipse(img, center, (rx, ry), 0, 0, 360, (0,0,0), 2, cv2.LINE_AA)
    pr = max(2, int(r*0.20))
    cv2.circle(img, center, pr, (0,0,0), -1, cv2.LINE_AA)

def draw_face(canvas, cx, cy, R, expr):
    # expr: dict with eyeL, eyeR, smile, jaw, brow
    skin = (255,230,180)
    line = (0,0,0)

    # 头部
    cv2.circle(canvas, (int(cx), int(cy)), int(R), skin, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(cx), int(cy)), int(R), line, 2, cv2.LINE_AA)

    # 眼睛与眉毛
    eye_y = cy - int(0.22*R)
    dx = int(0.48*R)
    r_eye = int(0.16*R)
    draw_eye(canvas, cx - dx, eye_y, r_eye, expr["eyeL"])
    draw_eye(canvas, cx + dx, eye_y, r_eye, expr["eyeR"])

    # 眉毛高度受 brow 提升
    brow_y_base = eye_y - int(0.35*R)
    brow_up = int(expr["brow"] * 0.18 * R)
    thick = max(2, int(0.09*R))
    cv2.line(canvas, (cx - dx - int(0.4*R), brow_y_base - brow_up),
                      (cx - dx + int(0.4*R), brow_y_base - brow_up), line, thick, cv2.LINE_AA)
    cv2.line(canvas, (cx + dx - int(0.4*R), brow_y_base - brow_up),
                      (cx + dx + int(0.4*R), brow_y_base - brow_up), line, thick, cv2.LINE_AA)

    # 嘴巴：smile 控制弧度，jaw 控制开口高度
    mcy = cy + int(0.38*R)
    mw = int(0.85*R)
    mh = max(2, int(R*(0.08 + 0.35*expr["jaw"])))
    # 曲率：越笑越上扬
    start = 200 - int(40*expr["smile"])
    end   = -20 + int(40*expr["smile"])
    cv2.ellipse(canvas, (int(cx), int(mcy)), (mw//2, mh), 0, start, 180-end, line, 4, cv2.LINE_AA)
    if expr["jaw"] > 0.35:
        # 画开口的内部
        cv2.ellipse(canvas, (int(cx), int(mcy)), (mw//2-3, mh-3), 0, 0, 360, (40,40,40), -1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 摄像头或视频路径")
    ap.add_argument("--bg", choices=["white","black"], default="white")
    ap.add_argument("--mirror", action="store_true", help="镜像显示")
    args = ap.parse_args()

    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    if not cap.isOpened():
        print("无法打开视频源"); return

    prev_center = None
    prev_R = None
    prev_expr = None
    bg = (255,255,255) if args.bg=="white" else (0,0,0)
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        canvas = np.full((H, W, 3), bg, dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time()-t0)*1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks:
            cv2.imshow("FaceAvatar", canvas)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]: break
            continue

        # 估计头部中心与半径（用关键点包围盒）
        lms = result.face_landmarks[0]
        xs = np.array([min(max(p.x,0.0),1.0)*W for p in lms], dtype=np.float32)
        ys = np.array([min(max(p.y,0.0),1.0)*H for p in lms], dtype=np.float32)
        minx, maxx = float(xs.min()), float(xs.max())
        miny, maxy = float(ys.min()), float(ys.max())
        cx, cy = (minx+maxx)/2.0, (miny+maxy)/2.0
        R = (maxx-minx) * 0.55  # 半径与脸宽成比例

        # 表情参数（0..1）
        bs = get_bs_dict(result)
        expr_now = {
            "eyeL": max(0.0, 1.0 - bs.get("eyeBlinkLeft", 0.0)),
            "eyeR": max(0.0, 1.0 - bs.get("eyeBlinkRight", 0.0)),
            "smile": max(bs.get("mouthSmileLeft",0.0), bs.get("mouthSmileRight",0.0)),
            "jaw": bs.get("jawOpen", 0.0),
            "brow": min(1.0, (bs.get("browInnerUp",0.0)
                              +bs.get("browOuterUpLeft",0.0)
                              +bs.get("browOuterUpRight",0.0))/3.0)
        }

        # 平滑
        center = ema(prev_center, np.array([cx,cy], dtype=np.float32), a=0.7)
        R = float(ema(prev_R, np.array([R], dtype=np.float32), a=0.7))
        if prev_expr is None:
            expr = expr_now
        else:
            expr = {k: float(ema(np.array([prev_expr[k]], dtype=np.float32),
                                 np.array([expr_now[k]], dtype=np.float32), a=0.7))
                    for k in expr_now.keys()}
        prev_center, prev_R, prev_expr = center, R, expr

        draw_face(canvas, center[0], center[1], R, expr)

        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(canvas, "Face Avatar | q/ESC退出", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0) if args.bg=="white" else (255,255,255), 2)
        cv2.imshow("FaceAvatar", canvas)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
