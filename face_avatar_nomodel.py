import argparse, time, math
import cv2, numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 关键点索引（468网格）
# 眼睛：上下睑
L_EYE_UP, L_EYE_DN = 159, 145
R_EYE_UP, R_EYE_DN = 386, 374
# 眼角
L_EYE_OUT, L_EYE_IN = 33, 133
R_EYE_IN,  R_EYE_OUT = 362, 263
# 嘴：上下内唇 + 嘴角
MOUTH_UP, MOUTH_DN = 13, 14
MOUTH_L,  MOUTH_R  = 61, 291
# 眉（粗略）
LBROW, RBROW = 105, 334
# 鼻尖
NOSE = 1

def dist(a,b):
    return float(np.linalg.norm(a-b))

def clamp(x,a,b): return max(a, min(b, x))

def smooth(prev, cur, a=0.7):
    if prev is None: return cur
    return a*prev + (1-a)*cur

def draw_eye(canvas, c, R, open01):
    rx = int(R*0.55); ry = max(2, int(R*0.45*open01))
    c = (int(c[0]), int(c[1]))
    cv2.ellipse(canvas, c, (rx, ry), 0, 0, 360, (255,255,255), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, c, (rx, ry), 0, 0, 360, (0,0,0), 2, cv2.LINE_AA)
    cv2.circle(canvas, c, max(2,int(R*0.2)), (0,0,0), -1, cv2.LINE_AA)

def draw_face(canvas, center, R, expr):
    skin=(255,230,180); line=(0,0,0)
    cx,cy=int(center[0]),int(center[1])
    cv2.circle(canvas,(cx,cy),int(R),skin,-1,cv2.LINE_AA)
    cv2.circle(canvas,(cx,cy),int(R),line,2,cv2.LINE_AA)

    eye_y = cy - int(0.22*R); dx = int(0.48*R); r_eye = int(0.16*R)
    draw_eye(canvas,(cx-dx,eye_y),r_eye,expr["eyeL"])
    draw_eye(canvas,(cx+dx,eye_y),r_eye,expr["eyeR"])

    brow_y_base = eye_y - int(0.35*R)
    brow_up = int(expr["brow"] * 0.18 * R)
    thick = max(2,int(0.09*R))
    cv2.line(canvas,(cx-dx-int(0.4*R),brow_y_base-brow_up),
                    (cx-dx+int(0.4*R),brow_y_base-brow_up),line,thick,cv2.LINE_AA)
    cv2.line(canvas,(cx+dx-int(0.4*R),brow_y_base-brow_up),
                    (cx+dx+int(0.4*R),brow_y_base-brow_up),line,thick,cv2.LINE_AA)

    mcy = cy + int(0.38*R)
    mw = int(0.85*R)
    mh = max(2, int(R*(0.08 + 0.35*expr["jaw"])))
    start = 200 - int(40*expr["smile"])
    end   = -20 + int(40*expr["smile"])
    cv2.ellipse(canvas,(cx,mcy),(mw//2,mh),0,start,180-end,line,4,cv2.LINE_AA)
    if expr["jaw"]>0.35:
        cv2.ellipse(canvas,(cx,mcy),(mw//2-3,mh-3),0,0,360,(40,40,40),-1,cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--bg", choices=["white","black"], default="white")
    ap.add_argument("--mirror", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    if not cap.isOpened():
        print("无法打开视频源"); return
    bg = (255,255,255) if args.bg=="white" else (0,0,0)

    prev_center=None; prev_R=None; prev_expr=None

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as fm:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if args.mirror: frame = cv2.flip(frame,1)
            H,W = frame.shape[:2]
            canvas = np.full((H,W,3), bg, dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                cv2.imshow("FaceAvatar", canvas)
                if cv2.waitKey(1)&0xFF in [27,ord('q')]: break
                continue

            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([(lm[i].x*W, lm[i].y*H) for i in range(468)], dtype=np.float32)

            # 归一化尺度：瞳距
            eye_base = dist(pts[L_EYE_OUT], pts[R_EYE_OUT]) + 1e-6

            # 眼睛开合（0..1）
            eyeL = clamp((dist(pts[L_EYE_UP], pts[L_EYE_DN]) / eye_base) * 4.5, 0.0, 1.0)
            eyeR = clamp((dist(pts[R_EYE_UP], pts[R_EYE_DN]) / eye_base) * 4.5, 0.0, 1.0)

            # 嘴张开（0..1）
            jaw = clamp((dist(pts[MOUTH_UP], pts[MOUTH_DN]) / eye_base) * 3.0, 0.0, 1.0)

            # 微笑：嘴角外拉程度（0..1）
            mouth_w = dist(pts[MOUTH_L], pts[MOUTH_R]) / eye_base
            smile = clamp((mouth_w - 1.6) * 1.5, 0.0, 1.0)

            # 眉提升：眉到眼的距离
            brow = clamp(((pts[LBROW][1]-pts[L_EYE_UP][1]) + (pts[RBROW][1]-pts[R_EYE_UP][1]))/2.0
                         / (eye_base*0.18) - 0.6, 0.0, 1.0)

            # 头中心与半径
            minx,maxx = float(pts[:,0].min()), float(pts[:,0].max())
            miny,maxy = float(pts[:,1].min()), float(pts[:,1].max())
            cx,cy = (minx+maxx)/2.0, (miny+maxy)/2.0
            R = (maxx-minx)*0.55

            center = smooth(prev_center, np.array([cx,cy],dtype=np.float32), a=0.7)
            R = float(smooth(prev_R, np.array([R],dtype=np.float32), a=0.7))
            expr_now={"eyeL":eyeL,"eyeR":eyeR,"jaw":jaw,"smile":smile,"brow":brow}
            if prev_expr is None: expr=expr_now
            else: expr={k: float(smooth(np.array([prev_expr[k]]), np.array([expr_now[k]]), a=0.7)) for k in expr_now}
            prev_center, prev_R, prev_expr = center, R, expr

            draw_face(canvas, center, R, expr)
            cv2.putText(canvas,"Face Avatar | q/ESC退出",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0) if args.bg=="white" else (255,255,255),2)
            cv2.imshow("FaceAvatar", canvas)
            if cv2.waitKey(1)&0xFF in [27,ord('q')]: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
