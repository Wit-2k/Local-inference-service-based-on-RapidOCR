import os
import time

import cv2

os.makedirs("frames", exist_ok=True)

# 目前一次只能识别一个芯片

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 通常表示默认摄像头

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    # 用最高分辨率保证识别成功
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("当前宽度:", width)
    print("当前高度:", height)
    print("当前帧率:", fps)

    last_save_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取画面失败")
                break

            now = time.time()

            if now - last_save_time >= 1:
                filename = time.strftime("frames/%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(filename, frame)
                print(f"保存:{filename}")
                last_save_time = now

            # 如果不需要预览，可以删掉下面这几行
            cv2.imshow("camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("❌ 被手动打断")

    cap.release()
    cv2.destroyAllWindows()
