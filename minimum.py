import time

import cv2
import gradio as gr


def camera_stream():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV 读取的是 BGR，Gradio 显示需要 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield frame_rgb

        # 控制显示帧率
        # time.sleep(0.033)  # 约 30 FPS

    cap.release()


with gr.Blocks() as demo:
    gr.Markdown("## OpenCV 摄像头预览")

    image_output = gr.Image(label="摄像头画面", type="numpy")

    start_btn = gr.Button("启动摄像头")

    start_btn.click(fn=camera_stream, inputs=None, outputs=image_output)


if __name__ == "__main__":
    demo.launch()
