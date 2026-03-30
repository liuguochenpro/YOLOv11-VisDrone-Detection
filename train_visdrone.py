from ultralytics import YOLO
import os

# 解决 OpenMP 环境变量冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # 加载官方基础模型作为强大的底座
    model = YOLO("yolo11n.pt")

    # 精准指向你刚才配置好的 yaml 文件（注意这里保留了前面的 r 和 Windows 默认的反斜杠）
    yaml_path = r"D:\python\AIproject\Detect-YOLO V11\TRAIN3-VisDrone\VisDrone_Dataset\visdrone.yaml"

    print("真·工业级无人机数据集已挂载，开启 15 轮极速炼丹！")

    # 火力全开：15 轮速通，应对咱们的食堂俯拍视频绝对足够
    results = model.train(data=yaml_path, epochs=15, imgsz=640, batch=8, workers=0)

    print("15 轮训练圆满结束！")
