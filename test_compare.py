from ultralytics import YOLO

if __name__ == '__main__':
    # 你的考题：测试视频的绝对路径
    source_path = r"D:\python\AIproject\Detect-YOLO V11\TRAIN2-person-car\test-video.MOV"

    print("=== 阶段一：用旧模型生成【反面教材】 ===")
    # 加载旧模型 (注意路径已更新为D盘)
    old_model_path = r"D:\python\AIproject\Detect-YOLO V11\TRAIN2-person-car\pythonProject\runs\detect\train\weights\best.pt"
    model_old = YOLO(old_model_path)
    # 存入专属比对文件夹
    model_old.predict(source=source_path, save=True, conf=0.15, imgsz=1280, project="D:/期末对比", name="1_旧模型_原版效果")

    print("=== 阶段二：用新模型生成【降维打击】效果 ===")
    # 加载刚刚新鲜出炉的新模型
    new_model_path = r"D:\python\AIproject\Detect-YOLO V11\TRAIN3-VisDrone\runs\detect\train2\weights\best.pt"
    model_new = YOLO(new_model_path)
    # 存入专属比对文件夹
    model_new.predict(source=source_path, save=True, conf=0.15, imgsz=1280, project="D:/期末对比", name="2_新模型_VisDrone强化版")

    print("测试全部搞定！请前往 D盘 的 '期末对比' 文件夹查收战果！")
