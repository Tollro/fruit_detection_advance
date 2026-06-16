"""
对水果数据集进行完整验证，输出所有 Ultralytics 标准产物
数据集路径: D:/AI_train_Datasets/FruitsDataset/combine
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# ================= 配置区 =================
DATA_ROOT = r"D:/AI_train_Datasets/FruitsDataset/combine"
MODEL_PATH = r"runs/detect/embed/exp1/weights/last.pt"          # 如果存在则直接用，否则会自动训练一个临时模型
TRAIN_IF_NO_MODEL = False        # 没有模型时是否自动训练（仅1 epoch）
EPOCHS_IF_TRAIN = 1
IMG_SIZE = 640
# ==========================================

def create_dataset_yaml(root):
    """根据目录结构生成 dataset.yaml"""
    yaml_path = Path(root) / "dataset.yaml"
    if yaml_path.exists():
        print(f"✅ 发现已有配置文件: {yaml_path}")
        return yaml_path

    content = f"""path: {root}  # 数据集的根目录绝对路径
train:
  - images/train
val: images/val      # 验证集图像路径，相对于 `path`

# 自动推导出标注路径的规则：
# 训练标注路径 = path + 'labels/train'
# 验证标注路径 = path + 'labels/val'

# 类别数量
nc: 9
# 类别名称
names:
  0: Apple
  1: Watermelon
  2: Orange
  3: Banana
  4: Strawberry
  5: Kiwifruit
  6: Pineapple
  7: Durian
  8: Pitaya
"""
    yaml_path.write_text(content, encoding='utf-8')
    print(f"📝 已自动生成数据集配置文件: {yaml_path}")
    return yaml_path

def print_normalized_confusion_matrix(metrics):
    """
    从验证 metrics 中提取混淆矩阵并打印归一化版本（每行和为1）
    """
    try:
        # 获取混淆矩阵（原始计数） shape: (nc+1, nc+1)，最后一行/列是背景
        cm = metrics.confusion_matrix.matrix
        nc = cm.shape[0] - 1   # 实际类别数
        cm = cm[:nc, :nc]      # 去掉背景
        cm = cm.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        # 避免除以零
        row_sums[row_sums == 0] = 1
        normalized_cm = cm / row_sums

        # 类别名称
        if hasattr(metrics, 'names') and metrics.names:
            names = metrics.names
        else:
            names = {i: f'Class {i}' for i in range(nc)}

        # 打印表格
        header = ["真实\\预测"] + [names[i] for i in range(nc)]
        col_width = max(len(h) for h in header) + 2
        print("\n📊 归一化混淆矩阵 (每行和为1)")
        print("".join(f"{h:>{col_width}}" for h in header))
        for i in range(nc):
            row = [f"{names[i]:>{col_width-2}}"] + [f"{normalized_cm[i, j]:.2f}" for j in range(nc)]
            print("".join(f"{r:>{col_width}}" for r in row))
    except Exception as e:
        print(f"⚠️ 无法打印混淆矩阵: {e}")

def main():
    # 1. 准备数据集 yaml
    data_yaml = create_dataset_yaml(DATA_ROOT)

    # 2. 准备模型
    model_file = Path(MODEL_PATH)
    if model_file.exists():
        print(f"✅ 使用已有模型: {model_file}")
        model = YOLO(str(model_file))
    else:
        if TRAIN_IF_NO_MODEL:
            print("⚠️ 未找到模型文件，使用 yolov8n.pt 进行快速训练（仅用于演示）...")
            model = YOLO("yolov8n.pt")
            # 训练一个 epoch，模型会保存到 runs/detect/train/weights/best.pt
            model.train(data=str(data_yaml), epochs=EPOCHS_IF_TRAIN, imgsz=IMG_SIZE,
                        project="runs/detect", name="train_fruits", exist_ok=True)
            model_path = "runs/detect/train_fruits/weights/best.pt"
            model = YOLO(model_path)
            print(f"✅ 训练完成，模型保存在: {model_path}")
        else:
            print("❌ 没有模型文件，且未开启自动训练。请修改 MODEL_PATH 或设置 TRAIN_IF_NO_MODEL=True")
            sys.exit(1)

    # 3. 对整个验证集进行验证
    print("\n🚀 开始在验证集上评估...")
    metrics = model.val(data=str(data_yaml), imgsz=IMG_SIZE, split="val")

    # 4. 打印额外结果
    print("\n" + "="*50)
    print("验证完成！所有图片和曲线已保存到:", metrics.save_dir)
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"总体 precision: {metrics.box.p[0]:.4f}")   # 所有类平均
    print(f"总体 recall: {metrics.box.r[0]:.4f}")

    # 5. 打印归一化混淆矩阵文本
    print_normalized_confusion_matrix(metrics)

    # 6. 列出生成的重要文件
    save_dir = Path(metrics.save_dir)
    print("\n📁 生成的关键文件:")
    for f in ["confusion_matrix.png", "confusion_matrix_normalized.png",
              "P_curve.png", "R_curve.png", "F1_curve.png", "PR_curve.png"]:
        fp = save_dir / f
        if fp.exists():
            print(f"  ✅ {fp}")
        else:
            print(f"  ❌ {fp} 未找到")

    # 预测可视化
    pred_imgs = list(save_dir.glob("val_batch*_pred.jpg"))
    if pred_imgs:
        print(f"  ✅ {len(pred_imgs)} 张预测可视化图 (val_batch*_pred.jpg)")
    else:
        print("  ❌ 未找到预测可视化图")

    print("\n✅ 全部工作完成！")

if __name__ == "__main__":
    main()