from ultralytics import YOLO
import os

if __name__ == '__main__':
    exp_name = "train_2"
    yaml_path = "F:/DaSai/XunLianShuJu/classes.yaml"
    device = 0
    # 补齐中间这层 runs！
    best_weight_path = f"F:/DaSai/XunLianShuJu/runs/detect/runs/{exp_name}/weights/best.pt"
    
    print("正在寻找模型路径：", best_weight_path)
    print("文件是否存在：", os.path.exists(best_weight_path))

    model = YOLO(best_weight_path)
    metrics = model.val(
        data=yaml_path,
        device=device,
        workers=0
    )
    print("="*50)
    print(f"实验：{exp_name} 评估结果")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.p:.4f}")
    print(f"Recall:     {metrics.box.r:.4f}")
