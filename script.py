import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import ConvNextV2Model, AutoImageProcessor
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import os
import optuna
import copy

df = pd.read_csv("../spinal-level-detection_data/coords_pretrain.csv")
pivot = df.pivot(index=['filename', 'source'], columns='level', values=['x','y'])
pivot.columns = [f"{level.replace('/','_')}_{coord}" for coord, level in pivot.columns]
data = pivot.reset_index()

H, W = 256, 256
LEVELS = ["L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"] 
SIGMA = 4 ## Std deviation for the guassian heatmap

def row_to_points_xy(row, levels=LEVELS, H=H, W=W):
    """
    row: a pandas Series representing one CSV row with columns like:
         L1_L2_x, L1_L2_y, L2_L3_x, L2_L3_y, ...
    returns: numpy array shape (K,2) with (x,y) in pixel coords scaled to (W,H)
             if a coordinate is missing/NaN, we set (-1,-1) for that keypoint.
    """
    pts = np.full((len(levels), 2), -1.0, dtype=np.float32)
    for i, lvl in enumerate(levels):
        x_col = f"{lvl}_x"
        y_col = f"{lvl}_y"
        if x_col in row and y_col in row:
            xv = row[x_col]
            yv = row[y_col]
            if not (pd.isna(xv) or pd.isna(yv)):
                pts[i, 0] = float(xv)
                pts[i, 1] = float(yv)
    return pts

class SpineLevelKeypointDataset(Dataset):
    def __init__(self, data : pd.DataFrame, levels=LEVELS, H=H, W=W, sigma=SIGMA, use_resize=True, rgb=False):
        """
        csv_path: path
        levels: list of level name prefixes matching CSV pattern e.g. 'L1_L2'
        use_resize: if True, images will be resized to (W,H) and points will be scaled accordingly
                   If False, assumes coords already match (W,H).
        """
        self.df = data
        assert "filename" in self.df.columns, "CSV must contain 'filename' column"
        self.levels = levels
        self.H = H
        self.W = W
        self.sigma = sigma
        self.use_resize = use_resize
        self.source_dict = {
            'tseg' : '../spinal-level-detection_data/processed_tseg_jpgs',
            'osf' : '../spinal-level-detection_data/processed_osf_jpgs',
            'lsd' : '../spinal-level-detection_data/processed_lsd_jpgs',
            'spider' : '../spinal-level-detection_data/processed_spider_jpgs'
        }
        self.rgb = rgb
    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_source = row['source']
        fname = row["filename"]
        img_path = os.path.join(self.source_dict[img_source], fname)
        img = Image.open(img_path).convert("L") if not self.rgb else Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        if self.use_resize:
            img_resized = img.resize((self.W, self.H), Image.BICUBIC)
        else:
            img_resized = img

        img_tensor = torch.from_numpy(np.array(img_resized)).float().unsqueeze(0) / 255.0  # normalized
        pts = row_to_points_xy(row, levels=self.levels, H=self.H, W=self.W)
        if self.use_resize:
            if orig_w > 0 and orig_h > 0:
                scale_x = self.W / orig_w
                scale_y = self.H / orig_h
                for k in range(pts.shape[0]):
                    if pts[k,0] >= 0 and pts[k,1] >= 0:
                        pts[k,0] = np.clip(pts[k,0] * scale_x, 0, self.W - 1)
                        pts[k,1] = np.clip(pts[k,1] * scale_y, 0, self.H - 1)

        target, mask = make_gaussian_heatmaps(pts, self.H, self.W, sigma=self.sigma)  # (K,H,W), (K,1,1)
        sample = {
            "image": img_tensor,           # (1,H,W)
            "target": target,              # (K,H,W)
            "mask": mask,                  # (K,1,1)
            "points_xy": torch.from_numpy(pts).float(),
        }
        return sample
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, random_state = 69, test_size=0.15, shuffle=True)

train_loader = DataLoader(SpineLevelKeypointDataset(train_data), batch_size = 16, shuffle = True)
test_loader = DataLoader(SpineLevelKeypointDataset(test_data), batch_size=32, shuffle = True)

def make_gaussian_heatmaps(points_xy, H, W, sigma=3.0):
    """
    points_xy: (K, 2) numpy or torch arr with x,y pixel coords in *target* image space (W,H). List data type
    returns: target (K,H,W), mask (K,1)
    """
    points_xy = torch.as_tensor(points_xy, dtype=torch.float32)
    K = points_xy.shape[0]
    target = torch.zeros((K, H, W), dtype=torch.float32) # Target heatmap
    mask   = torch.zeros((K, 1, 1), dtype=torch.float32) # defaulting to 0

    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij') 
    var = sigma ** 2

    for k in range(K):
        x, y = points_xy[k] # K, 2
        if x >= 0 and y >= 0:
            g = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * var))
            target[k] = g
            mask[k, 0, 0] = 1.0
    return target.clamp_(0, 1), mask

def BCEWithLogitsLoss(logits, targets, mask):
    # We apply the mask to the computed loss before reducing it
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    masked_loss = loss * mask # masked loss values will be zeroed out 
    # Return the mean of the loss over non-masked elements
    return masked_loss.sum() / mask.sum() 

## Loading the UNet model with EfficientNet backbone
K = 5 # Number of levels for which the keypoints need to be detected
keypoint_detector = smp.Unet(
    encoder_name="efficientnet-b6",
    encoder_weights="imagenet",
    in_channels=1,
    classes=K,
    activation=None
)

import matplotlib.pyplot as plt

batch = next(iter(train_loader))

img = batch['image'][0].squeeze(0).numpy()
target = batch['target'][0].numpy()
points = batch["points_xy"][0].numpy()    

plt.figure(figsize=(15, 4))
plt.subplot(1, len(target)+1, 1)
plt.imshow(img, cmap="gray")
plt.title("Original MRI")
plt.axis("off")

for k in range(target.shape[0]):
    plt.subplot(1, len(target)+1, k+2)
    plt.imshow(img, cmap="gray")
    plt.imshow(target[k], cmap="jet", alpha=0.5)   
    x, y = points[k]
    if x >= 0 and y >= 0:                        
        plt.scatter([x], [y], c="white", s=40, marker="x")
    plt.title(f"Heatmap {k}")
    plt.axis("off")

plt.tight_layout()
plt.show()
print("instruction massage output")

torch.save(keypoint_detector, 'spine_level_detector_v1.pth')

## TRAINING without hp tuning##
EPOCHS = 15
LR = 5e-3
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

### UNet + EfficientNetB6 as encoder -- outputs 
optimizer = torch.optim.AdamW(keypoint_detector.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # step size is the number of epochs after which the lr reduces

from tqdm import tqdm
keypoint_detector.to(DEVICE)

# 在训练循环之前初始化存储列表
train_losses = []
val_losses = []
epochs_list = []

for epoch in range(1, EPOCHS+1):
    ### TRAINING
    keypoint_detector.train()
    running_loss = 0.0
    n = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Training", leave=False)
    for batch in train_bar:
        imgs = batch['image'].to(DEVICE, non_blocking=True)
        targets = batch['target'].to(DEVICE, non_blocking=True)
        masks = batch.get('mask').to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = keypoint_detector(imgs)
        loss = BCEWithLogitsLoss(logits, targets, mask=masks) 
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        n += bs
        train_bar.set_postfix({"Running Loss": f"{loss.item():.4f}"})

    avg_train_loss = running_loss / max(1, n)
    train_losses.append(avg_train_loss)  # 存储训练损失
    print(f"Epoch {epoch}/{EPOCHS} Training complete. Avg Loss: {avg_train_loss:.6f}")

    # VALIDATION
    keypoint_detector.eval()
    running_loss = 0.0
    n = 0
    val_bar = tqdm(test_loader, desc=f"Epoch {epoch}/{EPOCHS} Validation", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            imgs = batch['image'].to(DEVICE, non_blocking=True)
            targets = batch['target'].to(DEVICE, non_blocking=True)
            masks = batch.get('mask').to(DEVICE, non_blocking=True)

            logits = keypoint_detector(imgs)
            loss = BCEWithLogitsLoss(logits, targets, mask=masks)

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            n += bs
            val_bar.set_postfix({"Running Loss": f"{loss.item():.4f}"})

    avg_val_loss = running_loss / max(1, n)
    val_losses.append(avg_val_loss)  # 存储验证损失
    epochs_list.append(epoch)  # 存储epoch数
    
    scheduler.step()
    print(f"Epoch {epoch}/{EPOCHS} Validation complete. Avg Loss: {avg_val_loss:.6f}")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_losses, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_list, val_losses, 'r-', label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图像
plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印最终结果
print(f"\nFinal Training Loss: {train_losses[-1]:.6f}")
print(f"Final Validation Loss: {val_losses[-1]:.6f}")

