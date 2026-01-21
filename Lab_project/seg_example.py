import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import echonet

# ================= 設定區 =================
VIDEO_DIR = r"../data_ready"
WEIGHTS_SEG = "output/deeplabv3_resnet50_random.pt"
OUTPUT_DIR = "visualization_results"
NUM_FRAMES = 5  # 要顯示的幀數
# =========================================

def get_mean_std(video_dir, num_samples=50):
    """計算 Dataset 的 Mean/Std"""
    from tqdm import tqdm
    
    print("Calculating Mean/Std from dataset...")
    files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".avi")]
    if not files:
        raise FileNotFoundError(f"No .avi files found in {video_dir}")
        
    sample_files = np.random.choice(files, min(len(files), num_samples), replace=False)
    
    total_sum = np.zeros(3)
    total_sq_sum = np.zeros(3)
    total_pixels = 0
    
    for f in tqdm(sample_files, desc="Computing stats"):
        v = echonet.utils.loadvideo(f).astype(np.float32)
        v_flat = v.reshape(3, -1)
        total_sum += v_flat.sum(axis=1)
        total_sq_sum += (v_flat ** 2).sum(axis=1)
        total_pixels += v_flat.shape[1]
        
    mean = total_sum / total_pixels
    std = np.sqrt(total_sq_sum / total_pixels - mean ** 2)
    
    print(f"Mean: {mean}")
    print(f"Std: {std}")
    return mean.astype(np.float32), std.astype(np.float32)

def load_segmentation_model(weights_path, device_str):
    """載入分割模型"""
    import torchvision
    
    device = torch.device(device_str)
    
    model_seg = torchvision.models.segmentation.deeplabv3_resnet50(
        pretrained=False, 
        progress=False
    )
    last_layer = model_seg.classifier[-1]
    model_seg.classifier[-1] = torch.nn.Conv2d(
        last_layer.in_channels, 1, 
        kernel_size=last_layer.kernel_size
    )
    
    if os.path.exists(weights_path):
        print(f"✓ Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
        model_seg.load_state_dict(state_dict)
        print("✓ Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    model_seg.to(device).eval()
    return model_seg

def visualize_segmentation(video_path, model_seg, device_str, mean, std, 
                          num_frames=5, output_dir="visualization_results"):
    """
    視覺化單一影片的分割結果
    
    Args:
        video_path: 影片路徑
        model_seg: 分割模型
        device_str: 'cpu' 或 'cuda'
        mean, std: 正規化參數
        num_frames: 要顯示的幀數（預設5張）
        output_dir: 儲存目錄
    """
    device = torch.device(device_str)
    video_name = os.path.basename(video_path)
    
    # 建立輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing: {video_name}")
    
    # 載入影片
    video_raw = echonet.utils.loadvideo(video_path).astype(np.float32)
    video_norm = (video_raw - mean.reshape(3, 1, 1, 1)) / std.reshape(3, 1, 1, 1)
    video_tensor = torch.from_numpy(video_norm)
    
    C, F, H, W = video_tensor.shape
    print(f"Video shape: C={C}, F={F}, H={H}, W={W}")
    
    # 選擇要視覺化的幀（均勻分布）
    frame_indices = np.linspace(0, F-1, num_frames, dtype=int)
    print(f"Selected frames: {frame_indices}")
    
    # 準備輸入
    inputs = video_tensor.permute(1, 0, 2, 3)  # (F, C, H, W)
    selected_frames = inputs[frame_indices].to(device)
    
    # 執行分割
    print("Running segmentation...")
    with torch.no_grad():
        output = model_seg(selected_frames)['out']
        masks = torch.sigmoid(output) > 0.5
        masks = masks.squeeze(1).cpu().numpy()  # (num_frames, H, W)
    
    # 準備原始影像（反正規化以便顯示）
    original_frames = video_raw[:, frame_indices, :, :]  # (C, num_frames, H, W)
    original_frames = original_frames.transpose(1, 2, 3, 0)  # (num_frames, H, W, C)
    original_frames = np.clip(original_frames / 255.0, 0, 1)  # 正規化到 [0, 1]
    
    # 計算整個影片的 LV 面積變化
    print("Computing LV size curve...")
    lv_sizes = []
    batch_size = 20
    inputs_all = video_tensor.permute(1, 0, 2, 3)
    
    with torch.no_grad():
        for i in range(0, F, batch_size):
            batch = inputs_all[i:i+batch_size].to(device)
            output = model_seg(batch)['out']
            mask = torch.sigmoid(output) > 0.5
            lv_sizes.extend(mask.sum(dim=(1, 2, 3)).cpu().numpy().tolist())
    lv_sizes = np.array(lv_sizes)
    
    # 偵測收縮期
    trim_min = sorted(lv_sizes)[int(round(len(lv_sizes) ** 0.05))]
    trim_max = sorted(lv_sizes)[int(round(len(lv_sizes) ** 0.95))]
    trim_range = trim_max - trim_min
    systole_indices, _ = scipy.signal.find_peaks(
        -lv_sizes, distance=20, prominence=(0.50 * trim_range)
    )
    
    print(f"Detected {len(systole_indices)} heartbeats at frames: {systole_indices}")
    
    # ========== 繪製分割結果 ==========
    fig, axes = plt.subplots(3, num_frames, figsize=(3*num_frames, 9))
    
    # 如果只有一幀，axes 會是 1D array，需要轉換
    if num_frames == 1:
        axes = axes.reshape(3, 1)
    
    for i, frame_idx in enumerate(frame_indices):
        # 第一行：原始影像
        axes[0, i].imshow(original_frames[i])
        axes[0, i].set_title(f'Frame {frame_idx}', fontsize=10)
        axes[0, i].axis('off')
        
        # 第二行：分割遮罩
        axes[1, i].imshow(masks[i], cmap='Reds', vmin=0, vmax=1)
        axes[1, i].set_title(f'LV Area: {masks[i].sum():.0f} px', fontsize=10)
        axes[1, i].axis('off')
        
        # 第三行：疊加結果
        axes[2, i].imshow(original_frames[i])
        axes[2, i].imshow(masks[i], cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        axes[2, i].set_title('Overlay', fontsize=10)
        axes[2, i].axis('off')
    
    # 添加行標題
    axes[0, 0].text(-0.3, 0.5, 'Original', transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', rotation=90)
    axes[1, 0].text(-0.3, 0.5, 'Segmentation', transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', rotation=90)
    axes[2, 0].text(-0.3, 0.5, 'Overlay', transform=axes[2, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', rotation=90)
    
    plt.tight_layout()
    
    # 添加整體標題
    fig.suptitle(f'Left Ventricle Segmentation: {video_name}\n'
                 f'Total Frames: {F} | Device: {device_str.upper()} | '
                 f'Detected Heartbeats: {len(systole_indices)}',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.93)
    
    # 儲存分割結果
    seg_path = os.path.join(output_dir, f"{video_name[:-4]}_segmentation.png")
    plt.savefig(seg_path, dpi=150, bbox_inches='tight')
    print(f"✓ Segmentation saved to {seg_path}")
    plt.close()
    
    # ========== 繪製 LV 面積變化曲線 ==========
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # 繪製 LV 面積曲線
    ax.plot(lv_sizes, linewidth=1.5, color='blue', label='LV Size', zorder=1)
    
    # 標記視覺化的幀位置
    for idx in frame_indices:
        ax.axvline(x=idx, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        ax.scatter(idx, lv_sizes[idx], color='red', s=100, zorder=5, 
                  edgecolors='darkred', linewidths=1.5)
    
    # 標記收縮末期（心跳位置）
    if len(systole_indices) > 0:
        for j, sys_idx in enumerate(systole_indices):
            ax.axvline(x=sys_idx, color='green', linestyle=':', alpha=0.7, linewidth=2)
            ax.scatter(sys_idx, lv_sizes[sys_idx], color='green', s=200, 
                      marker='v', zorder=6, edgecolors='darkgreen', linewidths=1.5,
                      label='End-Systole' if j == 0 else '')
    
    ax.set_xlabel('Frame Number', fontsize=13, fontweight='bold')
    ax.set_ylabel('LV Area (pixels)', fontsize=13, fontweight='bold')
    ax.set_title(f'Left Ventricle Size Over Cardiac Cycles: {video_name}\n'
                 f'Total Frames: {F} | Detected {len(systole_indices)} Heartbeats',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # 添加統計資訊
    stats_text = f"LV Area Stats:\n"
    stats_text += f"  Mean: {lv_sizes.mean():.1f} px\n"
    stats_text += f"  Min: {lv_sizes.min():.1f} px (Frame {lv_sizes.argmin()})\n"
    stats_text += f"  Max: {lv_sizes.max():.1f} px (Frame {lv_sizes.argmax()})"
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 儲存曲線圖
    curve_path = os.path.join(output_dir, f"{video_name[:-4]}_lv_curve.png")
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    print(f"✓ LV curve saved to {curve_path}")
    plt.close()
    
    return {
        'frame_indices': frame_indices,
        'lv_sizes': lv_sizes,
        'systole_indices': systole_indices,
        'masks': masks,
        'num_heartbeats': len(systole_indices)
    }

def main():
    print("="*70)
    print("ECHONET SEGMENTATION VISUALIZATION")
    print("="*70)
    print(f"Video Directory : {VIDEO_DIR}")
    print(f"Weights         : {WEIGHTS_SEG}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Frames to Show  : {NUM_FRAMES}")
    print("="*70)
    
    # 1. 取得影片列表
    files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".avi")]
    if not files:
        print("❌ No .avi files found in the directory!")
        return
    
    print(f"\nFound {len(files)} video(s)")
    
    # 2. 計算 Mean/Std
    mean, std = get_mean_std(VIDEO_DIR, num_samples=min(50, len(files)))
    
    # 3. 載入分割模型
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device_str.upper()}")
    
    model_seg = load_segmentation_model(WEIGHTS_SEG, device_str)
    
    # 4. 詢問要視覺化哪些影片
    print("\n" + "-"*70)
    print("Available videos:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f}")
    
    print("\nOptions:")
    print("  - Enter video index (e.g., 0)")
    print("  - Enter 'all' to visualize all videos")
    print("  - Press Enter to visualize the first video only")
    
    choice = input("\nYour choice: ").strip().lower()
    
    if choice == 'all':
        selected_videos = files
    elif choice == '':
        selected_videos = [files[0]]
    else:
        try:
            idx = int(choice)
            if 0 <= idx < len(files):
                selected_videos = [files[idx]]
            else:
                print(f"Invalid index. Using first video.")
                selected_videos = [files[0]]
        except:
            print(f"Invalid input. Using first video.")
            selected_videos = [files[0]]
    
    # 5. 執行視覺化
    print("\n" + "="*70)
    print(f"PROCESSING {len(selected_videos)} VIDEO(S)")
    print("="*70)
    
    results = []
    for video_file in selected_videos:
        video_path = os.path.join(VIDEO_DIR, video_file)
        
        result = visualize_segmentation(
            video_path=video_path,
            model_seg=model_seg,
            device_str=device_str,
            mean=mean,
            std=std,
            num_frames=NUM_FRAMES,
            output_dir=OUTPUT_DIR
        )
        
        results.append({
            'filename': video_file,
            'heartbeats': result['num_heartbeats'],
            'frames': len(result['lv_sizes'])
        })
    
    # 6. 總結
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        print(f"  {r['filename']}: {r['heartbeats']} heartbeats in {r['frames']} frames")
    
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}/")
    print("="*70)

if __name__ == "__main__":
    main()