import os
import time
import json
import torch
import numpy as np
import pandas as pd
import torchvision
import scipy.signal
import echonet
from tqdm import tqdm
import traceback
import platform
import psutil  # 需要安裝: pip install psutil

# ================= 設定區 =================
VIDEO_DIR = r"../data_ready"
WEIGHTS_EF = "output/r2plus1d_18_32_2_pretrained.pt"
WEIGHTS_SEG = "output/deeplabv3_resnet50_random.pt"
OUTPUT_CSV = "EchoNet_Hardware_Benchmark_per_video.csv"

FRAMES = 32
PERIOD = 2
BATCH_SIZE = 20
# =========================================

def get_hardware_info():
    """取得硬體配置資訊"""
    info = {}
    
    # CPU 資訊
    info['CPU_Model'] = platform.processor() or "Unknown"
    info['CPU_Physical_Cores'] = psutil.cpu_count(logical=False)
    info['CPU_Logical_Cores'] = psutil.cpu_count(logical=True)
    info['CPU_Max_Frequency'] = f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() else "Unknown"
    
    # 記憶體資訊
    mem = psutil.virtual_memory()
    info['RAM_Total'] = f"{mem.total / (1024**3):.2f} GB"
    info['RAM_Available'] = f"{mem.available / (1024**3):.2f} GB"
    
    # GPU 資訊
    if torch.cuda.is_available():
        info['GPU_Available'] = True
        info['GPU_Count'] = torch.cuda.device_count()
        
        gpu_details = []
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_details.append({
                'GPU_ID': i,
                'GPU_Name': gpu_name,
                'GPU_Memory': f"{gpu_memory:.2f} GB"
            })
        info['GPU_Details'] = gpu_details
        
        # CUDA 版本
        info['CUDA_Version'] = torch.version.cuda
        info['cuDNN_Version'] = torch.backends.cudnn.version()
    else:
        info['GPU_Available'] = False
        info['GPU_Count'] = 0
        info['GPU_Details'] = []
    
    # PyTorch 版本
    info['PyTorch_Version'] = torch.__version__
    
    # 作業系統
    info['OS'] = f"{platform.system()} {platform.release()}"
    
    return info

def print_hardware_info(hw_info):
    """美化列印硬體資訊"""
    print("\n" + "="*60)
    print("HARDWARE CONFIGURATION")
    print("="*60)
    
    # 系統資訊
    print(f"Operating System    : {hw_info['OS']}")
    print(f"PyTorch Version     : {hw_info['PyTorch_Version']}")
    
    print("\n" + "-"*60)
    print("CPU INFORMATION")
    print("-"*60)
    print(f"Model               : {hw_info['CPU_Model']}")
    print(f"Physical Cores      : {hw_info['CPU_Physical_Cores']}")
    print(f"Logical Cores       : {hw_info['CPU_Logical_Cores']}")
    print(f"Max Frequency       : {hw_info['CPU_Max_Frequency']}")
    
    print("\n" + "-"*60)
    print("MEMORY INFORMATION")
    print("-"*60)
    print(f"Total RAM           : {hw_info['RAM_Total']}")
    print(f"Available RAM       : {hw_info['RAM_Available']}")
    
    print("\n" + "-"*60)
    print("GPU INFORMATION")
    print("-"*60)
    if hw_info['GPU_Available']:
        print(f"GPU Available       : Yes")
        print(f"GPU Count           : {hw_info['GPU_Count']}")
        print(f"CUDA Version        : {hw_info['CUDA_Version']}")
        print(f"cuDNN Version       : {hw_info['cuDNN_Version']}")
        
        for gpu in hw_info['GPU_Details']:
            print(f"\n  GPU {gpu['GPU_ID']}:")
            print(f"    Name            : {gpu['GPU_Name']}")
            print(f"    Memory          : {gpu['GPU_Memory']}")
    else:
        print(f"GPU Available       : No")
    
    print("="*60 + "\n")

def print_config():
    print("="*60)
    print(f"BENCHMARK CONFIGURATION")
    print("="*60)
    print(f"Video Directory     : {VIDEO_DIR}")
    print(f"Batch Size          : {BATCH_SIZE}")
    print(f"Frames/Clip         : {FRAMES}")
    print(f"Period              : {PERIOD}")
    print(f"EF Model Weights    : {WEIGHTS_EF}")
    print(f"Seg Model Weights   : {WEIGHTS_SEG}")
    print(f"Output CSV          : {OUTPUT_CSV}")
    print("="*60)

def get_mean_std(video_dir, num_samples=50):
    """ 計算 Dataset 的 Mean/Std """
    print("\nCalculating Mean/Std from dataset...")
    files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".avi")]
    if not files:
        raise FileNotFoundError(f"No .avi files found in {video_dir}")
        
    sample_files = np.random.choice(files, min(len(files), num_samples), replace=False)
    
    total_sum = np.zeros(3)
    total_sq_sum = np.zeros(3)
    total_pixels = 0
    
    for f in tqdm(sample_files, desc="Computing stats", leave=False):
        v = echonet.utils.loadvideo(f).astype(np.float32)
        v_flat = v.reshape(3, -1)
        total_sum += v_flat.sum(axis=1)
        total_sq_sum += (v_flat ** 2).sum(axis=1)
        total_pixels += v_flat.shape[1]
        
    mean = total_sum / total_pixels
    std = np.sqrt(total_sq_sum / total_pixels - mean ** 2)
    
    print(f"Computed: Mean={mean}, Std={std}")
    return mean.astype(np.float32), std.astype(np.float32)

def load_models(device_str):
    """ 載入模型並移至指定裝置 (不計入推論時間) """
    device = torch.device(device_str)
    
    # EF Model
    model_ef = torchvision.models.video.r2plus1d_18(pretrained=False)
    model_ef.fc = torch.nn.Linear(model_ef.fc.in_features, 1)
    if os.path.exists(WEIGHTS_EF):
        checkpoint = torch.load(WEIGHTS_EF, map_location=device, weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
        model_ef.load_state_dict(state_dict)
        print("EF Model weights loaded successfully")
    else:
        print("EF Model weights not found") 
    model_ef.to(device).eval()

    # Segmentation Model
    model_seg = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=False)
    last_layer = model_seg.classifier[-1]
    model_seg.classifier[-1] = torch.nn.Conv2d(
        last_layer.in_channels, 1, 
        kernel_size=last_layer.kernel_size
    )
    if os.path.exists(WEIGHTS_SEG):
        checkpoint = torch.load(WEIGHTS_SEG, map_location=device, weights_only=False)
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
        model_seg.load_state_dict(state_dict)
        print("Segmentation Model weights loaded successfully")
    else:
        print("Segmentation Model weights not found")
    model_seg.to(device).eval()
    
    return model_ef, model_seg

def run_pipeline_steps(video_path, model_ef, model_seg, device_str, mean, std):
    """
    執行單一影片的完整推論流程
    回傳結果與該影片的執行時間 (不含模型載入，但包含影片讀取與前處理)
    """
    device = torch.device(device_str)
    
    # 0. 前處理 (Loading + Normalization)
    t0 = time.time()
    video_raw = echonet.utils.loadvideo(video_path).astype(np.float32)
    video_norm = (video_raw - mean.reshape(3, 1, 1, 1)) / std.reshape(3, 1, 1, 1)
    video_tensor = torch.from_numpy(video_norm)
    
    C, F, H, W = video_tensor.shape
    clip_len_real = FRAMES * PERIOD
    
    # Stage 1: Dense Prediction
    t1_start = time.time()
    if F < clip_len_real:
        pad_len = clip_len_real - F
        zeros = torch.zeros((C, pad_len, H, W), dtype=video_tensor.dtype)
        padded_video = torch.cat((video_tensor, zeros), dim=1)
        start_indices = [0]
        source_video = padded_video
    else:
        start_indices = np.arange(F - (FRAMES - 1) * PERIOD)
        source_video = video_tensor

    dense_preds = {}
    with torch.no_grad():
        for i in range(0, len(start_indices), BATCH_SIZE):
            batch_starts = start_indices[i : i + BATCH_SIZE]
            batch_clips = []
            for s in batch_starts:
                clip = source_video[:, s : s + clip_len_real : PERIOD, :, :]
                batch_clips.append(clip)
            
            if batch_clips:
                batch_tensor = torch.stack(batch_clips).to(device)
                output = model_ef(batch_tensor)
                preds = output.view(-1).cpu().numpy()
                for idx, start_frame in enumerate(batch_starts):
                    dense_preds[start_frame] = preds[idx]
    t1_end = time.time()

    # Stage 2: Beat Detection
    t2_start = time.time()
    inputs = video_tensor.permute(1, 0, 2, 3) 
    lv_sizes = []
    with torch.no_grad():
        for i in range(0, inputs.shape[0], BATCH_SIZE):
            batch = inputs[i : i + BATCH_SIZE].to(device)
            output = model_seg(batch)['out']
            mask = torch.sigmoid(output) > 0.5
            lv_sizes.append(mask.sum(dim=(1, 2, 3)).cpu().numpy())
    lv_sizes = np.concatenate(lv_sizes)
    
    trim_min = sorted(lv_sizes)[int(round(len(lv_sizes) ** 0.05))]
    trim_max = sorted(lv_sizes)[int(round(len(lv_sizes) ** 0.95))]
    trim_range = trim_max - trim_min
    systole_indices, _ = scipy.signal.find_peaks(
        -lv_sizes, distance=20, prominence=(0.50 * trim_range)
    )
    t2_end = time.time()

    # Stage 3: Selection
    t3_start = time.time()
    beat_efs = []
    beat_details = []
    offset = FRAMES 
    
    for i, peak_frame in enumerate(systole_indices):
        target_start_frame = peak_frame - offset
        available_frames = np.array(list(dense_preds.keys()))
        if len(available_frames) == 0: continue
            
        closest_idx = (np.abs(available_frames - target_start_frame)).argmin()
        matched_frame = available_frames[closest_idx]
        
        if abs(matched_frame - target_start_frame) < 10:
            ef_val = dense_preds[matched_frame]
            beat_efs.append(float(ef_val))
            beat_details.append({
                "beat_id": i+1,
                "es_frame": int(peak_frame),
                "matched_clip_start": int(matched_frame),
                "ef_pred": round(float(ef_val), 2)
            })
            
    global_avg_ef = np.mean(list(dense_preds.values())) if dense_preds else None
    beat_avg_ef = np.mean(beat_efs) if beat_efs else None
    beat_std_ef = np.std(beat_efs) if beat_efs else None
    t3_end = time.time()
    
    total_time = t3_end - t0
    
    return {
        "Global_EF": round(global_avg_ef, 2) if global_avg_ef else None,
        "Beat_Avg_EF": round(beat_avg_ef, 2) if beat_avg_ef else None,
        "Beats_Matched": len(beat_efs),
        "Total_Time": total_time,
        "Device": device_str
    }

def run_benchmark_epoch(device_str, files, mean, std):
    """
    針對列表中的所有檔案執行一輪 Benchmark
    回傳: 平均每部影片的推論時間 (不含載入模型)
    """
    print(f"\n[{device_str.upper()}] Loading Models...", end="", flush=True)
    try:
        model_ef, model_seg = load_models(device_str)
        print(" Done.")
    except Exception as e:
        print(f" Failed to load models: {e}")
        return None, []

    # Warmup for GPU
    if device_str == "cuda":
        print(f"[{device_str.upper()}] Warming up GPU...", end="", flush=True)
        dummy = torch.zeros(1, 3, 32, 112, 112).cuda()
        model_ef(dummy)
        print(" Done.")

    print(f"[{device_str.upper()}] Starting Inference on {len(files)} videos...")
    
    times = []
    results = []
    
    for filename in tqdm(files, desc=f"Running {device_str.upper()}"):
        path = os.path.join(VIDEO_DIR, filename)
        try:
            res = run_pipeline_steps(path, model_ef, model_seg, device_str, mean, std)
            res["Filename"] = filename
            times.append(res['Total_Time'])
            results.append(res)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    avg_time = np.mean(times) if times else 0
    return avg_time, results

def save_hardware_info_to_file(hw_info, filename="hardware_config.json"):
    """將硬體資訊存成 JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(hw_info, f, indent=2, ensure_ascii=False)
    print(f"Hardware configuration saved to {filename}")

def main():
    # 1. 取得並顯示硬體資訊
    hw_info = get_hardware_info()
    print_hardware_info(hw_info)
    save_hardware_info_to_file(hw_info)
    
    # 2. 顯示 Benchmark 配置
    print_config()
    
    # 3. 準備檔案列表
    files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".avi")]
    if not files:
        print("No videos found.")
        return
    print(f"\nTotal Videos Found: {len(files)}")
    
    # 4. 計算 Mean/Std
    mean, std = get_mean_std(VIDEO_DIR, num_samples=50)
    
    # 5. 執行 CPU Benchmark
    print("\n" + "-"*60)
    print("PHASE 1: CPU BENCHMARK")
    print("-"*60)
    cpu_avg_time, cpu_results = run_benchmark_epoch("cpu", files, mean, std)
    
    # 6. 執行 GPU Benchmark
    print("\n" + "-"*60)
    print("PHASE 2: GPU BENCHMARK")
    print("-"*60)
    
    gpu_avg_time = float('nan')
    gpu_results = []
    
    if torch.cuda.is_available():
        gpu_avg_time, gpu_results = run_benchmark_epoch("cuda", files, mean, std)
    else:
        print("GPU not available. Skipping Phase 2.")
    
    # 7. 輸出比較報告
    print("\n" + "="*60)
    print("HARDWARE PERFORMANCE REPORT (Average Time per Video)")
    print("="*60)
    print(f"Videos Processed : {len(files)}")
    print(f"CPU Average Time : {cpu_avg_time:.4f} sec/video")
    
    if torch.cuda.is_available():
        print(f"GPU Average Time : {gpu_avg_time:.4f} sec/video")
        if gpu_avg_time > 0:
            speedup = cpu_avg_time / gpu_avg_time
            print(f"Speedup Factor      : {speedup:.2f}x")
    else:
        print("GPU Average Time    : N/A")
    print("="*60)
    
    # 8. 建立比較表格
    comparison_data = []
    cpu_dict = {r['Filename']: r for r in cpu_results}
    gpu_dict = {r['Filename']: r for r in gpu_results} if gpu_results else {}
    
    for filename in files:
        row = {'Filename': filename}
        
        if filename in cpu_dict:
            cpu_data = cpu_dict[filename]
            row['CPU_Time'] = cpu_data['Total_Time']
            row['CPU_Global_EF'] = cpu_data['Global_EF']
            row['CPU_Beat_Avg_EF'] = cpu_data['Beat_Avg_EF']
            row['CPU_Beats_Matched'] = cpu_data['Beats_Matched']
        else:
            row['CPU_Time'] = None
            row['CPU_Global_EF'] = None
            row['CPU_Beat_Avg_EF'] = None
            row['CPU_Beats_Matched'] = None
        
        if filename in gpu_dict:
            gpu_data = gpu_dict[filename]
            row['GPU_Time'] = gpu_data['Total_Time']
            row['GPU_Global_EF'] = gpu_data['Global_EF']
            row['GPU_Beat_Avg_EF'] = gpu_data['Beat_Avg_EF']
            row['GPU_Beats_Matched'] = gpu_data['Beats_Matched']
            
            if row['CPU_Time'] and row['GPU_Time'] and row['GPU_Time'] > 0:
                row['Speedup'] = round(row['CPU_Time'] / row['GPU_Time'], 2)
            else:
                row['Speedup'] = None
        else:
            row['GPU_Time'] = None
            row['GPU_Global_EF'] = None
            row['GPU_Beat_Avg_EF'] = None
            row['GPU_Beats_Matched'] = None
            row['Speedup'] = None
        
        comparison_data.append(row)
    
    # 9. 加入統計摘要
    df = pd.DataFrame(comparison_data)
    summary = {
        'Filename': '=== AVERAGE ===',
        'CPU_Time': df['CPU_Time'].mean(),
        'GPU_Time': df['GPU_Time'].mean() if torch.cuda.is_available() else None,
        'CPU_Global_EF': df['CPU_Global_EF'].mean(),
        'GPU_Global_EF': df['GPU_Global_EF'].mean() if torch.cuda.is_available() else None,
        'CPU_Beat_Avg_EF': df['CPU_Beat_Avg_EF'].mean(),
        'GPU_Beat_Avg_EF': df['GPU_Beat_Avg_EF'].mean() if torch.cuda.is_available() else None,
        'CPU_Beats_Matched': df['CPU_Beats_Matched'].mean(),
        'GPU_Beats_Matched': df['GPU_Beats_Matched'].mean() if torch.cuda.is_available() else None,
        'Speedup': df['Speedup'].mean() if torch.cuda.is_available() else None
    }
    
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    
    # 10. 儲存結果
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Detailed comparison saved to {OUTPUT_CSV}")
    print(f"✓ Hardware configuration saved to hardware_config.json")

if __name__ == "__main__":
    main()