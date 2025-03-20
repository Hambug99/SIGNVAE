import torch
from motion_vae import MotionVAE
from text_encoder import WordEncoder, encode_word
from FlagEmbedding import BGEM3FlagModel
from data_process import denormalize_motion_sequence, visualize_motion_sequence,load_dataset
import os
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='vgg').to(device)

# 加载模型
motion_vae = MotionVAE().to(device)
word_encoder = WordEncoder().to(device)

motion_vae.load_state_dict(torch.load("motion_vae.pth"))
word_encoder.load_state_dict(torch.load("word_encoder.pth"))

motion_vae.eval()
word_encoder.eval()

def generate_motion_from_text(word,bge_model):
    text_init = encode_word(word,bge_model).to(device) # 生成文本嵌入

    # text_vector = torch.tensor(text_init, dtype=torch.float32).unsqueeze(0).to(device)  # 添加批次维度
    text_vector = word_encoder(text_init)

    motion_vector = text_vector  # 直接使用文本嵌入作为 VAE 输入
    mean = motion_vector[:, 0, :]
    logvar = motion_vector[:, 1, :]
    z = motion_vae.reparameterization(mean, logvar).to(device)
    generated_motion = motion_vae.decoder(z)  # 解码手语动作
    return generated_motion.view(150, 137, 2).detach().cpu().numpy()


def compute_psnr_ssim_lpips(gt_path, pred_path):
    """
    计算两个文件夹下所有帧的 PSNR、SSIM、LPIPS，并返回平均值。
    :param gt_path: 真实图片的文件夹
    :param pred_path: 生成图片的文件夹
    :return: 平均 PSNR, SSIM, LPIPS
    """
    psnr_list, ssim_list, lpips_list = [], [], []
    
    gt_files = sorted(os.listdir(gt_path))
    pred_files = sorted(os.listdir(pred_path))
    
    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_img = np.array(Image.open(os.path.join(gt_path, gt_file)).convert("L"))  # 转灰度图
        pred_img = np.array(Image.open(os.path.join(pred_path, pred_file)).convert("L"))

        # 计算 PSNR
        psnr_value = psnr(gt_img, pred_img, data_range=255)
        psnr_list.append(psnr_value)

        # 计算 SSIM
        ssim_value = ssim(gt_img, pred_img, data_range=255)
        ssim_list.append(ssim_value)

        # 计算 LPIPS（需要转换为 Tensor）
        gt_tensor = torch.from_numpy(gt_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        pred_tensor = torch.from_numpy(pred_img).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        gt_tensor, pred_tensor = gt_tensor.to(device), pred_tensor.to(device)
        lpips_value = lpips_model(gt_tensor, pred_tensor).item()
        lpips_list.append(lpips_value)

    has_inf = False
    for lst in [psnr_list, ssim_list, lpips_list]:
        if np.isinf(lst).any():
            has_inf = True
            break

    if has_inf:
        valid_psnr = [x for x in psnr_list if not np.isinf(x)]
        max_psnr = max(valid_psnr) if valid_psnr else 0
        
        valid_ssim = [x for x in ssim_list if not np.isinf(x)]
        max_ssim = max(valid_ssim) if valid_ssim else 0
        
        valid_lpips = [x for x in lpips_list if not np.isinf(x)]
        min_lpips = min(valid_lpips) if valid_lpips else 0
        
        psnr_list = [max_psnr if np.isinf(x) else x for x in psnr_list]
        ssim_list = [max_ssim if np.isinf(x) else x for x in ssim_list]
        lpips_list = [min_lpips if np.isinf(x) else x for x in lpips_list]

    return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)


# 示例
if __name__ == "__main__":
    bge_model = BGEM3FlagModel(rf'D:\全部资料\PersonalData\lzq\Study\Project\SLKG\code\SignKG\thirdparty\bge\bge-m3',
                                    use_fp16=True)

    data_path = "data_demo"  # 你的数据目录
    motions, labels = load_dataset(data_path)
    print(f"数据加载完成，共 {len(motions)} 个样本")

    output_base_path = "./output_demo"
    total_psnr, total_ssim, total_lpips = 0, 0, 0
    results = []

    for i, label in tqdm(enumerate(labels), total=len(labels)):
        # 生成预测结果
        motion_sequence = generate_motion_from_text(label, bge_model)
        motion_sequence_re = denormalize_motion_sequence(motion_sequence)

        # GT 关键点
        motion_sequence_gt = denormalize_motion_sequence(motions[i])

        # 可视化并保存图像
        visualize_motion_sequence(motion_sequence_gt, output_base_path, i, "gt")
        visualize_motion_sequence(motion_sequence_re, output_base_path, i, "pred")

        # 计算评估指标
        gt_path = os.path.join(output_base_path, f"sample_{i:03d}", "gt")
        pred_path = os.path.join(output_base_path, f"sample_{i:03d}", "pred")

        psnr_avg, ssim_avg, lpips_avg = compute_psnr_ssim_lpips(gt_path, pred_path)
        total_psnr += psnr_avg
        total_ssim += ssim_avg
        total_lpips += lpips_avg
        
        results.append(f"Sample {i + 1}: PSNR = {psnr_avg:.6f}, SSIM = {ssim_avg:.6f}, LPIPS = {lpips_avg:.6f}")

    # 计算数据集的平均指标
    avg_psnr = total_psnr / len(labels)
    avg_ssim = total_ssim / len(labels)
    avg_lpips = total_lpips / len(labels)

    results.append(f"Average PSNR: {avg_psnr:.6f}, Average SSIM: {avg_ssim:.6f}, Average LPIPS: {avg_lpips:.6f}")

    # 保存结果
    with open("image_quality_results.txt", "w") as f:
        f.write("\n".join(results) + "\n")

    print(f"计算完成！平均 PSNR: {avg_psnr:.6f}, 平均 SSIM: {avg_ssim:.6f}, 平均 LPIPS: {avg_lpips:.6f}")
