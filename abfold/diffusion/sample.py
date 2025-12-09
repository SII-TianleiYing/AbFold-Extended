from diffusers import DiffusionPipeline, DDPMPipeline
import torch
import torch.nn.functional as F

# pipeline = DiffusionPipeline.from_pretrained("/home/pengchao/projects/abdiff/sd-antibody", use_safetensors=True)
# generator = torch.manual_seed(0)

def sample(s, z):
#     pipeline.to(s.device)

#     z = z.permute(0, 3, 1, 2)
    # 通过pad来保证下采样、上采样后的图像大小一致
    # 计算 padding 大小
#     height, width = z.shape[2], z.shape[3]
#     base = 32
#     new_height = (height + (base - 1)) // base * base  # 向上取 8 的倍数
#     new_width = (width + (base - 1)) // base * base

#     # 填充
#     z = F.pad(z, (0, new_width - width, 0, new_height - height), mode="constant", value=0)

#     image = torch.randn_like(z)

#     for t in pipeline.progress_bar(pipeline.scheduler.timesteps):
#             # 1. predict noise model_output
#             model_output = pipeline.unet(image, t, encoder_hidden_states=s).sample

#             # 2. compute previous image: x_t -> x_t-1
#             image = pipeline.scheduler.step(model_output, t, image, generator=generator).prev_sample

#     length = s.shape[1]
#     z = image[:, :, :length, :length]
#     z = z.permute(0, 2, 3, 1)

    data = torch.load('/home/pengchao/data/abdiff/test/sample_embedding/7lyw_HL_pred.pt', map_location=s.device)
    z_new = data['z']

#     std_path='/home/pengchao/data/abdiff/train/std_mean.pt'
#     std_mean = torch.load(std_path)
#     z_new = z_new * std_mean['std'].to(s.device) + std_mean['mean'].to(s.device)

    return z_new
