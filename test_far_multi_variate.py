from far.models import build_model
from far.pipelines.pipeline_far import FARPipeline

import decord
import numpy as np
import torch
import json
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from far.utils.vis_util import log_paired_video
from einops import rearrange
decord.bridge.set_bridge('torch')
from torchvision import transforms
from read_multi_variate_data import *
import torch.nn.functional as F


def build_inference_pipeline(model_cfg, device="cuda", weight_dtype=torch.bfloat16):
    # build model
    if model_cfg['transformer'].get('from_pretrained'):
        raise NotImplementedError
    else:
        init_cfg = model_cfg['transformer']['init_cfg']
        model = build_model(init_cfg['type'])(**init_cfg.get('config', {}))
        if init_cfg.get('pretrained_path'):
            state_dict = torch.load(init_cfg['pretrained_path'], map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)

    if model_cfg['vae'].get('from_pretrained'):
        raise NotImplementedError
    elif model_cfg['vae'].get('from_config'):
        with open(model_cfg['vae']['from_config'], 'r') as fr:
            config = json.load(fr)
        vae = build_model(model_cfg['vae']['type']).from_config(config)
        if model_cfg['vae'].get('from_config_pretrained'):
            state_dict_path = model_cfg['vae']['from_config_pretrained']
            state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
            vae.load_state_dict(state_dict)
    else:
        raise NotImplementedError

    if model_cfg['scheduler']['from_pretrained']:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_cfg['scheduler']['from_pretrained'], subfolder='scheduler')
    else:
        raise NotImplementedError

    model.requires_grad_(False).to(device, dtype=weight_dtype).eval()
    vae.requires_grad_(False).to(device, dtype=weight_dtype).eval()

    pipeline = FARPipeline(transformer=model, vae=vae, scheduler=scheduler)
    pipeline.execution_device = device

    return pipeline

def read_video(video_path):
    video_reader = decord.VideoReader(video_path)
    total_frames = len(video_reader)
    frame_idxs = list(range(total_frames))
    frames = video_reader.get_batch(frame_idxs)

    frames = (frames / 255.0).float().permute(0, 3, 1, 2).contiguous()

    frames = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256)
    ])(frames)
    return frames

def read_video_dmlab(video_path):
    data = np.load(video_path)
    total_frames = len(data['video'])

    video = data['video']
    actions = data['actions']

    video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
    actions = torch.from_numpy(actions)

    return video, actions

def predict(video, context_frame, pred_len):
    context_frame = context_frame # you can adjust use how many frames as vision contexts
    save_suffix = "short_video_frame"

    # unconditinoal generation
    input_params = {
        # 'conditions': {"action": ref_actions.unsqueeze(0)},
        'context_sequence': video[:context_frame, ...].unsqueeze(0),
        'unroll_length': pred_len,
        'num_inference_steps': 50,
        'guidance_scale': 1.0,
        'generator': torch.Generator('cuda').manual_seed(0),
        # 'sample_size': 16,
        'sample_size': 4,
        'use_kv_cache': True
    }

    with torch.no_grad():
        pred_video = pipeline.generate(**input_params)

    pred_video = rearrange(pred_video, '(b n) f c h w -> b n f c h w', n=1)

    return pred_video

def padding_video(video, target_size=256):
    
    h, w = video.shape[-2:]
    pad_h = (target_size - h) // 2
    pad_w = (target_size - w) // 2

    # In case target is not divisible by 2 of difference
    pad = (pad_w, target_size - w - pad_w, pad_h, target_size - h - pad_h)
    scaled_data = F.pad(video, pad, mode='constant', value=0)
    return scaled_data, h, w, pad

def predict_loop(name, data, seq_len, pred_len, seeds=None, scaler_list=None):
    data = torch.tensor(data).float().cuda() # N x L x C
    data = data.transpose(1,2)  # N x L x C
    print('data shape: ', data.shape)
    if seeds is None:
        seeds = list(range(1000))
    N, L, C = data.shape
    step = seq_len + pred_len
    
    outs = []

    for i in range(N):
        print('data[i][:seq_len]: ', data[i][:seq_len].shape)
        # ctx = data[i][:seq_len].unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1)  # (1, ctx, C, H, W)
        ctx = data[i][:seq_len].unsqueeze(0).unsqueeze(2).unsqueeze(-1).repeat(1, 1, 3, 1, 1)
        # ctx = ctx.unsqueeze(0)  # (1, 1, ctx, C, H, W)
        print('before padding ctx: ', ctx.shape)
        # ctx = ctx.transpose()
        ctx, h, w, pad = padding_video(ctx, target_size=256)
        print('ctx: ', ctx.shape)
        print('pred_len: ', pred_len)
        input_params = {
            'context_sequence': ctx,
            'unroll_length': pred_len,
            'num_inference_steps': 50,
            'guidance_scale': 1.0,
            'generator': torch.Generator('cuda').manual_seed(seeds[i]),
            'sample_size': 8,
            'use_kv_cache': True
        }

        with torch.no_grad():
            out = pipeline.generate(**input_params)

        if out.dim() == 4:
            out = out.unsqueeze(0)
        out = rearrange(out, '(b n) f c h w -> b n f c h w', b=1, n=1)
        print('out: ', out.shape)
        out = out[:, 0, seq_len:, :, :, :].squeeze(0)
        out = out[..., pad[2]:pad[2]+h, pad[0]:pad[0]+w]
        # print('out: ', out.shape)
        out = out[:,0,:,:]
        # if scaler_list is not None:
        #     scaler = scaler_list[i]
        #     out_np = out.cpu().to(torch.float32).numpy()
        #     # print('out_np: ', out_np.shape)
        #     out_np = scaler.inverse_transform(out_np.reshape(out_np.shape[0], -1)).reshape(out_np.shape)
        #     # out_np = scaler.inverse_transform(out_np.reshape(1, -1)).reshape(out_np.shape)
        #     out = torch.tensor(out_np).float().cuda()
        outs.append(out.unsqueeze(0))

    return torch.cat(outs, dim=0)
def metric(pred, gt):
    # compute rmse and mae
    # pred B x T x H x W
    diff = pred - gt
    # print('pred: ', pred.shape, ' gt: ', gt.shape)
    # print('diff: ', diff)
    diff = diff.reshape(-1)
    rmse = torch.sqrt(torch.mean(diff ** 2))
    mae = torch.mean(torch.abs(diff))
    
    return rmse, mae

if __name__ == "__main__":
    model_cfg = {
        "transformer": {
            "init_cfg": {
                "type": "FAR_L",
                "pretrained_path": "/scratch/s225250685/project/videots/FAR/experiments/pretrained_models/FAR_Models/video_generation/FAR_L_UCF101_Uncond256-adea51e9.pth"
            }
        },
        "vae": {
            "type": "MyAutoencoderDC",
            "from_config": "options/model_cfg/dcae/model_32x_c32_config.json",
            "from_config_pretrained": "/scratch/s225250685/project/videots/FAR/experiments/pretrained_models/FAR_Models/dcae/DCAE_UCF101_Res256-9c4355c8.pth"
        },
        "scheduler": {
            "from_pretrained": "options/model_cfg/far/scheduler_config.json"
        }
    }

    device = torch.device("cuda")
    pipeline = build_inference_pipeline(model_cfg, device=device)
    pipeline.set_progress_bar_config(disable=True)
     
    dataset_name = 'ETTh1'
    data, scaler_list, eval_scaler = read_data(data_path_dict[dataset_name])
    data = data[100:102]
    # raw_data = raw_data[100:105]
    print('data shape: ', data.shape)  # N x T x H x W
    seq_len = 512
    pred_len = 4
    pred = predict_loop(dataset_name, data, seq_len, pred_len, scaler_list=scaler_list)
    # pred # N x T x C x 1
    # data # N x C x T
    
    pred = pred.squeeze(-1) # N x T x C
    target = data[:, seq_len:seq_len+pred_len, :] # N x T x C    12 x 7 x 96
    pred = pred.cpu().to(torch.float32).numpy()
    # target = target.cpu().to(torch.float32).numpy()
    # eval normalize
    pred = eval_scaler.transform(pred.reshape(-1, pred.shape[-1])).reshape(pred.shape)
    target = eval_scaler.transform(target.reshape(-1, target.shape[-1])).reshape(target.shape)

    print('pred: ', pred.shape)
    print('data: ', data.shape)
    # gt = torch.tensor(raw_data[:, seq_len:seq_len+pred_len]).float().cuda()
    
    rmse, mae = metric(pred, target)
    print(f"dataset_name: {dataset_name}, RMSE: {rmse}, MAE: {mae}")