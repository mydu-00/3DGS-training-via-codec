#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def _qdq_tensor(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor (global) symmetric int8 QDQ. One scale for the entire tensor."""
    max_abs = x.abs().max().clamp(min=1e-12)
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q.to(x.dtype) * scale


def _qdq_channel(x: torch.Tensor) -> torch.Tensor:
    """Per-channel symmetric int8 QDQ.

    Computes one scale per (k, c) position shared across the N (gaussian) dimension.
    For _features_rest [N, K, 3]: K*3 = 45 scales (SH degree 3).
    For _features_dc  [N, 1, 3]:  1*3 = 3 scales.
    """
    max_abs = x.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q.to(x.dtype) * scale


def _qdq_group(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Per-group symmetric int8 QDQ (channel-wise grouping with global scales).

    Flattens dims 1+ into a single channel dimension of size flat_C, then groups
    into chunks of group_size. Each group has ONE scale shared across ALL gaussians.

    For _features_rest [N, 15, 3] -> flat_C = 45.  Practical group sizes that
    divide 45: 1, 3, 5, 9, 15, 45.  Default 15 yields 3 groups (one per RGB
    channel across all SH orders); 9 yields 5 groups; 5 yields 9 groups.

    For _features_dc [N, 1, 3] -> flat_C = 3.  Divisors of 3: 1, 3.  If the
    requested group_size does not divide flat_C, the largest divisor that does is
    used automatically (fallback guarantee).

    This is now consistent with channel granularity: scales are shared across
    the gaussian (dim 0) dimension.
    """
    orig_shape = x.shape
    N = x.shape[0]
    flat = x.reshape(N, -1)      # [N, flat_C]
    flat_C = flat.shape[1]

    # Find effective group size: largest divisor of flat_C that is <= group_size.
    gs = group_size
    if flat_C % gs != 0:
        # Collect all divisors of flat_C up to group_size, pick the largest.
        divisors = [d for d in range(1, min(gs, flat_C) + 1) if flat_C % d == 0]
        gs = divisors[-1]  # divisors is always non-empty (1 always divides flat_C)

    G = flat_C // gs
    grouped = flat.reshape(N, G, gs)                                   # [N, G, gs]
    # Compute max across both gaussians (dim 0) and group elements (dim 2)
    # This gives G scales total, shared across all N gaussians
    max_abs = grouped.abs().amax(dim=(0, 2), keepdim=True).clamp(min=1e-12)  # [1, G, 1]
    scale = max_abs / 127.0
    q = torch.clamp(torch.round(grouped / scale), -127, 127).to(torch.int8)
    return (q.to(x.dtype) * scale).reshape(orig_shape)


def _qdq_gaussian_group(x: torch.Tensor, gaussian_group_size: int) -> torch.Tensor:
    """Per-gaussian-group symmetric int8 QDQ.

    Groups gaussians (along dim 0) into blocks of gaussian_group_size, and computes
    one scale per group across all channels. This groups spatially close gaussians
    together for quantization.

    For _features_rest [N, 15, 3] with gaussian_group_size=256:
    - Creates N//256 groups, each with 256 gaussians
    - Each group has one scale for all 45 channel dimensions

    For _features_dc [N, 1, 3] with gaussian_group_size=256:
    - Creates N//256 groups, each with 256 gaussians
    - Each group has one scale for all 3 channel dimensions

    Args:
        x: Input tensor [N, K, C] where N = num gaussians
        gaussian_group_size: Number of gaussians per group (e.g., 256)

    Returns:
        Quantized and dequantized tensor with same shape as input.
    """
    orig_shape = x.shape
    N = x.shape[0]

    # Flatten all channels into a single dimension
    flat = x.reshape(N, -1)  # [N, flat_C]
    flat_C = flat.shape[1]

    # Determine effective group size
    ggs = min(gaussian_group_size, N)  # Can't group more than N gaussians
    if N % ggs != 0:
        # Find largest divisor of N that is <= gaussian_group_size
        divisors = [d for d in range(1, min(ggs, N) + 1) if N % d == 0]
        ggs = divisors[-1]

    G = N // ggs  # Number of groups
    # Reshape to [G, ggs, flat_C] - each group contains ggs gaussians
    grouped = flat.reshape(G, ggs, flat_C)

    # Compute scale per group (across both gaussians and channels in that group)
    # Shape: [G, 1, 1]
    max_abs = grouped.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
    scale = max_abs / 127.0

    # Quantize and dequantize
    q = torch.clamp(torch.round(grouped / scale), -127, 127).to(torch.int8)
    return (q.to(x.dtype) * scale).reshape(orig_shape)


def _qdq_gaussian_group_channel(x: torch.Tensor, gaussian_group_size: int) -> torch.Tensor:
    """Per-gaussian-group with per-channel quantization within each group.

    Groups gaussians into blocks, then within each group, uses per-channel quantization.
    This is finer-grained than gaussian_group: each group has one scale per channel.

    For _features_rest [N, 15, 3] with gaussian_group_size=256:
    - Creates N//256 groups, each with 256 gaussians
    - Each group has 45 scales (one per channel, shared across the 256 gaussians in that group)

    Args:
        x: Input tensor [N, K, C] where N = num gaussians
        gaussian_group_size: Number of gaussians per group (e.g., 256)

    Returns:
        Quantized and dequantized tensor with same shape as input.
    """
    orig_shape = x.shape
    N = x.shape[0]

    # Determine effective group size
    ggs = min(gaussian_group_size, N)
    if N % ggs != 0:
        divisors = [d for d in range(1, min(ggs, N) + 1) if N % d == 0]
        ggs = divisors[-1]

    G = N // ggs  # Number of groups
    # Reshape to [G, ggs, K, C] to preserve channel structure
    grouped = x.reshape(G, ggs, x.shape[1], x.shape[2])

    # Compute scale per group per channel (across the ggs gaussians in each group)
    # Shape: [G, 1, K, C]
    max_abs = grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = max_abs / 127.0

    # Quantize and dequantize
    q = torch.clamp(torch.round(grouped / scale), -127, 127).to(torch.int8)
    return (q.to(x.dtype) * scale).reshape(orig_shape)


def _qdq_gaussian_group_group(x: torch.Tensor, gaussian_group_size: int, channel_group_size: int) -> torch.Tensor:
    """Per-gaussian-group with per-group channel quantization within each group.

    Groups gaussians into blocks, then within each group, uses channel-group quantization.
    This provides a middle ground between gaussian_group and gaussian_group_channel.

    For _features_rest [N, 15, 3] with gaussian_group_size=256 and channel_group_size=15:
    - Creates N//256 groups, each with 256 gaussians
    - Each group has 3 scales (45 channels / 15 = 3 groups per gaussian group)

    Args:
        x: Input tensor [N, K, C] where N = num gaussians
        gaussian_group_size: Number of gaussians per group (e.g., 256)
        channel_group_size: Channel group size within each gaussian group (e.g., 15)

    Returns:
        Quantized and dequantized tensor with same shape as input.
    """
    orig_shape = x.shape
    N = x.shape[0]

    # Flatten channel dimensions
    flat = x.reshape(N, -1)  # [N, flat_C]
    flat_C = flat.shape[1]

    # Determine effective gaussian group size
    ggs = min(gaussian_group_size, N)
    if N % ggs != 0:
        divisors = [d for d in range(1, min(ggs, N) + 1) if N % d == 0]
        ggs = divisors[-1]

    # Determine effective channel group size
    cgs = channel_group_size
    if flat_C % cgs != 0:
        divisors = [d for d in range(1, min(cgs, flat_C) + 1) if flat_C % d == 0]
        cgs = divisors[-1]

    GG = N // ggs  # Number of gaussian groups
    CG = flat_C // cgs  # Number of channel groups

    # Reshape to [GG, ggs, CG, cgs]
    grouped = flat.reshape(GG, ggs, CG, cgs)

    # Compute scale per gaussian group per channel group (across ggs gaussians and cgs channels)
    # Shape: [GG, 1, CG, 1]
    max_abs = grouped.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = max_abs / 127.0

    # Quantize and dequantize
    q = torch.clamp(torch.round(grouped / scale), -127, 127).to(torch.int8)
    return (q.to(x.dtype) * scale).reshape(orig_shape)


def _quant_dequant_int8_inplace(
    param: torch.nn.Parameter,
    granularity: str = "tensor",
    group_size: int = 15,
    gaussian_group_size: int = 256,
) -> None:
    """Apply symmetric int8 QDQ to a parameter tensor in-place.

    Args:
        param:              The parameter to quantize (modified in-place).
        granularity:        ``'tensor'`` – one global scale (original behaviour);
                            ``'channel'`` – one scale per (SH-order, colour) position;
                            ``'group'``   – one scale per block of *group_size* flat
                                           channel elements (global across gaussians);
                            ``'gaussian_group'`` – one scale per block of *gaussian_group_size*
                                                  gaussians (spatial grouping);
                            ``'gaussian_group_channel'`` – gaussian groups with per-channel
                                                          quantization within each group;
                            ``'gaussian_group_group'`` – gaussian groups with channel-group
                                                        quantization within each group.
        group_size:         Block size for ``'group'`` and ``'gaussian_group_group'`` modes.
        gaussian_group_size: Block size for ``'gaussian_group'``, ``'gaussian_group_channel'``,
                            and ``'gaussian_group_group'`` modes.
    """
    with torch.no_grad():
        if param.data.abs().max() <= 0:
            return
        x = param.data
        if granularity == "channel":
            param.data.copy_(_qdq_channel(x))
        elif granularity == "group":
            param.data.copy_(_qdq_group(x, group_size))
        elif granularity == "gaussian_group":
            param.data.copy_(_qdq_gaussian_group(x, gaussian_group_size))
        elif granularity == "gaussian_group_channel":
            param.data.copy_(_qdq_gaussian_group_channel(x, gaussian_group_size))
        elif granularity == "gaussian_group_group":
            param.data.copy_(_qdq_gaussian_group_group(x, gaussian_group_size, group_size))
        else:  # "tensor" – default, backward-compatible
            param.data.copy_(_qdq_tensor(x))


def _apply_sh_int8_quantization(
    gaussians: GaussianModel,
    mode: str,
    granularity: str = "tensor",
    group_size: int = 15,
    gaussian_group_size: int = 256,
) -> None:
    """Apply SH int8 quantization to the Gaussian model.

    Args:
        gaussians:          The Gaussian model whose SH features are quantized.
        mode:               Which SH components to quantize: ``'none'``, ``'dc'``,
                            ``'rest'``, or ``'all'``.
        granularity:        Quantization granularity (``'tensor'`` / ``'channel'`` /
                            ``'group'`` / ``'gaussian_group'``).
                            See :func:`_quant_dequant_int8_inplace`.
        group_size:         Group size for ``'group'`` granularity (channel-wise).
        gaussian_group_size: Group size for ``'gaussian_group'`` granularity (spatial).
    """
    if mode == "none":
        return
    if mode in ("all", "dc"):
        _quant_dequant_int8_inplace(gaussians._features_dc, granularity, group_size, gaussian_group_size)
    if mode in ("all", "rest"):
        _quant_dequant_int8_inplace(gaussians._features_rest, granularity, group_size, gaussian_group_size)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, sh_int8_quantization, *, quant_granularity="tensor", quant_group_size=15, quant_gaussian_group_size=256):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    _apply_sh_int8_quantization(gaussians, sh_int8_quantization, quant_granularity, quant_group_size, quant_gaussian_group_size)
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    _apply_sh_int8_quantization(gaussians, sh_int8_quantization, quant_granularity, quant_group_size, quant_gaussian_group_size)
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--sh_int8_quantization', type=str, choices=["none", "dc", "rest", "all"], default="none")
    parser.add_argument(
        '--quant_granularity', type=str,
        choices=["tensor", "channel", "group", "gaussian_group", "gaussian_group_channel", "gaussian_group_group"],
        default="tensor",
        help=(
            "SH int8 quantization granularity. "
            "'tensor': one global scale per tensor (original behaviour, default). "
            "'channel': one scale per (SH-order, colour) position across all Gaussians "
            "(45 scales for rest, 3 for dc). "
            "'group': one scale per block of --quant_group_size flat-channel elements "
            "(global across gaussians, e.g. group_size=15 gives 3 groups for rest's 45 dims). "
            "'gaussian_group': one scale per block of --quant_gaussian_group_size gaussians "
            "(spatial grouping, e.g. 256 gaussians share one scale for all channels). "
            "'gaussian_group_channel': gaussian groups with per-channel quantization within each group "
            "(e.g. 256 gaussians per group, 45 scales per group for rest). "
            "'gaussian_group_group': gaussian groups with channel-group quantization within each group "
            "(e.g. 256 gaussians per group, 3 channel-group scales per gaussian group)."
        ),
    )
    parser.add_argument(
        '--quant_group_size', type=int, default=15,
        help=(
            "Group size for --quant_granularity=group (channel-wise grouping). "
            "Must ideally divide the flat channel count (45 for SH-rest, 3 for dc). "
            "Divisors of 45: 1, 3, 5, 9, 15, 45. "
            "If not a divisor, the largest divisor <= this value is used automatically. "
            "Default 15 gives 3 groups (one per RGB channel across all SH orders)."
        ),
    )
    parser.add_argument(
        '--quant_gaussian_group_size', type=int, default=256,
        help=(
            "Group size for gaussian-based granularities (spatial grouping). "
            "Number of gaussians per quantization group. "
            "Used by: gaussian_group, gaussian_group_channel, gaussian_group_group. "
            "Default 256. Common values: 128, 256, 512 (balance between granularity and parallelism)."
        ),
    )
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.sh_int8_quantization, quant_granularity=args.quant_granularity, quant_group_size=args.quant_group_size, quant_gaussian_group_size=args.quant_gaussian_group_size)

    # All done
    print("\nTraining complete.")
