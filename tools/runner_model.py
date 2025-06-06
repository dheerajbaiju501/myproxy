from email.mime import base
import torch
import torch.nn as nn
import numpy as np
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from utils import pc_util
from utils.other_utils import write_ply
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import matplotlib.pyplot as plt
# from pytorch3d.ops import estimate_pointcloud_normals
# from pytorch3d.ops.knn import _KNN, knn_gather, knn_points
from torchstat import stat
from PIL import Image
from pyntcloud import PyntCloud
import pandas as pd

TEST_DIR = "result/ShapeNet55/our_model/valid"
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

def save_pc(pc, filename, path):
    """
    pc: (N, 3) numpy array
    """
    points = pd.DataFrame(pc, columns=["x", "y", "z"])
    cloud = PyntCloud(points)
    cloud.to_file(os.path.join(path, filename))

# def get_missing_part(pcl, pcl_input):
#     B = pcl.size(0)
#     λ = 0.8
#     ε = 1e-2
#     pcl_input_normals = estimate_pointcloud_normals(pcl_input)  # (1, n, 3)

#     idx = knn_points(pcl, pcl_input, K=1).idx
#     nearest_points = knn_gather(pcl_input, idx) # (1, N, K=1, 3)
#     # print(nearest_points.shape)
#     nearest_points_normals = knn_gather(pcl_input_normals, idx) # (1, N, K=1, 3)
#     # print(nearest_points_normals.shape)
#     nearest_points, nearest_points_normals = nearest_points.squeeze(-2), nearest_points_normals.squeeze(-2) # (1, N, 3)

#     p2point_dist = ((pcl - nearest_points)**2).sum(dim=-1).sqrt()  # (1, N)
#     # print(p2point_dist.shape)
#     p2plane_dist = (((pcl - nearest_points) * nearest_points_normals)**2).sum(-1).sqrt()    # (1, N)
#     # print(p2plane_dist.shape)

#     dist = λ * p2plane_dist + (1 - λ) * p2point_dist

#     pcl_input_dense = pcl[:, dist[0] <= ε, :]
#     pcl_missing_dense = pcl[:, dist[0] > ε, :]
#     # print(pcl_missing_dense.shape)
#     return pcl_input_dense, pcl_missing_dense

def plot_pcd_one_view(
    filename,
    pcds,
    titles,
    suptitle="",
    sizes=None,
    cmap="Reds",
    zdir="y",
    xlim=(-0.5, 0.5),
    ylim=(-0.5, 0.5),
    zlim=(-0.5, 0.5),
):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30  # 水平倾斜
    azim = -45  # 旋转
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection="3d")
        ax.view_init(elev, azim)
        ax.scatter(
            pcd[:, 0],
            pcd[:, 1],
            pcd[:, 2],
            zdir=zdir,
            c=color,
            s=size,
            cmap=cmap,
            vmin=-1.0,
            vmax=0.5,
        )
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1
    )
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])
        f_losses = AverageMeter()

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            # print(npoints)
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                # print(partial.size())
                # print(gt.size())
                _, missing_part = get_missing_part(gt, partial)
                # save_pc(partial[0].cpu().numpy(), f"{taxonomy_ids}_{model_ids}_partial_part.ply", "/home/lss/Project/PoinTr/result/PCN/separate")
                # save_pc(existing_part[0].cpu().numpy(), f"{taxonomy_ids}_{model_ids}_existing_part.ply", "/home/lss/Project/PoinTr/result/PCN/separate")
                # save_pc(missing_part[0].cpu().numpy(), f"{taxonomy_ids}_{model_ids}_missing_part.ply", "/home/lss/Project/PoinTr/result/PCN/separate")
                # save_pc(gt[0].cpu().numpy(), f"{taxonomy_ids}_{model_ids}_gt.ply", "/home/lss/Project/PoinTr/result/PCN/separate")
                # print(missing_part.size())
                missing_part = misc.fps(missing_part, 2048)
                # print(partial.size())
                # print(missing_part.size())
                # print(partial.size())
                # print(gt.size())

                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, missing_part = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
           
            # ret, pre_missing_part_feature, true_missing_part_feature = base_model(partial, missing_part)
            ret, pre_missing_part_feature = base_model(partial, None)

            # print(pre_missing_part_feature.size())
            

            # with torch.no_grad():
            #     missing_part_feature = base_model(partial, missing_part)
                # print(missing_part_feature.size())
            
            # feature_loss = base_model.module.get_feature_loss(true_missing_part_feature, pre_missing_part_feature)

            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)

            # _loss = sparse_loss + dense_loss + feature_loss
            _loss = sparse_loss + dense_loss
            # if epoch <= 30:
            #     _loss = 0.5 * sparse_loss + 0.5 * dense_loss + 1 * feature_loss
            # elif epoch <= 60:
            #     _loss = 1 * sparse_loss + 0.5 * dense_loss + 0.5 * feature_loss
            # else:
            #     _loss = 0.5 * sparse_loss + 1 * dense_loss + 0.5 * feature_loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
                # f_losses.update(feature_loss)


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                # print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s Feature_Loss = %.3f lr = %.6f' %
                #             (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                #             ['%.4f' % l for l in losses.val()], f_losses.val(), optimizer.param_groups[0]['lr']), logger = logger)
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
            # train_writer.add_scalar('Loss/Epoch/F_Loss', f_losses.avg(), epoch)
        # print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s F_Losses = %s' %
        #     (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], [f_losses.avg().item()]), logger = logger)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 10:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    train_writer.close()
    val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)

    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                # print(gt.size())
                _, missing_part = get_missing_part(gt, partial)
                missing_part = misc.fps(missing_part, 2048)
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, missing_part = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret, _ = base_model(partial, None)
            coarse_points = ret[0]
            dense_points = ret[1]

            TEST_DIR = f"result/{args.exp_name}/valid/{shapenet_dict[taxonomy_id]}"
            if not os.path.exists(TEST_DIR):
                os.makedirs(TEST_DIR)

            plot_pcd_one_view(
                os.path.join(
                    TEST_DIR, f"{args.exp_name}_{shapenet_dict[taxonomy_id]}_{idx}.png"
                ),
                [
                    partial[0].detach().cpu().numpy(),
                    missing_part[0].detach().cpu().numpy(),
                    coarse_points[0].detach().cpu().numpy(),
                    dense_points[0].detach().cpu().numpy(),
                    gt[0].detach().cpu().numpy(),
                ],
                [
                    "parttial",
                    "m_p",
                    "Coarse",
                    "Dense",
                    "gt",
                ],
                xlim=(-0.7, 0.7),
                ylim=(-0.7, 0.7),
                zlim=(-0.7, 0.7),
            )

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            # dense_points_all = dist_utils.gather_tensor(dense_points, args)
            # gt_all = dist_utils.gather_tensor(gt, args)

            # _metrics = Metrics.get(dense_points_all, gt_all)
            _metrics = Metrics.get(dense_points, gt)
            # _metrics = [dist_utils.reduce_tensor(item, args) for item in _metrics]

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % 200 == 0:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse)
                val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense)
                val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
                gt_ptcloud = gt.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % 20 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):
    start_time = time.time()
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    base_model.eval()  # set model to eval mode

    # stat(base_model, (1, 2048, 3))

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                # print(gt.size())
                _, missing_part = get_missing_part(gt, partial)
                missing_part = misc.fps(missing_part, 3584)
                
                # partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                # partial = partial.cuda()

                ret, _ = base_model(partial, None)
                # ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[1]
                outdir = f"result/PCN/our_model/test/ply_file"
                if not os.path.exists(os.path.join(outdir, taxonomy_id)):
                    os.makedirs(os.path.join(outdir, taxonomy_id))
                if not os.path.exists(os.path.join(outdir, taxonomy_id+'_images')):
                    os.makedirs(os.path.join(outdir, taxonomy_id+'_images'))
                for mm, model_name in enumerate(model_ids):
                    output_file = os.path.join(outdir, taxonomy_id, model_name)
                    write_ply(output_file + '_coarse.ply', coarse_points[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                    write_ply(output_file + '_dense.ply', dense_points[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                    write_ply(output_file + '_gt.ply', gt[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                    write_ply(output_file + '_partial.ply', partial[mm, :].detach().cpu().numpy(), ['x', 'y', 'z'])
                    # output img files
                    img_filename = os.path.join(outdir, taxonomy_id+'_images', model_name+'.jpg')
                    img_filename2 = os.path.join(outdir, taxonomy_id+'_images', model_name+'_partial.jpg')
                    output_img = pc_util.point_cloud_three_views(dense_points[mm, :].detach().cpu().numpy(), diameter=7)
                    output_img = (output_img*255).astype('uint8')
                    output_img2 = pc_util.point_cloud_three_views(partial[mm, :].detach().cpu().numpy(), diameter=7)
                    output_img2 = (output_img2*255).astype('uint8')
                    im = Image.fromarray(output_img)
                    im2 = Image.fromarray(output_img2)
                    im.save(img_filename)
                    im2.save(img_filename2)
            #     TEST_DIR = f"result/PCN/PoinTr/test/{shapenet_dict[taxonomy_id]}"
            #     if not os.path.exists(TEST_DIR):
            #         os.makedirs(TEST_DIR)

            #     plot_pcd_one_view(
            #     os.path.join(
            #         TEST_DIR, f"{shapenet_dict[taxonomy_id]}_{idx}.png"
            #     ),
            #     [
            #         partial[0].detach().cpu().numpy(),
            #         # missing_part[0].detach().cpu().numpy(),
            #         coarse_points[0].detach().cpu().numpy(),
            #         dense_points[0].detach().cpu().numpy(),
            #         gt[0].detach().cpu().numpy(),
            #     ],
            #     [
            #         "parttial",
            #         # "m_p",
            #         "Coarse",
            #         "Dense",
            #         "gt",
            #     ],
            #     xlim=(-0.6, 0.6),
            #     ylim=(-0.6, 0.6),
            #     zlim=(-0.6, 0.6),
            # )

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points ,gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret, _ = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[1]

                    TEST_DIR = f"result/ShapeNet34_Unseen/our_model/test/{args.mode}/{shapenet_dict[taxonomy_id]}"
                    if not os.path.exists(TEST_DIR):
                        os.makedirs(TEST_DIR)

                    plot_pcd_one_view(
                        os.path.join(
                            TEST_DIR, f"{shapenet_dict[taxonomy_id]}_{idx}.png"
                        ),
                        [
                            partial[0].detach().cpu().numpy(),
                            # missing_part[0].detach().cpu().numpy(),
                            coarse_points[0].detach().cpu().numpy(),
                            dense_points[0].detach().cpu().numpy(),
                            gt[0].detach().cpu().numpy(),
                        ],
                        [
                            "parttial",
                            # "m_p",
                            "Coarse",
                            "Dense",
                            "gt",
                        ],
                        xlim=(-0.6, 0.6),
                        ylim=(-0.6, 0.6),
                        zlim=(-0.6, 0.6),
                    )

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)

                    # test_metrics.update(_metrics)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )

                plot_pcd_one_view(
                        os.path.join(
                            target_path, f"{model_id}_{idx:03d}.png"
                        ),
                        [
                            partial[0].detach().cpu().numpy(),
                            dense_points[0].detach().cpu().numpy()
                        ],
                        [
                            "partial",
                            "Pred"
                        ],
                        xlim=(-0.7, 0.7),
                        ylim=(-0.7, 0.7),
                        zlim=(-0.7, 0.7),
                    )

                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    end_time = time.time()
    total_test_time = end_time - start_time
    print("Our_model test time on PCN is: ", total_test_time)
    return 