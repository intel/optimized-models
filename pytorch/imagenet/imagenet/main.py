import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils import mkldnn as mkldnn_utils
import models as m
from models.mobilenet_v2.mobilenet_v2_model import mobilenet_v2

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--ppn', default=1, type=int,
                    help='number of processes on each node of distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--mkldnn', action='store_true', default=False,
                    help='use mkldnn weight cache')
parser.add_argument('--bf16', action='store_true', default=False,
                    help='enable bf16 operator')
parser.add_argument('--ipex', action='store_true', default=False,
                    help='enable Intel_PyTorch_Extension')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA')
parser.add_argument('-i', '--iterations', default=0, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('--iter-calib', default=0, type=int, metavar='N',
                    help='number of iterations when calibration to run')
parser.add_argument('-qe', '--qengine', type=str, default="all",
                    help='Choose qengine to run. \"all\", \"fbgemm\" or \"mkldnn\".'
                    '(DEFAULT: %(default)s)')
parser.add_argument('-w', '--warmup-iterations', default=0, type=int, metavar='N',
                    help='number of warmup iterations to run')
parser.add_argument('--INT8', type=str, default="no_INT8",
                    help='Choose run mode. \"no_INT8\", \"calibration_olny\", \"INT8_only\", \"INT8_and_fp32\".'
                    '(DEFAULT: %(default)s)')
parser.add_argument("-t", "--profile", action='store_true',
                    help="Trigger profile on current topology.")
parser.add_argument("-qs", "--qscheme", type=str, default="perTensor",
                    help="The scheme of quantizer:\"perTensor\", \"perChannel\"")
parser.add_argument("-r", "--reduce_range", action='store_true',
                    help="Choose reduce range flag. True or False.")
parser.add_argument("--performance", action='store_true',
                    help="measure performance only, no accuracy.")

parser.add_argument("--dummy", action='store_true',
                    help="using  dummu data to test the performance of inference")
parser.add_argument('--checkpoint-dir', default='', type=str, metavar='PATH',
                    help='path to user checkpoint (default: none), just for user defined model')
best_acc1 = 0


def main():
    args = parser.parse_args()
    print(args)

    if args.ipex:
        import intel_pytorch_extension as ipex

    # print("args.world_size", args.world_size)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None and args.cuda:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.ppn > 1 or args.multiprocessing_distributed

    if args.gpu is not None and args.cuda:
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = args.ppn

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            print(os.environ["RANK"])
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print("args.dist_backend {}".format(args.dist_backend))
        print("args.dist_url {}".format(args.dist_url))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.workers != 0:
        blocking = True
    else:
        blocking = False
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, blocking=blocking)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, blocking=blocking)

    # define loss function (criterion) and optimizer
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss()

    if (args.INT8 == "FP32_only"):
        model_info = m.GetModelInfo(args.arch)
        float_model_file = os.path.join(model_info['model_dir'], model_info['float_Model'])
        layers_tmp = ((model_info["layers"]).split('/')[-1]).split(' ')
        layers = list(map(int, layers_tmp))
        groups = int(model_info['groups'])
        width_per_group = int(model_info['width_per_group'])
        loaded_model = models.resnet.ResNet(models.resnet.Bottleneck, layers, groups=groups, width_per_group=width_per_group)
        state_dict = torch.load(float_model_file)
        if (model_info['model_name'] == 'resnext101_32x4d'):
            state_dict = state_dict['state_dict']
            import collections
            state_dict_new = collections.OrderedDict()
            for key in state_dict.keys():
                key_new = key.split("module.")[1]
                state_dict_new[key_new] = state_dict[key]
            loaded_model.load_state_dict(state_dict_new)
        else:
            loaded_model.load_state_dict(state_dict)
        loaded_model.to('cpu')
        loaded_model.eval()
        top1 = validate(val_loader, loaded_model, criterion, args, is_INT8=False)
        return

    if (args.INT8 != "no_INT8"):
        model_info = m.GetModelInfo(args.arch)
        float_model_file = os.path.join(model_info['model_dir'], model_info['float_Model'])
        quantized_model_state_dict_file = os.path.join(model_info['model_dir'], args.qscheme + "_reduceRange_" + str(args.reduce_range) + "_" + model_info['quantized_Model_State_Dict'])
        quantized_model_file = os.path.join(model_info['model_dir'], args.qscheme + "_" + model_info['quantized_Model'])
        scripted_quantized_model_file = os.path.join(model_info['model_dir'], args.qscheme + "_reduceRange_" + str(args.reduce_range) + "_" + model_info['scripted_Quantized_Model'])
        layers_tmp = ((model_info["layers"]).split('/')[-1]).split(' ')
        layers = list(map(int, layers_tmp))
        groups = int(model_info['groups'])
        width_per_group = int(model_info['width_per_group'])

        loaded_model = torch.quantization.QuantWrapper(models.resnet.ResNet(QuantizableBottleneck, layers, groups=groups, width_per_group=width_per_group))
        if not os.path.exists(quantized_model_state_dict_file):
            state_dict = torch.load(float_model_file)
            if (model_info['model_name'] == 'resnext101_32x4d'):
                loaded_model.load_state_dict(state_dict['state_dict'])
            else:
                loaded_model.module.load_state_dict(state_dict)
        loaded_model.to('cpu')
        loaded_model.eval()
        if args.qscheme == "perTensor":
            qconfig = torch.quantization.QConfig(activation=torch.quantization.observer.MinMaxObserver.with_args(reduce_range=args.reduce_range),
                              weight=torch.quantization.default_weight_observer)
            loaded_model.qconfig = qconfig
        elif args.qscheme == "perChannel":
            qconfig = torch.quantization.QConfig(activation=torch.quantization.observer.MinMaxObserver.with_args(reduce_range=args.reduce_range),
                              weight=torch.quantization.default_per_channel_weight_observer)
            loaded_model.qconfig = qconfig
        else:
            assert False
        fuse_resnext_modules(loaded_model.module)
        print('Transforming model, qscheme:{}'.format(args.qscheme))
        torch.quantization.prepare(loaded_model, inplace=True)

        if not os.path.exists(quantized_model_state_dict_file):
            # with open(args.model + "_" + args.qscheme + "_prepare_model.txt", "w") as f:
            #    print(loaded_model, file=f)
            # Calibrate
            val_loader_calib = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True, blocking=blocking)
            print("calibration model first..., iteration:{}".format(args.iter_calib))
            validate(val_loader_calib, loaded_model, criterion, args, is_INT8=True, is_calibration=True)
            print('Calibration... done')
            print("convert model......")
            torch.quantization.convert(loaded_model, inplace=True)
            print('Conversion... done')

            # Now, let us serialize the model and also script and serialize it.
            # Serialize using state dict
            print('save state_dict to {}'.format(quantized_model_state_dict_file))
            torch.save(loaded_model.state_dict(),quantized_model_state_dict_file)

            # Serialize without state dict
            # torch.save(myModel,saved_model_dir+args.model+quantized_model_file)
            # test_model = torch.load(quantized_model_file)

            # Script and serialize Does not work with mode@opt
            scriptedModel = torch.jit.script(loaded_model)
            print('save scriptedModel to {}'.format(scripted_quantized_model_file))
            torch.jit.save(scriptedModel,scripted_quantized_model_file)

            print("saved_model_dir:{}".format(model_info['model_dir']))
            print('Serialization done')
            if args.INT8 == 'calibration_olny':
                return
        else:
            print("convert model......")
            torch.quantization.convert(loaded_model, inplace=True)

        state_dict_quantized = torch.load(quantized_model_state_dict_file)

        if args.qengine == 'fbgemm' or args.qengine == 'all':
            torch.backends.quantized.engine = 'fbgemm'
            print('Loaded quantized model:state_dict')
            loaded_model.load_state_dict(state_dict_quantized)
            print('Testing on fbgemm')
            top1 = validate(val_loader, loaded_model, criterion, args, is_INT8=True)
        if args.qengine == 'mkldnn' or args.qengine == 'all':
            torch.backends.quantized.engine = 'mkldnn'
            print('Loaded quantized model:state_dict')
            if (model_info['model_name'] == 'resnext101_32x4d' or \
                model_info['model_name'] == 'resnet50'):
                state_dict_quantized['module.fc.scale'] = 1
                state_dict_quantized['module.fc.zero_point'] = 0
            loaded_model.load_state_dict(state_dict_quantized)
            print('Testing on mkldnn')
            top1 = validate(val_loader, loaded_model, criterion, args, is_INT8=True)
        if args.INT8 == 'INT8_only':
            return

    if args.gpu is not None and args.cuda:
        print("Use GPU: {} for training".format(args.gpu))

    print("Use Instance: {} for training".format(gpu))
    print("Use num threads: {} for training".format(torch.get_num_threads()))
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    if args.arch == "resnext101_32x4d" and args.pretrained:
        if args.gpu is None:
            checkpoint = torch.load(args.checkpoint_dir)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.checkpoint_dir, map_location=loc)
        model = models.__dict__[args.arch](pretrained=False)
        # our saved checkpoint may have module key, getted by calling nn.DataParallel
        # see https://discuss.pytorch.org/t/does-torch-jit-script-support-custom-operators/65759/2
        import collections
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model = build_model(args.arch, args.pretrained, args.mkldnn)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None and args.cuda:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            if args.cuda:
                model.cuda()
                print("create DistributedDataParallel in GPU")
            else:
                print("create DistributedDataParallel in CPU")
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set

            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and args.cuda:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            if args.cuda:
                model.cuda()
        else:
            model = torch.nn.DataParallel(model)
            if args.cuda:
                model.cuda()

    if args.ipex:
        model = model.to(device = 'dpcpp:0')

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None and args.cuda:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.cuda:
        cudnn.benchmark = True

    if args.evaluate:
        if args.mkldnn and not args.cuda:
            print("using mkldnn model to do inference\n")
        validate(val_loader, model, criterion, args)
        return

    if args.mkldnn and not args.cuda:
        print("using mkldnn modle to training\n")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        print("run epoch '{}'".format(epoch))
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if not args.performance:
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if args.rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)


def build_model(arch, pretrained=False, use_mkldnn=False):
    if arch == "MobileNetV2" or arch == "mobilenet_v2":
        return mobilenet_v2(pretrained)
    else:
        return models.__dict__[arch](pretrained=pretrained)

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.iterations > 0 and i >= (args.warmup_iterations + args.iterations):
            break
        # measure data loading time
        if i >= args.warmup_iterations:
            data_time.update(time.time() - end)

        if args.gpu is not None and args.cuda:
            images = images.cuda(args.gpu, non_blocking=True)

        if args.cuda:
            target = target.cuda(args.gpu, non_blocking=True)


        if args.bf16 and not args.cuda:
            images = images.to_mkldnn(torch.bfloat16)
        elif args.mkldnn and not args.cuda:
            images = images.to_mkldnn()

        if args.ipex:
            images = images.to(device = 'dpcpp:0')

        # compute output
        output = model(images)

        if args.mkldnn and not args.cuda:
            output = output.to_dense()

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        if args.mkldnn and not args.cuda:
            images = images.to_dense()

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        if i >= args.warmup_iterations:
            batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    if args.performance:
        batch_size = train_loader.batch_size
        latency = batch_time.avg / batch_size * 1000
        perf = batch_size/batch_time.avg
        print('training latency %3.0f ms on %d epoch'%(latency, epoch))
        print('training performance %3.0f fps on %d epoch'%(perf, epoch))


def validate(val_loader, model, criterion, args, is_INT8=False, is_calibration=False):
    if is_calibration:
        iterations = args.iter_calib
        warmup = 0
    else:
        iterations = args.iterations
        warmup = args.warmup_iterations
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.evaluate and args.mkldnn and not args.cuda and not is_INT8:
        if args.bf16:
            model = mkldnn_utils.to_mkldnn(model, torch.bfloat16)
        else:
            model = mkldnn_utils.to_mkldnn(model)
        # TODO using mkldnn weight cache

    if args.dummy:
        images = torch.randn(args.batch_size, 3, 224, 224)
        target = torch.arange(1, args.batch_size + 1).long()

        if not is_INT8:
            if args.gpu is not None and args.cuda:
                images = images.cuda(args.gpu, non_blocking=True)
            if args.cuda:
                target = target.cuda(args.gpu, non_blocking=True)

            if args.bf16 and not args.cuda:
                images = images.to_mkldnn(torch.bfloat16)
            elif args.mkldnn and not args.cuda:
                images = images.to_mkldnn()

        if args.ipex:
            images = images.to(device = 'dpcpp:0')

        number_iter = len(val_loader)
        with torch.no_grad():
            for i in range(number_iter):
                if not args.evaluate or iterations == 0 or i < iterations + warmup:
                    if i >= warmup:
                        end = time.time()
                    # compute output
                    output = model(images)

                    # measure elapsed time
                    if i >= warmup:
                        batch_time.update(time.time() - end)

                    if args.mkldnn and not args.cuda and not is_INT8:
                        output = output.to_dense()

                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))

                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    if i % args.print_freq == 0:
                        progress.display(i)
                elif i == iterations + warmup:
                    break

            if args.profile:
                # print("export profiling file to {}".format(torch.backends.quantized.engine + "_result.json"))
                with torch.autograd.profiler.profile() as prof:
                    output = model(images)
                prof.export_chrome_trace(torch.backends.quantized.engine + "_result.json")
                table_res = prof.key_averages().table(sort_by="cpu_time_total")
                print(table_res)
                save_profile_result(torch.backends.quantized.engine + "_result_average.xlsx", table_res)

            # TODO: this should also be done with the ProgressMeter
            if args.evaluate:
                batch_size = val_loader.batch_size
                latency = batch_time.avg / batch_size * 1000
                perf = batch_size/batch_time.avg
                print('inference latency %3.0f ms'%latency)
                print('inference performance %3.0f fps'%perf)

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    else:
        with torch.no_grad():
            for i, (images, target) in enumerate(val_loader):
                if not args.evaluate or iterations == 0 or i < iterations + warmup:
                    if i >= warmup:
                        end = time.time()
                    if not is_INT8:
                        if args.gpu is not None and args.cuda:
                            images = images.cuda(args.gpu, non_blocking=True)
                        if args.cuda:
                            target = target.cuda(args.gpu, non_blocking=True)

                        if args.bf16 and not args.cuda:
                            images = images.to_mkldnn(torch.bfloat16)
                        elif args.mkldnn and not args.cuda:
                            images = images.to_mkldnn()

                    if args.ipex:
                        images = images.to(device = 'dpcpp:0')

                    # compute output
                    output = model(images)

                    # measure elapsed time
                    if i >= warmup:
                        batch_time.update(time.time() - end)

                    if args.mkldnn and not args.cuda and not is_INT8:
                        output = output.to_dense()

                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))

                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    if i % args.print_freq == 0:
                        progress.display(i)
                elif i == iterations + warmup:
                    break

            if args.profile:
                # print("export profiling file to {}".format(torch.backends.quantized.engine + "_result.json"))
                with torch.autograd.profiler.profile() as prof:
                    output = model(images)
                prof.export_chrome_trace(torch.backends.quantized.engine + "_result.json")
                table_res = prof.key_averages().table(sort_by="cpu_time_total")
                print(table_res)
                save_profile_result(torch.backends.quantized.engine + "_result_average.xlsx", table_res)

            # TODO: this should also be done with the ProgressMeter
            if args.evaluate:
                batch_size = val_loader.batch_size
                latency = batch_time.avg / batch_size * 1000
                perf = batch_size/batch_time.avg
                print('inference latency %3.0f ms'%latency)
                print('inference performance %3.0f fps'%perf)

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# We have modified the module below with the following changes
#     1. Unique names for all ReLU modules
#     2. Replace out+=identity with a module created using nn.quantized.Floatfunctional(). This is needed
#     to collect statistics on the activations at the output of the addition with the skip connection.
class QuantizableBottleneck(torch.nn.Module):
    __constants__ = ['downsample']
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(QuantizableBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = models.resnet.conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = models.resnet.conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = models.resnet.conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add.add(out, identity)
        out = self.relu3(out)

        return out


# Quantization requires batch norms to be folded into convolutions as scalar multiplies are not yet supported.
# In addition, fusion provides for faster execution and is recommended.
def fuse_resnext_modules(model):
    torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu']], inplace=True)
    for mod in model.modules():
        if type(mod) == QuantizableBottleneck:
            torch.quantization.fuse_modules(mod, [['conv1', 'bn1', 'relu1']], inplace=True)
            torch.quantization.fuse_modules(mod, [['conv2', 'bn2', 'relu2']], inplace=True)
            torch.quantization.fuse_modules(mod, [['conv3', 'bn3']], inplace=True)
            if mod.downsample:
                torch.quantization.fuse_modules(mod.downsample, [['0', '1']], inplace=True)

def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0,j,keys[j])

    lines = table.split("\n")
    for i in range(3,len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()

def save_profile_result_json(filename, table):
    with open(filename, "w") as outfile:
        keys = ["Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
        "CPU time avg", "CUDA total %", "CUDA total", "CUDA time avg", "Number of Calls"]
        outfile.write("{0} , {1} , {2} , {3} , {4}, {5}, {6}, {7}, {8}, {9}\n".format("Name", \
        "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
        "CPU time avg", "CUDA total %", "CUDA total", "CUDA time avg", "Number of Calls"))
        lines = table.split("\n")
        for i in range(3,len(lines)-4):
            words = lines[i].split(" ")
            j = 0
            for word in words:
                if not word == "":
                    outfile.write("{0} , ".format(word))
                    j += 1
            outfile.write("\n")


if __name__ == '__main__':
    main()
