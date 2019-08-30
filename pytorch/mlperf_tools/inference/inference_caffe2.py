"""main func to run inference"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from random import randint

import os
import sys
import timeit
import logging
import copy
import numpy as np
import onnx
from caffe2.proto import caffe2_pb2
from caffe2.python import core, utils
from caffe2.python import workspace as ws
from caffe2.python import transformations as tf

import inference.models as m
from common import common_caffe2 as cc2

def PrintNetDef(model, net="predict_net"):
    """print weight or predict"""
    if net not in ["predict_net", "init_net", "predict_net_int8", "onnx_model"]:
        logging.error("Unsupported net def file {}".format(net))
        return
    dummy = 0
    model_info = m.GetModelInfo(model)
    with open(model_info[net]) as p:
        if model_info["model_type"] == "prototext" or model_info[net].split('.')[-1] == "pbtxt":
            import google.protobuf.text_format as ptxt
            predict_def = ptxt.Parse(p.read(), caffe2_pb2.NetDef())
        else:
            predict_def = caffe2_pb2.NetDef()
            predict_def.ParseFromString(p.read())
        if dummy == 1 and net != "predict_net":
            for op in predict_def.op:
                op.type = 'ConstantFill'
                if op.output[0] != 'data':
                    arg_new = caffe2_pb2.Argument()
                    arg_new.name = 'value'
                    arg_new.f = 0.01
                    for i, arg in enumerate(op.arg[::-1]):
                        if arg.name == 'shape':
                            continue
                        else:
                            del op.arg[len(op.arg) - 1 - i]
                    op.arg.extend([arg_new])
                else:
                    arg_new = caffe2_pb2.Argument()
                    arg_new.name = 'shape'
                    arg_new.ints.extend([1])
                    for i, arg in enumerate(op.arg[::-1]):
                        if arg.name == 'shape':
                            del op.arg[len(op.arg) - 1 - i]
                    op.arg.extend([arg_new])
        logging.critical("The {} definition of model {}:\n{}"
                         .format(net, model_info["model_name"], predict_def))
    if dummy == 1:
        with open("{}.pbtxt".format(model_info["model_name"]), "w") as fid:
            fid.write(str(predict_def))
        with open("{0}.pb".format(model_info["model_name"]), "w") as fid:
            fid.write(predict_def.SerializeToString())


def Run(args, extra_args):
    """main func of run inference"""
    if not m.IsSupported(args.model):
        logging.error("Not supported model: {}".format(args.model))
        m.ShowModels()
        return
    images_path = None
    if args.images_path:
        images_path = os.path.abspath(args.images_path)
    elif "CAFFE2_INF_IMG_PATH" in os.environ:
        images_path = os.path.abspath(os.environ["CAFFE2_INF_IMG_PATH"])
    if not args.dummydata and not os.path.isdir(images_path):
        logging.error("Can not find image path {}.".format(images_path))
        return
    labels = None
    validation = None
    if args.label_file:
        labels = cc2.LoadLabels(args.label_file)
    elif args.validation_file:
        validation = cc2.LoadValidation(args.validation_file)
    elif "CAFFE2_INF_LABEL_FILE" in os.environ:
        labels = cc2.LoadLabels(os.environ["CAFFE2_INF_LABEL_FILE"])
    elif "CAFFE2_INF_VAL_FILE" in os.environ:
        validation = cc2.LoadValidation(os.environ["CAFFE2_INF_VAL_FILE"])
    else:
        logging.warning("No validation or label file!")
    if args.annotations:
        apath = args.annotations
    elif args.model == 'faster-rcnn' or args.model == 'ssd':
        logging.error("currently only support fasterrcnn and ssd for voc dataset, so will just collect performance")
    iterations = args.iterations if args.iterations else sys.maxsize
    warmup_iter = args.warmup_iterations if args.warmup_iterations > 0 else 0
    optimization = []
    if args.optimization:
        optimization = [opt.strip() for opt in args.optimization.split(',')]
    batch_size = 1
    if args.batch_size:
        batch_size = int(args.batch_size)
        if batch_size <= 0:
            logging.error("Invalid batch size {}. Exit!".format(batch_size))
            return
    logging.warning("Run Caffe2 in inference mode with args:\n{}"
                    .format(vars(args)))
    model_info = m.GetModelInfo(args.model)
    logging.warning("The inference inputs of {0} model:\n{1}"
                    .format(
                        args.model,
                        {str(k): str(v) for k, v in model_info.items()}
                    ))

    crop_size = int(model_info["crop_size"])
    if args.crop_size:
        crop_size = args.crop_size

    need_normalize = False
    if model_info["need_normalize"]:
        need_normalize = True

    mean = 128
    image_mean = str(model_info["image_mean"])
    if str(model_info["image_mean"]) != 'None' and image_mean.split('.')[-1] == "binaryproto":
        pass

    mean_tmp = ((model_info["image_mean"]).split('/')[-1]).split(' ')
    if str(model_info["image_mean"]) != 'None' and len(mean_tmp) == 3:
        if need_normalize:
            mean = np.zeros([3, crop_size, crop_size], dtype=np.float)
            mean[0, :, :] = float(mean_tmp[0])  # 104
            mean[1, :, :] = float(mean_tmp[1])  # 117
            mean[2, :, :] = float(mean_tmp[2])  # 124
        else:
            mean = np.zeros([3, crop_size, crop_size], dtype=np.int32)
            mean[0, :, :] = int(mean_tmp[0])  # 104
            mean[1, :, :] = int(mean_tmp[1])  # 117
            mean[2, :, :] = int(mean_tmp[2])  # 124

    scale = [1]
    if str(model_info["scale"]) != '':
        scale = (model_info["scale"]).split(' ')
    rescale_size = 256
    if str(model_info["rescale_size"]) != '':
        rescale_size = int(model_info["rescale_size"])
    color_format = "BGR"
    if str(model_info["color_format"]) != '':
        color_format = model_info["color_format"]

    model_start_time = timeit.default_timer()
    if args.onnx_model:
        def remove_arg(op, name):
            for i in range(len(op.arg)):
                if op.arg[i].name == name:
                    del op.arg[i]
                    return True
            return False

        init_def, predict_def = cc2.OnnxToCaffe2(model_info["onnx_model"])
        if model_info["model_name"] == "resnet50":
            ws.RunNetOnce(init_def)
            dense_final_kernel = ws.FetchBlob("resnet_model/dense/kernel:0")
            dense_kernel_new = np.zeros([dense_final_kernel.shape[1], dense_final_kernel.shape[0]]).astype(np.float32)
            for i in range(dense_final_kernel.shape[0]):
                for j in range(dense_final_kernel.shape[1]):
                    dense_kernel_new[j][i] = dense_final_kernel[i][j]
            net = core.Net("resnet50")
            net.Proto().CopyFrom(init_def)
            init_proto = net.Proto()
            for i, op in enumerate(init_proto.op):
                if op.output[0] == "resnet_model/dense/kernel:0":
                    remove_arg(init_proto.op[i], "values")
                    init_proto.op[i].arg.extend([utils.MakeArgument("values", dense_kernel_new)])
                    remove_arg(init_proto.op[i], "shape")
                    init_proto.op[i].arg.extend([utils.MakeArgument("shape", dense_kernel_new.shape)])
        else:
            init_proto = init_def
        with open("init_net.pbtxt".format(model_info["model_name"]), "w") as fid:
            fid.write(str(init_proto))
        with open("predict_net.pbtxt".format(model_info["model_name"]), "w") as fid:
            fid.write(str(predict_def))
        return
    else:
        if args.int8_model or args.int8_cosim:
            init_file = model_info["init_net_int8"]
            predict_file = model_info["predict_net_int8"]
        else:
            init_file = model_info["init_net"]
            predict_file = model_info["predict_net"]
        with open(init_file, "rb") as i:
            print(model_info)
            if model_info["model_type"] == "prototext" or init_file.split('.')[-1] == "pbtxt":
                import google.protobuf.text_format as ptxt
                init_def = ptxt.Parse(i.read(), caffe2_pb2.NetDef())
            else:
                init_def = caffe2_pb2.NetDef()
                init_def.ParseFromString(i.read())
        with open(predict_file, "rb") as p:
            print(model_info["model_type"])
            if model_info["model_type"] == "prototext" or predict_file.split('.')[-1] == "pbtxt":
                import google.protobuf.text_format as ptxt
                predict_def = ptxt.Parse(p.read(), caffe2_pb2.NetDef())
            else:
                predict_def = caffe2_pb2.NetDef()
                predict_def.ParseFromString(p.read())
        if args.int8_cosim:
            with open(model_info["predict_net"], "rb") as p:
                if model_info["model_type"] == "prototext" or model_info["predict_net"].split('.')[-1] == "pbtxt":
                    import google.protobuf.text_format as ptxt
                    cosim_predict_def = ptxt.Parse(p.read(), caffe2_pb2.NetDef())
                else:
                    cosim_predict_def = caffe2_pb2.NetDef()
                    cosim_predict_def.ParseFromString(p.read())
    #cc2.SaveAsOnnxModel(init_def, predict_def, (1, 3, crop_size, crop_size),
    #            model_info["model_name"] + "_onnx.pb")

    dev_map = {
        "cpu": caffe2_pb2.CPU,
        "gpu": caffe2_pb2.CUDA,
        "cuda": caffe2_pb2.CUDA,
        "mkldnn": caffe2_pb2.MKLDNN,
        "opengl": caffe2_pb2.OPENGL,
        "opencl": caffe2_pb2.OPENCL,
        "ideep": caffe2_pb2.IDEEP,
    }
    device_opts = caffe2_pb2.DeviceOption()
    if  args.device.lower() in dev_map:
        device_opts.device_type = dev_map[args.device.lower()]
    else:
        logging.error("Wrong device {}. Exit!".format(args.device))
        return
    device_opts_cpu = caffe2_pb2.DeviceOption()
    device_opts_cpu.device_type = caffe2_pb2.CPU
    if model_info["allow_device_override"]:
        if (args.model == 'faster-rcnn' and args.device.lower() == 'gpu'):
            cc2.UpdateDeviceOption(device_opts_cpu, init_def)
        else:
            cc2.UpdateDeviceOption(device_opts, init_def)
    if model_info["allow_device_override"]:
        cc2.UpdateDeviceOption(device_opts, predict_def)
    # search params shape to replace the 0 with 1 when ideep and throw warning
    if args.device.lower() == 'ideep':
        cc2.FillZeroParamsWithOne(init_def)

    if os.environ.get('DEBUGMODE') == "1":
        with open("{0}_origin_init_net.pb".format(model_info["model_name"]), "w") as fid:
            fid.write(init_def.SerializeToString())
        with open("{}_origin_init_net.pbtxt".format(model_info["model_name"]), "w") as fid:
            fid.write(str(init_def))
        with open("{0}_origin_predict_net.pb".format(model_info["model_name"]), "w") as fid:
            fid.write(predict_def.SerializeToString())
        with open("{}_origin_predict_net.pbtxt".format(model_info["model_name"]), "w") as fid:
            fid.write(str(predict_def))

    if model_info["model_type"] == "caffe legacy":
        cc2.MergeScaleBiasInBN(predict_def)
        cc2.RemoveUselessExternalInput(predict_def)
        if args.int8_cosim:
            cc2.MergeScaleBiasInBN(cosim_predict_def)
            cc2.RemoveUselessExternalInput(cosim_predict_def)

    init_data = np.random.rand(batch_size, 3, crop_size, crop_size).astype(np.float32)
    init_label = np.ones((batch_size), dtype=np.int32)
    if args.cosim:
        def_ws_name = ws.CurrentWorkspace()
        inf_ws_name = "__inf_ws__"
        ws.SwitchWorkspace(inf_ws_name, True)
        ws.FeedBlob(str(predict_def.op[0].input[0]), init_data, device_opts)
        ws.RunNetOnce(init_def)
        cosim_ws_name = "__cosim_ws__"
        ws.SwitchWorkspace(cosim_ws_name, True)
        device_cosim = caffe2_pb2.DeviceOption()
        device_cosim.device_type = dev_map["cpu"]
        cosim_init_def = copy.deepcopy(init_def)
        cc2.UpdateDeviceOption(device_cosim, cosim_init_def)
        ws.FeedBlob(str(predict_def.op[0].input[0]), init_data, device_cosim)
        ws.RunNetOnce(cosim_init_def)
        cosim_predict_def = copy.deepcopy(predict_def)
        cc2.UpdateDeviceOption(device_cosim, cosim_predict_def)
    elif args.int8_cosim:
        inf_ws_name = "__int8_ws__"
        ws.SwitchWorkspace(inf_ws_name, True)
        ws.FeedBlob(str(predict_def.op[0].input[0]), init_data, device_opts)
        ws.RunNetOnce(init_def)

        net = core.Net(model_info["model_name"])
        net.Proto().CopyFrom(predict_def)
        tf.optimizeForMKLDNN(net)
        predict_def = net.Proto()

        cosim_ws_name = "__fp32_ws__"
        ws.SwitchWorkspace(cosim_ws_name, True)
        ws.FeedBlob(str(cosim_predict_def.op[0].input[0]), init_data, device_opts)
        ws.RunNetOnce(init_def)
        cc2.UpdateDeviceOption(device_opts, cosim_predict_def)

        net = core.Net(model_info["model_name"])
        net.Proto().CopyFrom(cosim_predict_def)
        tf.optimizeForMKLDNN(net)
        cosim_predict_def = net.Proto()
    else:
        # ApplyOptimizations(init_def, predict_def, model_info, optimization)
        if args.int8_model and args.quantize_input:
            ws.FeedBlob("X", init_data, device_opts)
            sw2nhwc = core.CreateOperator(
                "NCHW2NHWC",
                ["X"],
                ["X_nhwc"],
                device_option=device_opts
                )
            quantize_X = core.CreateOperator(
                "Int8Quantize",
                ["X_nhwc"],
                [str(predict_def.op[0].input[0])],
                engine="DNNLOWP",
                device_option=device_opts,
                Y_zero_point=128,
                Y_scale=1.18944883347,
                )
            ws.RunOperatorOnce(sw2nhwc)
            ws.RunOperatorOnce(quantize_X)
        else:
            ws.FeedBlob(str(predict_def.op[0].input[0]), init_data, device_opts)

        if os.environ.get('DEBUGMODE') == "1":
            cc2.SetOpName(predict_def)

        ws.RunNetOnce(init_def)
        net = core.Net(model_info["model_name"])
        net.Proto().CopyFrom(predict_def)
        if args.device.lower() == 'ideep' and not args.noptimize:
            logging.warning('Optimizing module {} ....................'
                            .format(model_info["model_name"]))
            tf.optimizeForMKLDNN(net)
        predict_def = net.Proto()

        # ws.CreateNet(predict_def)
        if (args.model == 'faster-rcnn' and args.device.lower() == 'gpu'):
            new_predict_def, _ = core.InjectCrossDeviceCopies(core.Net(predict_def))
            net = core.Net(new_predict_def._net)
            #ws.CreateNet(new_predict_def._net)
            predict_def = new_predict_def._net

        if os.environ.get('DEBUGMODE') == "1":
            with open("{0}_opt_init_net.pb".format(model_info["model_name"]), "w") as fid:
                fid.write(init_def.SerializeToString())
            with open("{}_opt_init_net.pbtxt".format(model_info["model_name"]), "w") as fid:
                fid.write(str(init_def))
            with open("{0}_opt_predict_net.pb".format(model_info["model_name"]), "w") as fid:
                fid.write(predict_def.SerializeToString())
            with open("{}_opt_predict_net.pbtxt".format(model_info["model_name"]), "w") as fid:
                fid.write(str(predict_def))

        if args.profile or predict_def.op[-1].type == 'Accuracy':
            #predict_model = model_helper.ModelHelper("predict")
            #predict_model.net = core.Net(predict_def)
            #predict_model.net.name = predict_def.name
            if predict_def.op[-1].type == 'Accuracy':
                label = net.AddExternalInput('label')
                if args.device.lower() == 'gpu':
                    ws.FeedBlob(label, init_label, device_opts)
                else:
                    ws.FeedBlob(label, init_label, device_opts_cpu)
                for i, op in enumerate(predict_def.op):
                    if op.type == 'Accuracy':
                        if args.device.lower() == 'gpu':
                            print(device_opts.device_type)
                            ws.FeedBlob(str(predict_def.op[i].output[0]), init_label, device_opts)
                        else:
                            ws.FeedBlob(str(predict_def.op[i].output[0]), init_label, device_opts_cpu)
            #if (args.model == 'faster-rcnn' and args.device.lower() == 'gpu'):
            #    ws.CreateNet(net, True)
            #else:
            ws.CreateNet(net)
            if  args.profile:
                #ob = predict_model.net.AddObserver("TimeObserver")
                ob = net.AddObserver("TimeObserver")
        else:
            #if (args.model == 'faster-rcnn' and args.device.lower() == 'gpu'):
            #    ws.CreateNet(net, True)
            #else:
            ws.CreateNet(net)

    model_elapsed_time = timeit.default_timer() - model_start_time

    outputs = []
    accuracy_top1 = []
    accuracy_top5 = []
    img_time = 0
    comp_time = 0
    processed_images = 0
    images = []
    labels = []
    fnames = []
    if args.dummydata:
        init_label = np.ones((batch_size), dtype=np.int32)
        if args.dummyvalue != "random":
            imgs = np.full((batch_size, 3, crop_size, crop_size), float(args.dummyvalue), dtype=np.float32)
        else:
            imgs = np.random.rand(batch_size, 3, crop_size, crop_size).astype(np.float32)
        for i in range(iterations):
            labels.append(init_label)
            images.append(imgs)
    else:
        process_data_start_time = timeit.default_timer()
        images, fnames = cc2.ImageProc.BatchImages(images_path, batch_size, iterations)
        process_data_elapsed_time = timeit.default_timer() - process_data_start_time
        logging.warning("processdata time = {}".format(process_data_elapsed_time))
    logging.warning("Start warmup {} iterations...".format(warmup_iter))
    forchw = 1
    if 'style-transfer' in args.model:
        forchw = 0
    wi = warmup_iter-1
    while warmup_iter and not args.cosim:
        warmup_iter -= 1
        if args.dummydata:
            imgs = images[wi-warmup_iter]
            oshape = (crop_size, crop_size, 3)
        else:
            r = randint(0, len(images) - 1)
            if model_info["model_type"] == "mlperf legacy vgg":
                imgs, oshape = cc2.ImageProc.PreprocessImagesMLPerfVGG(images[r])
            elif model_info["model_type"] == "mlperf legacy mb":
                imgs, oshape = cc2.ImageProc.PreprocessImagesMLPerfMB(images[r])
            else:
                imgs, oshape = cc2.ImageProc.PreprocessImages(
                    images[r], crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format)
            #imgs, oshape = cc2.ImageProc.PreprocessImagesByThreading(
            #    images[r], crop_size, rescale_size, mean, scale, forchw)
        if args.model == 'faster-rcnn':
            # init_def_update=copy.deepcopy(init_def)
            # cc2.UpdateImgInfo(oshape, init_def_update, predict_def, crop_size)
            # ws.RunNetOnce(init_def_update)
            im_info_name, blob = cc2.CreateIMBlob(oshape, predict_def, crop_size)
            if args.device.lower() == 'gpu':
                ws.FeedBlob(im_info_name, blob, device_opts_cpu)
            else:
                ws.FeedBlob(im_info_name, blob, device_opts)
        if 'style-transfer' in args.model or (args.model == 'faster-rcnn' and args.device.lower() == 'gpu'):
            ws.FeedBlob(str(predict_def.op[0].input[0]), imgs)
        else:
            if args.int8_model and (args.quantize_input or args.quantize_input_once):
                #args.quantize_input_once = False
                ws.FeedBlob("X", imgs, device_opts)
                sw2nhwc = core.CreateOperator(
                    "NCHW2NHWC",
                    ["X"],
                    ["X_nhwc"],
                    device_option=device_opts
                    )

                quantize_X = core.CreateOperator(
                    "Int8Quantize",
                    ["X_nhwc"],
                    [str(predict_def.op[0].input[0])],
                    engine="DNNLOWP",
                    device_option=device_opts,
                    Y_zero_point=128,
                    Y_scale=1.18944883347,
                    )
                ws.RunOperatorOnce(sw2nhwc)
                ws.RunOperatorOnce(quantize_X)
            else:
                ws.FeedBlob(str(predict_def.op[0].input[0]), imgs, device_opts)
        if predict_def.op[-1].type == 'Accuracy' and args.dummydata:
            init_label = labels[wi-warmup_iter]
            ws.FeedBlob(str(predict_def.op[-1].input[1]), init_label, device_opts_cpu)
            ws.FeedBlob(str(predict_def.op[-2].input[1]), init_label, device_opts_cpu)
        elif predict_def.op[-1].type == 'Accuracy' and len(validation) > 0:
            batch_fname = fnames[r]
            init_label = np.ones((len(fnames[r])), dtype=np.int32)
            for j in range(len(fnames[r])):
                init_label[j] = validation[batch_fname[j]]

            if args.device.lower() == 'gpu':
                ws.FeedBlob(str(predict_def.op[-1].input[1]), init_label, device_opts)
                ws.FeedBlob(str(predict_def.op[-2].input[1]), init_label, device_opts)
            else:
                ws.FeedBlob(str(predict_def.op[-1].input[1]), init_label, device_opts_cpu)
                ws.FeedBlob(str(predict_def.op[-2].input[1]), init_label, device_opts_cpu)

        #if args.profile or predict_def.op[-1].type == 'Accuracy':
        #    ws.RunNet(net)
        #else:
        ws.RunNet(net)

    logging.warning("Start running performance")
    for k, raw in enumerate(images):
        processed_images += len(raw)
        img_start_time = timeit.default_timer()
        if args.dummydata:
            imgs = images[0]
            oshape = (crop_size, crop_size)
        else:
            if model_info["model_type"] == "mlperf legacy vgg":
                imgs, oshape = cc2.ImageProc.PreprocessImagesMLPerfVGG(raw)
            elif model_info["model_type"] == "mlperf legacy mb":
                imgs, oshape = cc2.ImageProc.PreprocessImagesMLPerfMB(raw)
            else:
                imgs, oshape = cc2.ImageProc.PreprocessImages(
                    raw, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format)
            #imgs, oshape = cc2.ImageProc.PreprocessImagesByThreading(raw, crop_size, rescale_size, mean, scale, forchw)
        # im_info_name, blob = cc2.CreateIMBlob(oshape, predict_def, crop_size)
        # ws.FeedBlob(im_info_name, blob, device_opts)
        # x = ws.FetchBlob(im_info_name)

        init_label = None
        if predict_def.op[-1].type == 'Accuracy' and args.dummydata:
            init_label = labels[k]
        elif predict_def.op[-1].type == 'Accuracy' and len(validation) > 0:
            batch_fname = fnames[k]
            init_label = np.ones((len(fnames[k])), dtype=np.int32)
            for j in range(len(fnames[k])):
                init_label[j] = validation[batch_fname[j]]

        if args.model == 'faster-rcnn':
            # init_def_update=copy.deepcopy(init_def)
            # cc2.UpdateImgInfo(oshape, init_def_update, predict_def, crop_size)
            im_info_name, blob = cc2.CreateIMBlob(oshape, predict_def, crop_size)

            if args.cosim:
                ws.SwitchWorkspace(inf_ws_name, True)
                # ws.RunNetOnce(init_def_update)
                ws.FeedBlob(im_info_name, blob, device_opts)
                ws.SwitchWorkspace(cosim_ws_name, True)
                # cosim_init_def_update=copy.deepcopy(cosim_init_def)
                # cc2.UpdateImgInfo(oshape, cosim_init_def_update, cosim_predict_def, crop_size)
                # ws.RunNetOnce(cosim_init_def_update)
                ws.FeedBlob(im_info_name, blob, device_cosim)
            else:
                # ws.RunNetOnce(init_def_update)
                if args.device.lower() == 'gpu':
                    ws.FeedBlob(im_info_name, blob, device_opts_cpu)
                else:
                    ws.FeedBlob(im_info_name, blob, device_opts)
        # logging.info("output blob is: {}".format(x))
        # imgs = ImageProc.PreprocessImages(raw, crop_size, mean)
        img_elapsed_time = timeit.default_timer() - img_start_time
        img_time += img_elapsed_time
        if args.cosim or args.int8_cosim:
            ws.SwitchWorkspace(cosim_ws_name)
            if args.cosim:
                ws.FeedBlob(
                    str(cosim_predict_def.op[0].input[0]), imgs, device_cosim)
            else:
                ws.FeedBlob(
                    str(cosim_predict_def.op[0].input[0]), imgs, device_opts)
            ws.SwitchWorkspace(inf_ws_name)
            ws.FeedBlob(str(predict_def.op[0].input[0]), imgs, device_opts)
            for i in range(len(predict_def.op)):
                ws.SwitchWorkspace(inf_ws_name)
                inf_inputs = []
                for inp in predict_def.op[i].input:
                    inf_inputs.append(ws.FetchBlob(str(inp)))
                ws.RunOperatorOnce(predict_def.op[i])
                inf_results = []
                for res in predict_def.op[i].output:
                    inf_results.append(ws.FetchBlob(str(res)))
                ws.SwitchWorkspace(cosim_ws_name)
                cosim_inputs = []
                for inp in cosim_predict_def.op[i].input:
                    cosim_inputs.append(ws.FetchBlob(str(inp)))
                ws.RunOperatorOnce(cosim_predict_def.op[i])
                cosim_results = []
                for res in cosim_predict_def.op[i].output:
                    cosim_results.append(ws.FetchBlob(str(res)))
                if len(inf_inputs) != len(cosim_inputs):
                    logging.error("Wrong number of inputs")
                if len(inf_results) != len(cosim_results):
                    logging.error("Wrong number of outputs")
                    return
                if args.cosim:
                    tol = {'atol': 1e-02, 'rtol': 1e-03}
                else:
                    tol = {'atol': 5, 'rtol': 1e-01}
                logging.warning("begin to check op[{}] {} input".format(i, predict_def.op[i].type))
                for k in range(len(inf_inputs)):
                    if predict_def.op[i].input[k][0] == '_':
                        continue
                    #cc2.assert_allclose(inf_inputs[k], cosim_inputs[k], **tol)
                    #if not np.allclose(inf_inputs[k], cosim_inputs[k], **tol):
                    #    logging.error("Failure in cosim {} op {} input {}"
                    #        .format(
                    #        i,
                    #        predict_def.op[i].type,
                    #        predict_def.op[i].input[k]))
                    #    logging.error(inf_inputs[k].flatten())
                    #    logging.error(cosim_inputs[k].flatten())
                    #    logging.error("Max error: {}"
                    #        .format(
                    #        np.max(np.abs(
                    #            inf_inputs[k] - cosim_inputs[k]))))
                    #    return
                logging.warning("pass checking op[{0}] {1} input".format(i, predict_def.op[i].type))
                logging.warning("begin to check op[{0}] {1} output".format(i, predict_def.op[i].type))
                for j, _ in enumerate(inf_results):
                    if predict_def.op[i].output[j][0] == '_':
                        continue
                    if args.cosim:
                        if not cc2.assert_allclose(inf_results[j], cosim_results[j], **tol):
                            logging.error("failed checking op[{0}] {1} output".format(i, predict_def.op[i].type))
                            exit()
                    if args.int8_cosim:
                        cc2.assert_allclose(inf_results[j], cosim_results[j], **tol)
                        cc2.assert_compare(inf_results[j], cosim_results[j], 1e-01, 'ALL')
                    #if not np.allclose(inf_results[j], cosim_results[j], **tol):
                       # logging.error("Failure in cosim {} op {} output {}"
                       #     .format(
                       #     i,
                       #     predict_def.op[i].type,
                       #     predict_def.op[i].output[j]))
                       # logging.error(inf_results[j].flatten())
                       # logging.error(cosim_results[j].flatten())
                       # logging.error("Max error: {}"
                       #     .format(
                       #     np.max(np.abs(
                       #         inf_results[j] - cosim_results[j]))))
                       # return
                logging.warning("pass checking op[{0}] {1} output".format(i, predict_def.op[i].type))
        else:
            if 'style-transfer' in args.model or (args.model == 'faster-rcnn' and args.device.lower() == 'gpu'):
                ws.FeedBlob(str(predict_def.op[0].input[0]), imgs)
            else:
                if args.int8_model and (args.quantize_input or args.quantize_input_once):
                    args.quantize_input_once = False
                    ws.FeedBlob("X", imgs, device_opts)
                    sw2nhwc = core.CreateOperator(
                        "NCHW2NHWC",
                        ["X"],
                        ["X_nhwc"],
                        device_option=device_opts
                        )

                    quantize_X = core.CreateOperator(
                        "Int8Quantize",
                        ["X_nhwc"],
                        [str(predict_def.op[0].input[0])],
                        engine="DNNLOWP",
                        device_option=device_opts,
                        Y_zero_point=128,
                        Y_scale=1.18944883347,
                        )
                    ws.RunOperatorOnce(sw2nhwc)
                    ws.RunOperatorOnce(quantize_X)
                elif not args.int8_model:
                    ws.FeedBlob(str(predict_def.op[0].input[0]), imgs, device_opts)
            if predict_def.op[-1].type == 'Accuracy':
                if args.device.lower() == 'gpu':
                    ws.FeedBlob(str(predict_def.op[-1].input[1]), init_label, device_opts)
                    if predict_def.op[-2].type == 'Accuracy':
                        ws.FeedBlob(str(predict_def.op[-2].input[1]), init_label, device_opts)
                    elif predict_def.op[-3].type == 'Accuracy':
                        ws.FeedBlob(str(predict_def.op[-3].input[1]), init_label, device_opts)
                else:
                    ws.FeedBlob(str(predict_def.op[-1].input[1]), init_label, device_opts_cpu)
                    if predict_def.op[-2].type == 'Accuracy':
                        ws.FeedBlob(str(predict_def.op[-2].input[1]), init_label, device_opts_cpu)
                    elif predict_def.op[-3].type == 'Accuracy':
                        ws.FeedBlob(str(predict_def.op[-3].input[1]), init_label, device_opts_cpu)

            comp_start_time = timeit.default_timer()
            #if args.profile or predict_def.op[-1].type == 'Accuracy':
            #    ws.RunNet(net)
            #else:
            ws.RunNet(net)

            comp_elapsed_time = timeit.default_timer() - comp_start_time
            comp_time += comp_elapsed_time
            output = ws.FetchBlob(str(predict_def.op[-1].output[0]))
            if predict_def.op[-2].type == 'Accuracy':
                output2 = ws.FetchBlob(str(predict_def.op[-2].output[0]))
            elif predict_def.op[-3].type == 'Accuracy':
                output2 = ws.FetchBlob(str(predict_def.op[-3].output[0]))
            elif  predict_def.op[-1].type == 'BoxWithNMSLimit':
                output2 = ws.FetchBlob(str(predict_def.op[-1].output[1]))
                output3 = ws.FetchBlob(str(predict_def.op[-1].output[2]))
            logging.warning("[{0:.2%}] Output shape: {1}, computing in {2:.10f}"
                            " seconds, processing {3} images in {4:.10f} seconds."
                            .format(((k + 1) / len(images)), output.shape,
                                    comp_elapsed_time, len(raw), img_elapsed_time))
            if predict_def.op[-1].type == 'BoxWithNMSLimit':
                outputs.append([output, output2, output3])
            elif predict_def.op[-1].type != 'Accuracy':
                outputs.append(output)
                #logging.info(output)
            else:
                accuracy_top1.append(output2)
                accuracy_top5.append(output)
            if args.profile:
                logging.warning("observer time = {}".format(ob.average_time()))
                logging.warning("observer time = {}".format(ob.average_time_children()))

        del imgs
        if k >= (iterations - 1):
            logging.warning("Exit after running {} iterations"
                            .format(iterations))
            break
    if args.profile:
        net.RemoveObserver(ob)

    if args.cosim:
        ws.SwitchWorkspace(def_ws_name)
        logging.warning("Cosim passed Ran 1 test OK")
        return
    if comp_time <= 0:
        logging.error("The total time is invalid!")
        return
    info_str = ""
    if len(accuracy_top1) > 0:
        mean_accuracy_top1 = 0
        mean_accuracy_top5 = 0
        for i, _ in enumerate(accuracy_top1):
            mean_accuracy_top1 += accuracy_top1[i] * batch_size
            mean_accuracy_top5 += accuracy_top5[i] * batch_size
        mean_accuracy_top1 /= batch_size * len(accuracy_top1)
        mean_accuracy_top5 /= batch_size * len(accuracy_top5)
        info_str += "\nAccuracy: {:.5%}".format(mean_accuracy_top1)
        info_str += "\nTop5Accuracy: {:.5%}".format(mean_accuracy_top5)
        total_image = processed_images
        logging.critical("\nImages per second: {0:.10f}\nTotal computing time:"
                         " {1:.10f} seconds\nTotal image processing time: {2:.10f} seconds\n"
                         "Total model loading time: {3:.10f} seconds\nTotal images: {4}{5}"
                         .format(total_image / comp_time, comp_time, img_time,
                                 model_elapsed_time, total_image, info_str))
        return
    if args.annotations:
        logging.info(" the total length of outputs is {}".format(len(outputs)))
        logging.critical("result is ={}".format(cc2.prepare_and_compute_map_data(outputs, fnames, apath)))
    info_str = ""
    accuracy = None
    top5accuracy = None
    summary = None
    total_image = processed_images
    if model_info["output_type"] == "segmentation" or args.dummydata:
        total_image = processed_images
    elif model_info["output_type"] == "possibility":
        label_offset = 0
        if model_info["model_type"] == "mlperf legacy mb":
            label_offset = -1
        results, total_image = cc2.ParsePossOutputs(outputs, label_offset)
        summary = cc2.ParsePossResults(results, labels, validation, fnames)
        if not summary:
            logging.error("Failed to parse the results!")
            return
        elif total_image <= 0 or len(summary) != total_image:
            logging.error("No available results!")
            return
        if validation:
            accuracy = 0
            top5accuracy = 0
            for res in summary:
                if res[1] == "Pass":
                    accuracy += 1
                    top5accuracy += 1
                elif res[1] == "Top5Pass":
                    top5accuracy += 1
            accuracy = accuracy / total_image
            top5accuracy = top5accuracy / total_image
            info_str += "\nAccuracy: {:.5%}".format(accuracy)
            info_str += "\nTop5Accuracy: {:.5%}".format(top5accuracy)
    elif model_info["output_type"] == "argmax":
        results, total_image = cc2.ParsePossOutputsArgMax(outputs, -1)
        summary = cc2.ParsePossResults(results, labels, validation, fnames)
        if not summary:
            logging.error("Failed to parse the results!")
            return
        elif total_image <= 0 or len(summary) != total_image:
            logging.error("No available results!")
            return
        if validation:
            accuracy = 0
            for res in summary:
                if res[1] == "Pass":
                    accuracy += 1
            accuracy = accuracy / total_image
            info_str += "\nAccuracy: {:.5%}".format(accuracy)
    elif model_info["output_type"] == "post image":
        results, total_image = cc2.ParsePostOutputs(outputs)
        if args.post_images_path:
            cc2.SavePostImages(results, args.post_images_path, fnames)
    logging.critical("\nImages per second: {0:.10f}\nTotal computing time:"
                     " {1:.10f} seconds\nTotal image processing time: {2:.10f} seconds\n"
                     "Total model loading time: {3:.10f} seconds\nTotal images: {4}{5}"
                     .format(total_image / comp_time, comp_time, img_time,
                             model_elapsed_time, total_image, info_str))
    cc2.SaveOutput(args, summary, accuracy, top5accuracy, comp_time,
                   total_image, img_time, model_elapsed_time)


if __name__ == '__main__':
    logging.critical("Do not run this script independently!")
    exit()
