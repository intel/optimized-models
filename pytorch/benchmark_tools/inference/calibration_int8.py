"""
module to run calibration
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import logging
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python import transformations as tf
import inference.models as m
from common import common_caffe2 as cc2

def Calibration(args, extra_args):
    """
    function to run calibration
    """

    if not m.IsSupported(args.model):
        logging.error("Not supported model: {}".format(args.model))
        m.ShowModels()
        return

    images_path = None
    if args.images_path:
        images_path = os.path.abspath(args.images_path)
    elif "CAFFE2_INF_IMG_PATH" in os.environ:
        images_path = os.path.abspath(os.environ["CAFFE2_INF_IMG_PATH"])

    batch_size = 1
    if args.batch_size:
        batch_size = int(args.batch_size)
        if batch_size <= 0:
            logging.error("Invalid batch size {}. Exit!".format(batch_size))
            return

    iterations = args.iterations if args.iterations else sys.maxsize
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
    if str(model_info["image_mean"]) != 'None':
        mean_tmp = ((model_info["image_mean"]).split('/')[-1]).split(' ')
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
    if args.onnx_model:
        init_def, predict_def = cc2.OnnxToCaffe2(model_info["onnx_model"])
    else:
        with open(model_info["init_net"]) as i:
            if model_info["model_type"] == "prototext":
                import google.protobuf.text_format as ptxt
                init_def = ptxt.Parse(i.read(), caffe2_pb2.NetDef())
            else:
                init_def = caffe2_pb2.NetDef()
                init_def.ParseFromString(i.read())
        with open(model_info["predict_net"]) as p:
            if model_info["model_type"] == "prototext":
                import google.protobuf.text_format as ptxt
                predict_def = ptxt.Parse(p.read(), caffe2_pb2.NetDef())
            else:
                predict_def = caffe2_pb2.NetDef()
                predict_def.ParseFromString(p.read())

    if model_info["model_type"] == "caffe legacy":
        cc2.MergeScaleBiasInBN(predict_def)
        cc2.RemoveUselessExternalInput(predict_def)

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
    if args.device.lower() in dev_map:
        device_opts.device_type = dev_map[args.device.lower()]
    else:
        logging.error("Wrong device {}. Exit!".format(args.device))
        return

    logging.warning("Start running calibration")
    images, _ = cc2.ImageProc.BatchImages(images_path, batch_size, iterations)

    # for kl_divergence calibration, we use the first 100 images to get
    # the min and max values, and the remaing images are applied to compute the hist.
    # if the len(images) <= 100, we extend the images with themselves.
    def data_gen():
        for raw in images:
            imgs, _ = cc2.ImageProc.PreprocessImages(
                raw, crop_size, rescale_size, mean, scale, 1, need_normalize, color_format)
            #imgs, _ = cc2.ImageProc.PreprocessImagesByThreading(
            #        raw, crop_size,rescale_size, mean, scale, 1)
            yield imgs
            del imgs

    cc2.UpdateDeviceOption(device_opts, init_def)
    workspace.RunNetOnce(init_def)

    cc2.UpdateDeviceOption(device_opts, predict_def)
    net = core.Net(model_info["model_name"])
    net.Proto().CopyFrom(predict_def)
    if args.device.lower() == 'ideep' and not args.noptimize:
        logging.warning('Optimizing module {} ....................'
                        .format(model_info["model_name"]))
        tf.optimizeForIDEEP(net)
    predict_def = net.Proto()
    if predict_def.op[-1].type == 'Accuracy':
        init_label = np.ones((batch_size), dtype=np.int32)
        label = net.AddExternalInput('label')
        workspace.FeedBlob(label, init_label, device_opts)
        for i, op in enumerate(predict_def.op):
            if op.type == 'Accuracy':
                workspace.FeedBlob(str(predict_def.op[i].output[0]), init_label, device_opts)

    from inference.calibrator import Calibrator, KLCalib, AbsmaxCalib, EMACalib
    algorithm = AbsmaxCalib()
    kind = os.environ.get('INT8CALIB')
    if args.calib_algo:
        kind = args.calib_algo
    if kind == "absmax":
        algorithm = AbsmaxCalib()
    elif kind == "moving_average":
        ema_alpha = 0.5
        algorithm = EMACalib(ema_alpha)
    elif kind == "kl_divergence":
        kl_iter_num_for_range = 100
        while len(images) < 2*kl_iter_num_for_range:
            images += images
        algorithm = KLCalib(kl_iter_num_for_range)

    i = 0
    length = len(images)
    calib = Calibrator(algorithm, device_opts)
    for data in data_gen():
        i += 1
        workspace.FeedBlob(predict_def.op[0].input[0], data, device_opts)
        logging.warning("in progress {}/{}(batch/batch total)".format(i, length))
        calib.RunCalibIter(workspace, predict_def)

    predict_quantized, init_quantized = calib.DepositQuantizedModule(workspace, predict_def)

    cc2.SaveModel(args.output_file + '/init_net_int8.pb', init_quantized,
                  args.output_file + '/predict_net_int8.pb', predict_quantized)
    cc2.SaveModelPtxt(args.output_file + '/predict_net_int8.pbtxt', predict_quantized)
    cc2.SaveModelPtxt(args.output_file + '/init_net_int8.pbtxt', init_quantized)


if __name__ == '__main__':
    logging.critical("Do not run this script independently!")
    exit()
