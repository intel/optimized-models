#! /usr/bin/python
## @package caffe2_tools
# Module caffe2.tools.run_caffe2
"""
the main entry to run caffe2 model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import timeit
import logging
import argparse

LOG_FORMAT = "%(levelname)s:%(message)s"

def ArgError(error):
    """
    print help if arg is error
    """
    logging.error("Please set {}. "
                  "OR, refer to the help of this script (-h)"
                  .format(error))

def Calibration(args, extr_args):
    """
    function to do calibration.
    """
    if not args.model:
        ArgError("model to run (-m)")
        return
    if args.print_net_def:
        import inference as inf
        inf.PrintNetDef(args.model, args.print_net_def)
        return
    if not args.device:
        ArgError("device (-d)")
        return
    if (
            not args.dummydata and
            not args.images_path and
            not "CAFFE2_INF_IMG_PATH" in os.environ
    ):
        ArgError("the path of input images (-p)")
        return
    import inference as inf
    inf.Calibration(args, extra_args)


def Inference(args, extra_args):
    """
    function to do inference.
    """
    if not args.model:
        ArgError("model to run (-m)")
        return
    if args.print_net_def:
        import inference as inf
        inf.PrintNetDef(args.model, args.print_net_def)
        return
    if not args.device:
        ArgError("device (-d)")
        return
    if (
            not args.dummydata and
            not args.images_path and
            not "CAFFE2_INF_IMG_PATH" in os.environ
    ):
        ArgError("the path of input images (-p)")
        return
    import inference as inf
    inf.Run(args, extra_args)




def GetArgumentParser():
    """
    to parse the argument
    """
    parser = argparse.ArgumentParser(description="The scripts to run Caffe2.\n"
                                                 "for example, to run alexnet inference:\n"
                                                 "./run_caffe2.py -m alexnet\n"
                                                 " -p /path/to/imput/image\n"
                                                 " -v /path/to/image/validate/index/file\n"
                                     )
    parser.add_argument(
        "-a", "--optimization",
        type=str,
        help="Enable optimizations for running mode, split by comma.\n"
             "(Set 'all' to enable all optimizations for current running mode)\n"
             "-For inference, available optimizations:\n"
             "bn_folding,bn_inplace,fusion_conv_relu,fusion_conv_sum,remove_dropout,"
             "int8_mode.\n"
             "-For training, available optimizations:\n"
             " "
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=1,
        help="The batch size. (DEFAULT: %(default)i)"
    )
    parser.add_argument(
        "-c", "--crop_size",
        type=int,
        default=None,
        help="The crop size of input image. (DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="ideep",
        help="Choose device to run. cpu, gpu or ideep."
             "(DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-e", "--log_level",
        type=str,
        default="warning",
        help="The log level to show off. debug, info, warning, error, critical."
             "(DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-f", "--forward_only",
        action='store_true',
        help="If set, only run the forward path."
             "(DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-g", "--log",
        type=str,
        help="The log file path."
    )
    parser.add_argument(
        "-i", "--iterations",
        type=int,
        help="Number of iterations to run the network."
    )
    parser.add_argument(
        "-j", "--post_images_path",
        type=str,
        default=None,
        help="The path to store post images."
    )
    parser.add_argument(
        "-l", "--label_file",
        type=str,
        help="The input label index file."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="The model to run."
    )
    parser.add_argument(
        "-n", "--net_type",
        type=str,
        default="simple",
        help="The net type for Caffe2.(DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default=None,
        help="The output file to save the results of validating or label check."
    )
    parser.add_argument(
        "-calib", "--calib_algo",
        type=str,
        help="The algorithm of calibration. absmax, moving_average, or l_divergence"
    )
    parser.add_argument(
        "-int8", "--int8_model",
        action='store_true',
        help="Use the int8 model, instead of fp32 model."
    )
    parser.add_argument(
        "-onnx", "--onnx_model",
        action='store_true',
        help="Use the onnx model, instead of caffe2 model."
    )
    parser.add_argument(
        "-p", "--images_path",
        type=str,
        help="The path of input images."
    )
    parser.add_argument(
        "-tp", "--tr_images_path",
        type=str,
        help="The path of input images for training."
    )
    parser.add_argument(
        "-q", "--annotations",
        type=str,
        help="The path of Annotations file for VOC"
    )
    parser.add_argument(
        "-r", "--mode",
        type=str,
        default="inference",
        help="Choose running mode. inference, calibration or training."
             "(DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-s", "--show_supported_models",
        action='store_true',
        help="Show off all supported model for inference."
    )
    parser.add_argument(
        "-t", "--profile",
        action='store_true',
        help="Trigger profile on current topology."
    )
    parser.add_argument(
        "-u", "--dummydata",
        action='store_true',
        help="Trigger profile on current topology."
    )
    parser.add_argument(
        "-v", "--validation_file",
        type=str,
        help="The input validation index file."
    )
    parser.add_argument(
        "-w", "--warmup_iterations",
        type=int,
        default=0,
        help="Number of warm-up iterations before benchmarking."
             "(DEFAULT: %(default)i)"
    )
    parser.add_argument(
        "-x", "--print_net_def",
        type=str,
        default=None,
        help="If set, only print out the net definition for the model.\n"
             "predict_net for topology, init_net for weight data."
             "(DEFAULT: %(default)s)"
    )
    parser.add_argument(
        "-y", "--cosim",
        action='store_true',
        help="Trigger cosim on current topology."
    )
    parser.add_argument(
        "-yi", "--int8_cosim",
        action='store_true',
        help="Trigger int8 cosim on current topology."
    )
    parser.add_argument(
        "-z", "--noptimize",
        action='store_true',
        help="not Trigger optimization on current topology."
    )
    return parser


if __name__ == '__main__':
    args, extra_args = GetArgumentParser().parse_known_args()
    LOG_LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    if args.log_level.lower() in LOG_LEVEL_MAP:
        log_level = LOG_LEVEL_MAP[args.log_level.lower()]
    else:
        log_level = None
        logging.warning("Wrong log level {}. Ignored!".format(args.log_level))
    logging.basicConfig(
        format=LOG_FORMAT,
        filename=args.log,
        filemode="w",
        level=log_level)

    if args.show_supported_models:
        import inference.models as m

        m.ShowModels()
    elif len(sys.argv) == 1:
        GetArgumentParser().print_help()
    else:
        type_map = {
            "inference": Inference,
            "calibration": Calibration,
        }
        if args.mode.lower() in type_map:
            start_time = timeit.default_timer()
            type_map[args.mode.lower()](args, extra_args)
            elapsed_time = timeit.default_timer() - start_time
            logging.warning("Total time in {} mode: {:.10f} seconds"
                            .format(args.mode, elapsed_time))
        else:
            logging.error("Wrong running mode {}. Exit!".format(args.mode))
