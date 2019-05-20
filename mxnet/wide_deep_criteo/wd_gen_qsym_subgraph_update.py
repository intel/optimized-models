"""Generate quantized graph based on original fp32 graph"""
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import os
import logging
#import ctypes
import pickle
from mxnet import nd
import mxnet as mx
from mxnet.contrib.quantization import quantize_model



def load_model(symbol_file, param_file, mlogger=None):
    """load existing symbol model"""
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file)
    if mlogger is not None:
        mlogger.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, param_file)
    if mlogger is not None:
        mlogger.info('Loading params from file %s' % param_file_path)
    save_dict = nd.load(param_file_path)
    marg_params = {}
    maux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            marg_params[name] = v
        if tp == 'aux':
            maux_params[name] = v
    return symbol, marg_params, maux_params


def save_symbol(fname, symbol, slogger=None):
    if slogger is not None:
        slogger.info('Saving symbol into file at %s' % fname)
    symbol.save(fname)


def save_params(fname, parg_params, paux_params, plogger=None):
    if plogger is not None:
        plogger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in parg_params.items()}
    save_dict.update({('aux:%s' % k): v.as_in_context(mx.cpu()) for k, v in paux_params.items()})
    mx.nd.save(fname, save_dict)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized model from a FP32 model')
    parser.add_argument('--ctx', type=str, default='cpu')

    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--calib-dataset', type=str, default='data/adult.data',
                        help='path of the calibration dataset')
    parser.add_argument('--num-calib-batches', type=int, default=162,
                        help='number of batches for calibration')
    parser.add_argument('--exclude-first-conv', action='store_true', default=True,
                        help='excluding quantizing the first conv layer since the'
                             ' number of channels is usually not a multiple of 4 in that layer'
                             ' which does not satisfy the requirement of cuDNN')
    parser.add_argument('--calib-mode', type=str, default='naive',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                             ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                             ' in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    parser.add_argument('--quantized-dtype', type=str, default='uint8',
                        choices=['int8', 'uint8'],
                        help='quantization destination data type for input data')
    args = parser.parse_args()

    if args.ctx == 'gpu':
        ctx = mx.gpu(0)
    elif args.ctx == 'cpu':
        ctx = mx.cpu(0)
    else:
        raise ValueError('ctx %s is not supported in this script' % args.ctx)

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    # get batch size
    batch_size = args.batch_size
    logger.info('batch size = %d for calibration', batch_size)
    # get number of batches for calibration
    num_calib_batches = args.num_calib_batches

    calib_mode = args.calib_mode
    if calib_mode != 'none':
        logger.info('number of batches = %d for calibration', num_calib_batches)

    val_csr = load_object('val_csr.pkl')
    val_dns = load_object('val_dns.pkl')
    val_label = load_object('val_label.pkl')

    # creating data iterator
    data = mx.io.NDArrayIter({'csr_data': val_csr, 'dns_data': val_dns},
                             {'softmax_label': val_label}, batch_size,
                             shuffle=True, last_batch_handle='discard')
    # loading model
    sym, arg_params, aux_params = load_model('checkpoint-symbol.json', 'checkpoint-0000.params', logger)

    calib_layer = lambda name: (name.find('fullyconnected') != -1 or \
                                name.find('FullyConnected') != -1 or \
                                name.find('fully_connected') != -1 or \
                                name.find('concat0_output') != -1)
    sym = sym.get_backend_symbol('MKLDNN')
    excluded_sym_names = ['concat0', '_plus0']
    cqsym, qarg_params, aux_params = quantize_model(sym=sym, arg_params=arg_params, aux_params=aux_params,
                                                    data_names=['csr_data', 'dns_data'],
                                                    label_names=['softmax_label', ],
                                                    ctx=ctx, excluded_sym_names=excluded_sym_names,
                                                    calib_mode=calib_mode, calib_data=data,
                                                    num_calib_examples=num_calib_batches*batch_size,
                                                    calib_layer=calib_layer, quantized_dtype=args.quantized_dtype,
                                                    logger=logger)
    if calib_mode == 'entropy':
        suffix = '-quantized-%dbatches-entropy' % num_calib_batches
    elif calib_mode == 'naive':
        suffix = '-quantized-%dbatches-naive' % num_calib_batches
    else:
        raise ValueError('unknow calibration mode %s received, only supports `none`, `naive`, and `entropy`'
                         % calib_mode)
    prefix = 'WD'
    sym_name = '%s-symbol.json' % (prefix + suffix)
    cqsym = cqsym.get_backend_symbol('MKLDNN_QUANTIZE')
    save_symbol(sym_name, cqsym, logger)
    param_name = '%s-%04d.params' % (prefix + '-quantized', 0)
    save_params(param_name, qarg_params, aux_params, logger)
