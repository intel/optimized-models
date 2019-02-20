"""inference script to support accuracy and performance benchmark"""
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
import logging
import ctypes
import time
import os
import pickle
import mxnet as mx

from mxnet import nd
from mxnet.base import check_call, _LIB



def load_model(_symbol_file, _param_file, _logger=None):
    """load existing symbol model"""
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, _symbol_file)
    if _logger is not None:
        _logger.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, _param_file)
    if _logger is not None:
        _logger.info('Loading params from file %s' % param_file_path)
    save_dict = nd.load(param_file_path)
    _arg_params = {}
    _aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            _arg_params[name] = v
        if tp == 'aux':
            _aux_params[name] = v
    return symbol, _arg_params, _aux_params

def advance_data_iter(data_iter, n):
    """use to warm up data for performance benchmark"""
    assert n >= 0
    if n == 0:
        return data_iter
    has_next_batch = True
    while has_next_batch:
        try:
            data_iter.next()
            n -= 1
            if n == 0:
                return data_iter
        except StopIteration:
            has_next_batch = False

CRITEO = {
    'train': 'train.csv',
    'test': 'eval.csv',
    'num_linear_features': 26000,
    'num_embed_features': 26,
    'num_cont_features': 13,
    'embed_input_dims': 1000,
    'hidden_units': [32, 1024, 512, 256],
}
def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')

    parser.add_argument('--symbol-file', type=str, default='checkpoint-symbol.json', help='symbol file path')
    parser.add_argument('--param-file', type=str, default='checkpoint-0000.params', help='param file path')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--accuracy', type=bool, default=False)
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--num-omp-threads', type=int, default=28)
    parser.add_argument('--num-batches', type=int, default=8000000)
    args = parser.parse_args()

    ctx = mx.cpu()

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    if args.accuracy is True:
        logger.info('Accuracy Mode')
    else:
        logger.info('Performance Mode')

    symbol_file = args.symbol_file
    param_file = args.param_file


    batch_size = args.batch_size
    logger.info('batch size = %d for inference', batch_size)
    label_name = args.label_name
    logger.info('label_name = %s', label_name)

    if args.accuracy is False:
        val_csr = load_object('train_csr.pkl')
        val_dns = load_object('train_dns.pkl')
        val_label = load_object('train_label.pkl')
    else:
        val_csr = load_object('val_csr.pkl')
        val_dns = load_object('val_dns.pkl')
        val_label = load_object('val_label.pkl')

    # creating data iterator
    data = mx.io.NDArrayIter({'csr_data': val_csr, 'dns_data': val_dns},
                             {'softmax_label': val_label}, batch_size,
                             shuffle=False, last_batch_handle='discard')

    # loading model
    sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)


    # make sure that fp32 inference works on the same images as calibrated quantized model

    logger.info('Running model %s for inference', symbol_file)

    acc_m = mx.metric.create('acc')
    mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['csr_data', 'dns_data'], label_names=[label_name, ])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)

    check_call(_LIB.MXSetNumOMPThreads(ctypes.c_int(args.num_omp_threads)))
    batch_data = []
    nbatch = 0
    for batch in data:
        if nbatch < args.num_batches:
            batch_data.append(batch)
            nbatch += 1
        else:
            break
    #for data warmup
    wi = 50
    i = 0
    for batch in batch_data:
        if i < wi:
            mod.forward(batch, is_train=False)
            i += 1
        else:
            break
    data.hard_reset()
    mx.nd.waitall()
    #real run
    if "DO_WIDE_DEEP_PROFILING" in os.environ:
        print("wide_deep profiling start !!!!!!!!!!!!!")
        mx.profiler.set_config(profile_symbolic=True, profile_imperative=True, profile_memory=False, profile_api=False)
        mx.profiler.set_state('run')
    nbatch = 0
    tic = time.time()
    for batch in batch_data:
        nbatch += 1
        mod.forward(batch, is_train=False)
        if args.accuracy is True:
            for output in mod.get_outputs():
                output.wait_to_read()
            mod.update_metric(acc_m, batch.label)
        else:
            mx.nd.waitall()
    speed = nbatch * batch_size / (time.time() - tic)
    logger.info("Run [%d] Batchs \tSpeed: %.2f samples/sec", nbatch, speed)
    if args.accuracy is True:
        logger.info(acc_m.get())
    if "DO_WIDE_DEEP_PROFILING" in os.environ:
        print("wide_deep profiling end !")
        mx.profiler.set_state('stop')
        profiler_info = mx.profiler.dumps()
        print(profiler_info)
