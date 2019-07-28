import mxnet as mx
import time
import logging
import argparse

rnncell_type = ['rnn', 'lstm', 'gru', 'sru']


parser = argparse.ArgumentParser(description='MxNet RNN benchmark')
parser.add_argument('--gpu', '-p', type=bool, default=False, help="whether use GPU, default is False")
parser.add_argument('--cell_type', '-cell', type=str, default='lstm', 
                    help="cell type, can be \"LSTM, GRU, RNN, SRU\", default is LSTM.")
parser.add_argument('--layer_num', '-l', type=int, default=1, help="layer num, default is 1.")


warm_up = 20
iter_num = 200


def fused_module(input_shape, cell_type, layer_nums=1, ctx=mx.cpu(), layout="TNC"):
    
    assert cell_type in rnncell_type
    
    bs = input_shape[0]
    seq_len = input_shape[1]
    embed_dim = input_shape[2]
    hidden_size = input_shape[3]
    if layout == 'NTC':
        dshape = (bs, seq_len, embed_dim)
    elif layout == 'TNC':
        logging.warning('layout TNC is used!')
        dshape = (seq_len, bs, embed_dim)
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')

    if cell_type == 'lstm':
        lstm_cell = mx.rnn.FusedRNNCell(
            hidden_size, num_layers=layer_nums, mode='lstm', get_next_state=False, prefix='l0_')
        rnn_sym, _ = lstm_cell.unroll(
            seq_len, data, layout=layout, merge_outputs=True)
    elif cell_type == 'gru':
        gru_cell = mx.rnn.FusedRNNCell(hidden_size, num_layers=layer_nums, mode='gru', prefix='l0_')
        rnn_sym, _ = gru_cell.unroll(
            seq_len, data, layout=layout, merge_outputs=True)

    mod = mx.mod.Module(rnn_sym, label_names=None, context=ctx)
    mod.bind(data_shapes=[('data', dshape)], label_shapes=None)

    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_optimizer(optimizer='sgd')
    return mod


def rnncell_score_fused(mod):
    batch = mx.io.DataBatch(data=[mx.random.uniform(shape=mod.data_shapes[0][1])], label=[])
    tic = time.time()

    mod.forward(batch, is_train=False)
    output = mod.get_outputs()[0]
    output.wait_to_read()

    fwd = time.time() - tic
    return fwd


if __name__ == '__main__':

    '''
	cell: unidirection-lstm
	hidden_size: 512/1024
	BS: 1/32
	sentence length/time step: 50/
	layers: 1/4

    '''
    # [bs, sequence length, embedding size, hidden size]
    input_shape_list = [[1, 50, 512, 512], [1, 50, 1024, 1024], [32, 50, 512, 512], [32, 50, 1024, 1024]]
    
    logging.basicConfig(level = logging.INFO)
    args = parser.parse_args()
    if args.gpu:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()

    cell = args.cell_type
    layer_nums = args.layer_num
	
    logging.warning('Fused RNN API Inference benchmarking started')
    
 
    for input_shape in input_shape_list:
        total_fwd = 0
        mod = fused_module(input_shape, cell, layer_nums, ctx)
        # mod.save_checkpoint('gnmt', 0)
        for i in range(warm_up + iter_num):
            fwd = rnncell_score_fused(mod)
            if i >= warm_up:
                total_fwd += fwd

        total_fwd = total_fwd / iter_num
        logging.info(str(input_shape) + ' time cost ' + str(total_fwd) + 's samples/sec = ' + str(input_shape[0]/total_fwd))

