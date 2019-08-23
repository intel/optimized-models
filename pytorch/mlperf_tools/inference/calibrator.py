"""calibration tool"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import copy
import math
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import core, utils

hist = {}
hist_edges = {}
iteration_idx = 0


class algorithm(object):
    """algorithm class"""
    def get_predecessor_op(self, inp_index, op_pos, predict_def):
        """ get predecessor op from the net"""
        if op_pos < 1 or op_pos >= len(predict_def.op):
            return None, None
        pos = op_pos - 1
        input_name = predict_def.op[op_pos].input[inp_index]
        while pos >= 0:
            op = predict_def.op[pos]
            for outp in op.output:
                if outp == input_name:
                    return op, pos
            pos -= 1
        return None, None

    def get_successor_ops(self, op_pos, predict_def):
        """ get successor op from the net"""
        if op_pos < -1 or op_pos >= (len(predict_def.op) - 1):
            return []
        successors = []
        pos = op_pos + 1
        if op_pos == -1:
            output = predict_def.op[0].input[0]
        else:
            output = predict_def.op[op_pos].output[0]
        while pos < len(predict_def.op):
            op = predict_def.op[pos]
            for inp in op.input:
                if inp == output:
                    successors.append(op)
                    break
            for outp in op.output:
                if outp == output:
                    return successors
            pos += 1
        return successors

    def get_input_index(self, inp_name, op):
        for i, inp in enumerate(op.input):
            if inp == inp_name:
                return i
        return None

    def get_op_by_output(self, outp_name, net_def):
        for op in net_def.op:
            if outp_name == op.output[0]:
                return op
        return None

    def insert_op(self, index, new_op, predict_def):
        src_op = new_op
        cur_index = index
        while cur_index < len(predict_def.op):
            cur_op = predict_def.op[cur_index]
            buf_op = copy.deepcopy(cur_op)
            cur_op.CopyFrom(src_op)
            src_op = buf_op
            cur_index += 1
        predict_def.op.extend([src_op])

    def arg_index(self, op, name):
        for i in range(len(op.arg)):
            if op.arg[i].name == name:
                return i
        return None

    def get_arg(self, op, name):
        for i in range(len(op.arg)):
            if op.arg[i].name == name:
                return op.arg[i]
        return None

    def remove_arg(self, op, name):
        for i in range(len(op.arg)):
            if op.arg[i].name == name:
                del op.arg[i]
                return True
        return False

    def remove_max(self, predict_def):
        for op in predict_def.op:
            for i in range(len(op.input)):
                self.remove_arg(op, "absmax_input"+ "_" + str(i))
            for j in range(len(op.output)):
                self.remove_arg(op, "absmax_output"+ "_" + str(j))

    def has_weights(self, op):
        return {
            "FC"            : True,
            "Conv"          : True,
            "ConvFusion"    : True,
        }.get(op.type, False)

    def is_weights(self, op, inp_index):
        return self.has_weights(op) and inp_index == 1

    def has_bias(self, op):
        inp_len = len(op.input)
        if op.type == "ConvFusion" and inp_len == 3:
            arg = self.get_arg(op, "fusion_type")
            assert arg is not None
            # Check if is ConvRelu fusion
            return arg.i == 1
        return {
            "FC"            : inp_len == 3,
            "Conv"          : inp_len == 3,
            "ConvFusion"    : inp_len == 4,
        }.get(op.type, False)

    def is_bias(self, op, op_index):
        return self.has_bias(op) and op_index == 3

    def get_max(self, op, tensor, tensor_idx, tensor_name, max_name):
        raise Exception("Please add max value computation method!")

    def gather_max(self, predict_def):
        pass

    def update_status(self):
        pass

    def get_max_min(self, op, tensor, tensor_idx, tensor_name, name):
        """get max and min"""
        #name = name + "_" + str(tensor_idx)
        arg = self.get_arg(op, name)
        max_min = np.array([np.max(tensor), min(np.min(tensor), 0)]).astype(np.float32)
        if arg is not None:
            orig_max = arg.floats[0]
            orig_min = arg.floats[1]
            cur_max = max(orig_max, max_min[0])
            cur_min = min(orig_min, max_min[1])
            max_min = np.array([cur_max, cur_min]).astype(np.float32)
            self.remove_arg(op, name)
        # save max and min vaules in predict_def as operator arguments
        max_arg = utils.MakeArgument(name, max_min)
        op.arg.extend([max_arg])


class KLCalib(algorithm):
    """ KL calibrator """
    def __init__(self, kl_iter_num_for_range=100):
        self.kl_iter_num_for_range = kl_iter_num_for_range

    def update_status(self):
        global iteration_idx
        iteration_idx += 1

    def get_max(self, op, tensor, tensor_idx, tensor_name, max_name):
        global iteration_idx
        name = max_name + "_" + str(tensor_idx)
        op_hist_name = tensor_name + "_" + max_name + "_" + str(tensor_idx)

        arg = self.get_arg(op, name)
        if iteration_idx < self.kl_iter_num_for_range:
            max_min = np.array([np.max(tensor), np.min(tensor)]).astype(np.float32)
            if arg is not None:
                orig_max = arg.floats[0]
                orig_min = arg.floats[1]
                cur_max = max(orig_max, max_min[0])
                cur_min = min(orig_min, max_min[1])
                max_min = np.array([cur_max, cur_min]).astype(np.float32)
                self.remove_arg(op, name)
            # save max vaules in predict_def as operator arguments
            max_arg = utils.MakeArgument(name, max_min)
            op.arg.extend([max_arg])
        else:
            assert arg is not None
            max_val = arg.floats[0]
            min_val = arg.floats[1]
            self.get_kl_hist(tensor, min_val, max_val, op_hist_name)

    def update_max(self, op, max_name, tensor_idx, tensor_name):
        """update the max data of the collected data"""
        global hist
        global hist_edges
        global iteration_idx

        name = max_name + "_" + str(tensor_idx)
        hist_name = tensor_name + "_" + max_name + "_" + str(tensor_idx)

        P_sum = iteration_idx - self.kl_iter_num_for_range
        arg = self.get_arg(op, name)
        assert arg is not None
        max_val = arg.floats[0]
        min_val = arg.floats[1]

        hist_iter = hist[hist_name]
        hist_edges_iter = hist_edges[hist_name]
        layer_max = self.get_optimal_scaling_factor(hist_iter, hist_edges_iter,
                                                    P_sum, max_val, min_val)

        self.remove_arg(op, name)
        max_arg = utils.MakeArgument(name, np.array([layer_max]).astype(np.float32))
        # save max vaules in predict_def as operator arguments
        op.arg.extend([max_arg])

    def gather_max(self, predict_def):
        for op in predict_def.op[0:]:
            for j, input_name in enumerate(op.input):
                if self.is_weights(op, j) or self.is_bias(op, j):
                    continue
                self.update_max(op, "absmax_input", j, input_name)

            for m, output_name in enumerate(op.output):
                self.update_max(op, "absmax_output", m, output_name)

    def get_kl_hist(self, data, min_val, max_val, name):
        """get kl hist"""
        if min_val >= 0:
            hist_iter, hist_edges_iter = np.histogram(data, bins=2048, range=(min_val, max_val))
        else:
            th = max(abs(min_val), abs(max_val))
            min_range = -th
            hist_iter, hist_edges_iter = np.histogram(data, bins=2048, range=(-th, th))

        global hist
        global hist_edges
        if name not in hist:
            hist[name] = np.array(hist_iter)
            hist_edges[name] = np.array(hist_edges_iter)
        else:
            hist[name] += np.array(hist_iter)

    def expand_quantized_bins(self, quantized_bins, reference_bins):
        """expand quantized bins"""
        expanded_quantized_bins = [0]*len(reference_bins)
        num_merged_bins = int(len(reference_bins)/len(quantized_bins))
        j_start = 0
        j_end = num_merged_bins
        for idx in xrange(len(quantized_bins)): #pylint: disable=undefined-variable
            zero_count = reference_bins[j_start:j_end].count(0)
            num_merged_bins = j_end-j_start
            if zero_count == num_merged_bins:
                avg_bin_ele = 0
            else:
                avg_bin_ele = quantized_bins[idx]/(num_merged_bins - zero_count + 0.0)
            for idx1 in xrange(j_start, j_end): #pylint: disable=undefined-variable
                expanded_quantized_bins[idx1] = (0 if reference_bins[idx1] == 0
                                                 else avg_bin_ele)
            j_start += num_merged_bins
            j_end += num_merged_bins
            if (idx+1) == len(quantized_bins) - 1:
                j_end = len(reference_bins)
        return expanded_quantized_bins

    def safe_entropy(self, reference_distr_P, P_sum, candidate_distr_Q, Q_sum):
        """ safe entropy """
        assert len(reference_distr_P) == len(candidate_distr_Q)
        tmp_sum1 = 0
        tmp_sum2 = 0
        for idx in range(len(reference_distr_P)):  ##pylint: disable=consider-using-enumerate
            p_idx = reference_distr_P[idx]
            q_idx = candidate_distr_Q[idx]
            if p_idx == 0:
                tmp_sum1 += 0
                tmp_sum2 += 0
            else:
                if q_idx == 0:
                    print("Fatal error!, idx = " + str(idx) +
                          " qindex = 0! p_idx = " + str(p_idx))
                tmp_sum1 += p_idx * (math.log(Q_sum*p_idx))
                tmp_sum2 += p_idx * (math.log(P_sum*q_idx))
        return (tmp_sum1 - tmp_sum2)/P_sum

    def get_optimal_scaling_factor(self, hist, hist_edges, P_sum, max_val, min_val,
                                   num_quantized_bins=255):
        """get the optimal scaling factor"""
        if min_val >= 0:
            ending_iter = 2047
            starting_iter = int(ending_iter * 0.7)
        else:
            th = max(abs(max_val), abs(min_val))
            starting_iter = 0
            ending_iter = 2047
            if abs(max_val) > abs(min_val):
                while starting_iter < ending_iter:
                    if hist[starting_iter] == 0:
                        starting_iter += 1
                        continue
                    else:
                        break
                starting_iter += int((ending_iter - starting_iter)*0.6)
            else:
                while ending_iter > 0:
                    if hist[ending_iter] == 0:
                        ending_iter -= 1
                        continue
                    else:
                        break
                starting_iter = int(0.6 * ending_iter)

        bin_width = hist_edges[1]-hist_edges[0]
        min_kl_divergence = 0
        min_kl_index = 0
        kl_inited = False

        for i in range(starting_iter, ending_iter+1):
            reference_distr_P = hist[0:i].tolist()
            outliers_count = sum(hist[i:2048])
            if reference_distr_P[i-1] == 0:
                continue
            reference_distr_P[i-1] += outliers_count
            reference_distr_bins = reference_distr_P[:]
            candidate_distr_Q = hist[0:i].tolist()
            num_merged_bins = int(i/num_quantized_bins)
            candidate_distr_Q_quantized = [0]*num_quantized_bins
            j_start = 0
            j_end = num_merged_bins

            for idx in xrange(num_quantized_bins): #pylint: disable=undefined-variable
                candidate_distr_Q_quantized[idx] = sum(candidate_distr_Q[j_start:j_end])
                j_start += num_merged_bins
                j_end += num_merged_bins
                if (idx+1) == num_quantized_bins - 1:
                    j_end = i
            candidate_distr_Q = self.expand_quantized_bins(candidate_distr_Q_quantized,
                                                           reference_distr_bins)
            Q_sum = sum(candidate_distr_Q)
            kl_divergence = self.safe_entropy(reference_distr_P, P_sum,
                                              candidate_distr_Q, Q_sum)
            if not kl_inited:
                min_kl_divergence = kl_divergence
                min_kl_index = i
                kl_inited = True
            elif kl_divergence < min_kl_divergence:
                min_kl_divergence = kl_divergence
                min_kl_index = i
            else:
                pass

        if min_kl_index == 0:
            while starting_iter > 0:
                if hist[starting_iter] == 0:
                    starting_iter -= 1
                    continue
                else:
                    break
            min_kl_index = starting_iter
        return (min_kl_index+0.5)*bin_width


class AbsmaxCalib(algorithm):
    """ AbsMax calibrator"""
    def get_max(self, op, tensor, tensor_idx, tensor_name, max_name):
        name = max_name + "_" + str(tensor_idx)
        absmax = np.array([np.absolute(tensor).max()]).astype(np.float32)

        arg = self.get_arg(op, name)
        if arg is not None:
            orig_absmax = np.array(arg.floats).astype(np.float32)
            assert orig_absmax.shape == absmax.shape
            absmax = np.maximum(absmax, orig_absmax).astype(np.float32)
            self.remove_arg(op, name)

        max_arg = utils.MakeArgument(name, absmax)
        # save max vaules in predict_def as operator arguments
        op.arg.extend([max_arg])

class EMACalib(algorithm):
    """ Moving Average calibrator """
    def __init__(self, ema_alpha=0.5):
        self.ema_alpha = ema_alpha

    def get_max(self, op, tensor, tensor_idx, tensor_name, max_name):
        name = max_name + "_" + str(tensor_idx)
        absmax = np.array([np.absolute(tensor).max()]).astype(np.float32)

        arg = self.get_arg(op, name)
        if arg is not None:
            orig_absmax = np.array(arg.floats).astype(np.float32)
            assert orig_absmax.shape == absmax.shape
            absmax = np.array([self.ema_alpha * absmax + (1-self.ema_alpha) * orig_absmax]).astype(np.float32)
            self.remove_arg(op, name)

        max_arg = utils.MakeArgument(name, absmax)
        # save max vaules in predict_def as operator arguments
        op.arg.extend([max_arg])


class Calibrator(object):
    """main calss for calibrator"""
    def __init__(self, algorithm, device_option=None):
        self.algo = algorithm
        self.dev_opt = device_option

    def RunCalibIter(self, ws, predict_def):
        """run calibrator in iteration"""
        def wino_transform_data(blob):
            blob_wino = blob.copy()
            B_T = np.array([[0.87890625, 0, -2.640625, 0, 1, 0],
                            [0, -1.40625, -2.25, 0.625, 1, 0],
                            [0, 1.40625, -2.25, -0.625, 1, 0],
                            [0, -0.5859375, -0.390625, 1.5, 1, 0],
                            [0, 0.5859375, -0.390625, -1.5, 1, 0],
                            [0, 0.87890625, 0, -2.640625, 0, 1]])
            B = B_T.T
            blob_cell = np.zeros((6, 6), dtype=np.float)
            for n in range(blob.shape[0]):
                for c in range(blob.shape[1]):
                    for h in range(0, blob.shape[2], 4):
                        for w in range(0, blob.shape[3], 4):
                            for i in range(min(6, blob.shape[2]-h)):
                                for j in range(min(6, blob.shape[3]-w)):
                                    blob_cell[i][j] = blob[n][c][h + i][w + j]
                            blob_wino_tmp = np.dot(B_T, blob_cell)
                            blob_wino_cell = np.dot(blob_wino_tmp, B)
                            for i in range(min(6, blob.shape[2]-h)):
                                for j in range(min(6, blob.shape[3]-w)):
                                    blob_wino[n][c][h + i][w + j] = blob_wino_cell[i][j]
                            if (w + 6) >= blob.shape[3]:
                                break
                        if (h + 6) >= blob.shape[2]:
                            break
            return blob_wino

        for op in predict_def.op[0:]:
            for j, input_name in enumerate(op.input):
                if self.algo.is_weights(op, j) or self.algo.is_bias(op, j):
                    continue
                inp = ws.FetchBlob(input_name)
                self.algo.get_max(op, inp, j, input_name, "absmax_input")

                if j > 0:
                    continue
                calib_euler = self.algo.get_arg(op, "calib_euler")
                if calib_euler and calib_euler.i:
                    algo_wino = self.algo.get_arg(op, "conv_algorithm")
                    if algo_wino and algo_wino.i:
                        arg = self.algo.get_arg(op, "pad")
                        assert arg is not None
                        input_pad = np.zeros((inp.shape[0],
                                              inp.shape[1],
                                              inp.shape[2] + 2 * arg.i,
                                              inp.shape[3] + 2 * arg.i),
                                             dtype=np.float)
                        input_pad[:, :, arg.i : inp.shape[2]+arg.i, arg.i : inp.shape[3]+arg.i] = inp
                        wino_trans = wino_transform_data(input_pad)
                        self.algo.get_max_min(op, wino_trans, j, input_name, "wino_tinput_quant")
            this_op = copy.deepcopy(op)
            if self.dev_opt is not None:
                this_op.device_option.CopyFrom(self.dev_opt)
            ws.RunOperatorOnce(this_op)

            for m, output_name in enumerate(op.output):
                outp = ws.FetchBlob(output_name)
                self.algo.get_max(op, outp, m, output_name, "absmax_output")

        self.algo.update_status()
        return predict_def

    def DepositQuantizedModule(self, ws, predict_def):
        """deposit quantized module"""
        DATA_TYPE_UND = 0
        DATA_TYPE_FP32 = 1
        DATA_TYPE_S32 = 2
        DATA_TYPE_S16 = 4
        DATA_TYPE_S8 = 5
        DATA_TYPE_U8 = 6

        PER_CHANNEL_WEIGHTS = True

        def get_zero_point(data_type):
            return {
                DATA_TYPE_S32   : 0,
                DATA_TYPE_S8    : 128,
                DATA_TYPE_U8    : 0,
            }.get(data_type, None)

        def get_abs_max(data_type):
            return {
                DATA_TYPE_S32   : 0x7FFFFFFF,
                DATA_TYPE_S16   : 0x7FFF,
                DATA_TYPE_S8    : 0x7F,
                DATA_TYPE_U8    : 0xFF,
            }.get(data_type, None)

        def get_quantized_op_type(op_type):
            return {
                "Conv"            : "Int8Conv",
                "Relu"            : "Int8Relu",
                "Sum"             : "Int8Sum",
                "Add"             : "Int8Add",
                "MaxPool"         : "Int8MaxPool",
                "AveragePool"     : "Int8AveragePool",
                "UpsampleNearest" : "Int8UpsampleNearest",
                "FC"              : "Int8FC",
            }.get(op_type, None)

        def get_quantized_op_type_by_fusion_type(fusion_type):
            return {
                1 : "Int8ConvRelu",
                2 : "Int8ConvRelu",
                3 : "Int8ConvSum",
                4 : "Int8ConvSumRelu",
                5 : "Int8ConvSumRelu",

            }.get(fusion_type, None)

        def get_output_format(op_type):
            return {
                "FC"                : "NC",
                "Int8FC"            : "NC",
                "Conv"              : "NCHW",
                "ConvFusion"        : "NCHW",
                "NHWC2NCHW"         : "NCHW",
                "NCHW2NHWC"         : "NHWC",
                "Int8Conv"          : "NHWC",
                "Int8ConvRelu"      : "NHWC",
                "Int8ConvSum"       : "NHWC",
                "Int8ConvSumRelu"   : "NHWC",
            }.get(op_type, None)

        def not_need_quantize(op_type):
            if not op_type.startswith("Int8"):
                return True
            return {
                "Int8Quantize"      : True,
                "Int8Dequantize"    : True,
            }.get(op_type, False)

        def not_need_dequantize(op_type):
            if op_type.startswith("Int8"):
                return True
            return {
                "NCHW2NHWC" : True,
                "NHWC2NCHW" : True,
            }.get(op_type, False)

        def not_need_output_scale(op_type):
            if not op_type.startswith("Int8"):
                return True
            return {
                "Int8Dequantize" : True
            }.get(op_type, False)

        def is_data_type_changed(op_type):
            key_type_segment = ["Conv", "Sum", "FC", "Concat"]
            for key in key_type_segment:
                return op_type.find(key) != -1
            return False

        def has_weights(op):
            return {
                "Int8FC"            : True,
                "Int8Conv"          : True,
                "Int8ConvRelu"      : True,
                "Int8ConvSum"       : True,
                "Int8ConvSumRelu"   : True,
            }.get(op.type, False)

        def has_bias(op):
            inp_len = len(op.input)
            return {
                "Int8FC"            : inp_len == 3,
                "Int8Conv"          : inp_len == 3,
                "Int8ConvRelu"      : inp_len == 3,
                "Int8ConvSum"       : inp_len == 4,
                "Int8ConvSumRelu"   : inp_len == 4,
            }.get(op.type, False)

        def predict_output_format(predict_def, op_pos):
            if op_pos is None:
                return "NCHW"
            cur_pos = op_pos
            op = predict_def.op[cur_pos]
            while op is not None:
                fmt = get_output_format(op.type)
                if fmt is not None:
                    return fmt
                op, cur_pos = self.algo.get_predecessor_op(0, cur_pos, predict_def)
            return "NCHW"

        def update_op_type(predict_def):
            for i, op in enumerate(predict_def.op):
                op_type = get_quantized_op_type(op.type)
                if op_type is not None:
                    op.type = op_type
                    continue
                if op.type == "ConvFusion":
                    arg = self.algo.get_arg(op, "fusion_type")
                    assert arg is not None
                    op_type = get_quantized_op_type_by_fusion_type(arg.i)
                    assert op_type is not None
                    op.type = op_type

        def add_order_swtich(predict_def, op_pos, op_type="NCHW2NHWC", is_insert=True):
            op = predict_def.op[op_pos]
            if is_insert:
                insert_pos = op_pos
                data_in = op.input[0]
                data_out = data_in + "_" + op_type + "_" + str(op_pos)
                op.input[0] = data_out
            else:
                insert_pos = op_pos + 1
                data_out = op.output[0]
                data_in = data_out + "_" + op_type + "_" + str(op_pos)
                op.output[0] = data_in
            order_sw_op = core.CreateOperator(op_type, [data_in], [data_out])
            self.algo.insert_op(insert_pos, order_sw_op, predict_def)

        def reset_arg(op, name, value):
            for i in range(len(op.arg)):
                if op.arg[i].name == name:
                    op.arg[i].i = value;

        def reset_conv_algorithm(predict_def):
            cur_pos = 0
            op_len = len(predict_def.op)
            while cur_pos < op_len:
                cur_op = predict_def.op[cur_pos]
                if not cur_op.type.startswith("Int8Conv"):
                    cur_pos += 1
                    continue
                is_neg_input = True
                pre_op, pre_pos = self.algo.get_predecessor_op(0, cur_pos, predict_def)
                Y_zero_point = self.algo.get_arg(pre_op, "Y_zero_point")
                if pre_op and Y_zero_point.i == 0:
                    is_neg_input = False
                conv_algo = self.algo.get_arg(cur_op, "conv_algorithm")
                if is_neg_input and conv_algo and conv_algo.i == 1:
                    # MKL-DNN only supports int8 winograd conv for the input in u8 datatype.
                    # If the input is in s8 datatype, we reset the conv algorithm as direct conv.
                    reset_arg(cur_op, "conv_algorithm", 0)
                cur_pos += 1

        def insert_quantize(predict_def):
            cur_pos = 0
            op_len = len(predict_def.op)
            while cur_pos < op_len:
                op = predict_def.op[cur_pos]
                if not_need_quantize(op.type):
                    cur_pos += 1
                    continue
                inp_index = 0
                inp_len = len(op.input)
                new_pos = cur_pos
                while inp_index < inp_len:
                    op = predict_def.op[new_pos]
                    pre_op, pre_pos = self.algo.get_predecessor_op(inp_index, new_pos, predict_def)
                    if pre_op is None:
                        if inp_index != 0 or new_pos != 0:
                            inp_index += 1
                            continue
                    elif pre_op.type.startswith("Int8"):
                        inp_index += 1
                        continue
                    inp = op.input[inp_index]
                    outp = inp + "_quantized_" + str(cur_pos) + "_" + str(inp_index)
                    op.input[inp_index] = outp
                    qua_op = core.CreateOperator("Int8Quantize", [inp], [outp])
                    self.algo.insert_op(new_pos, qua_op, predict_def)
                    if predict_output_format(predict_def, pre_pos) == "NCHW":
                        add_order_swtich(predict_def, new_pos)
                        op_len += 1
                        new_pos += 1
                    if pre_op is None:
                        qua_data = predict_def.op[new_pos].output[0]
                        successors = self.algo.get_successor_ops(-1, predict_def)
                        for op in successors:
                            if op.type.startswith("Int8") and op.input[0] != qua_data:
                                op.input[0] = qua_data
                    op_len += 1
                    new_pos += 1
                    inp_index += 1
                cur_pos = new_pos + 1

        def insert_dequantize(predict_def):
            cur_pos = 0
            op_len = len(predict_def.op)
            while cur_pos < op_len:
                op = predict_def.op[cur_pos]
                if not_need_dequantize(op.type):
                    cur_pos += 1
                    continue
                pre_op, pre_pos = self.algo.get_predecessor_op(0, cur_pos, predict_def)
                if pre_op is None or not pre_op.type.startswith("Int8"):
                    cur_pos += 1
                    continue
                inp = op.input[0]
                outp = inp + "_dequantized_" + str(cur_pos)
                op.input[0] = outp
                deq_op = core.CreateOperator("Int8Dequantize", [inp], [outp])
                self.algo.insert_op(cur_pos, deq_op, predict_def)
                if predict_output_format(predict_def, pre_pos) == "NHWC":
                    add_order_swtich(predict_def, cur_pos, "NHWC2NCHW", False)
                    op_len += 1
                    cur_pos += 1
                op_len += 1
                cur_pos += 2

        def refine_module_outputs(predict_def):
            cur_pos = 0
            op_len = len(predict_def.op)
            while cur_pos < op_len:
                op = predict_def.op[cur_pos]
                if not_need_quantize(op.type):
                    cur_pos += 1
                    continue
                successors = self.algo.get_successor_ops(cur_pos, predict_def)
                if len(successors) > 0:
                    cur_pos += 1
                    continue
                deq_inp = op.output[0] + "_orig_" + str(cur_pos)
                deq_outp = op.output[0]
                op.output[0] = deq_inp
                if predict_output_format(predict_def, cur_pos) == "NHWC":
                    order_sw_inp = deq_outp + "_dequantized_" + str(cur_pos)
                    order_sw_outp = deq_outp
                    deq_outp = order_sw_inp
                    deq_op = core.CreateOperator("Int8Dequantize", [deq_inp], [deq_outp])
                    order_sw_op = core.CreateOperator("NHWC2NCHW", [order_sw_inp], [order_sw_outp])
                    predict_def.op.extend([deq_op, order_sw_op])
                    op_len += 2
                else:
                    deq_op = core.CreateOperator("Int8Dequantize", [deq_inp], [deq_outp])
                    predict_def.op.extend([deq_op])
                    op_len += 1
                cur_pos += 1

        def add_storage_order(predict_def):
            order_arg = utils.MakeArgument("order", str("NHWC"))
            for op in predict_def.op:
                if not op.type.startswith("Int8"):
                    continue
                if op.type == "Int8Quantize" or op.type == "Int8Dequantize":
                    continue
                arg = self.algo.get_arg(op, "order")
                if arg is not None:
                    arg.s = str("NHWC")
                else:
                    op.arg.extend([order_arg])

        def predict_output_data_type(predict_def):
            output_data_type = []
            pos = 0
            while pos < len(predict_def.op):
                op = predict_def.op[pos]
                if not op.type.startswith("Int8"):
                    output_data_type.append(DATA_TYPE_FP32)
                elif op.type == "Int8Quantize":
                    output_data_type.append(DATA_TYPE_S8)
                elif op.type == "Int8Dequantize":
                    output_data_type.append(DATA_TYPE_FP32)
                elif op.type.endswith("Relu"):
                    output_data_type.append(DATA_TYPE_U8)
                elif is_data_type_changed(op.type):
                    output_data_type.append(DATA_TYPE_S8)
                else:
                    _, pre_pos = self.algo.get_predecessor_op(0, pos, predict_def)
                    if pre_pos is None:
                        output_data_type.append(DATA_TYPE_S8)
                    elif output_data_type[pre_pos] == DATA_TYPE_FP32:
                        output_data_type.append(DATA_TYPE_S8)
                    else:
                        output_data_type.append(output_data_type[pre_pos])
                pos += 1
            return output_data_type

        def add_output_scale(predict_def, output_data_type):
            for i, op in enumerate(predict_def.op):
                if not_need_output_scale(op.type):
                    continue
                if op.type == "Int8Quantize":
                    successors = self.algo.get_successor_ops(i, predict_def)
                    assert len(successors) > 0
                    successor = successors[0]
                    input_index = self.algo.get_input_index(op.output[0], successor)
                    arg_name = "absmax_input" + "_" + str(input_index)
                    arg = self.algo.get_arg(successor, arg_name)
                    assert arg is not None
                    output_scale = arg.floats[0] / get_abs_max(output_data_type[i])
                elif is_data_type_changed(op.type):
                    arg_name = "absmax_output" + "_" + str(0)
                    arg = self.algo.get_arg(op, arg_name)
                    assert arg is not None
                    output_scale = arg.floats[0] / get_abs_max(output_data_type[i])
                else:
                    pre_op, _ = self.algo.get_predecessor_op(0, i, predict_def)
                    assert pre_op is not None
                    arg = self.algo.get_arg(pre_op, "Y_scale")
                    assert arg is not None
                    output_scale = arg.f
                self.algo.remove_arg(op, "Y_scale")
                op.arg.extend([utils.MakeArgument("Y_scale", output_scale)])
                self.algo.remove_arg(op, "Y_zero_point")
                output_zero_point = get_zero_point(output_data_type[i])
                op.arg.extend([utils.MakeArgument("Y_zero_point", output_zero_point)])

                calib_euler = self.algo.get_arg(op, "calib_euler")
                if calib_euler and calib_euler.i:
                    algo_wino = self.algo.get_arg(op, "conv_algorithm")
                    if algo_wino and algo_wino.i:
                        arg_name = "wino_tinput_quant"
                        max_min = self.algo.get_arg(op, "wino_tinput_quant")
                        assert max_min is not None
                        delta = max_min.floats[0] - max_min.floats[1] + 0.000001
                        wino_scale = delta / get_abs_max(DATA_TYPE_U8)
                        wino_zero_point = -math.ceil(max_min.floats[1] * get_abs_max(DATA_TYPE_U8) / delta)
                        wino_tinput_quant = np.array([wino_scale, wino_zero_point]).astype(np.float32)
                        self.algo.remove_arg(op, "wino_tinput_quant")
                        op.arg.extend([utils.MakeArgument("wino_tinput_quant", wino_tinput_quant)])

        def float_weights(ws, op, init_def):
            assert len(op.input) >= 2
            weights = ws.FetchBlob(op.input[1]).astype(np.float32)
            shape = weights.shape
            weights_extend = self.algo.get_arg(op, "weights_extend")
            if weights_extend and weights_extend.i:
                shape = (shape[0], shape[1], 1, 1)
            filler = core.CreateOperator(
                "GivenTensorFill",
                [], [op.input[1]],
                arg=[
                    utils.MakeArgument("shape", shape),
                    utils.MakeArgument("values", weights),
                    ]
            )
            init_def.op.extend([filler])

        def float_bias(ws, op, init_def):
            assert len(op.input) >= 3
            bias = ws.FetchBlob(op.input[2]).astype(np.float32)
            bias = core.CreateOperator(
                "GivenTensorFill",
                [], [op.input[2]],
                arg=[
                    utils.MakeArgument("shape", bias.shape),
                    utils.MakeArgument("values", bias),
                    ]
            )
            init_def.op.extend([bias])

        def quantize_weights(ws, op, init_def):
            assert len(op.input) >= 2
            weights = ws.FetchBlob(op.input[1]).astype(np.float32)
            if PER_CHANNEL_WEIGHTS:
                absmax = np.array([np.absolute(weights[i, ...]).max()
                                   for i in range(weights.shape[0])]).astype(np.float32)
            else:
                absmax = np.array([np.absolute(weights).max()]).astype(np.float32)
            output_scale = absmax / get_abs_max(DATA_TYPE_S8)
            output_zero_point = get_zero_point(DATA_TYPE_S8)
            if len(weights.shape) == 4:
                weights = np.transpose(weights, (0, 2, 3, 1)).astype(np.float32)
            if output_scale.shape[0] == weights.shape[0]:
                assert len(output_scale.shape) == 1
                values = np.rint([weights[i, ...] / output_scale[i]
                                  for i in range(weights.shape[0])]).astype(np.int8) + output_zero_point
            else:
                assert output_scale.size == 1
                values = np.rint((weights / output_scale[0])).astype(np.int8) + output_zero_point
            filler = core.CreateOperator(
                "Int8GivenTensorFill",
                [], [op.input[1]],
                arg=[
                    utils.MakeArgument("shape", weights.shape),
                    utils.MakeArgument("values", values.astype(np.uint8).tobytes()),
                    utils.MakeArgument("Y_zero_point", output_zero_point),
                    utils.MakeArgument("Y_scale", output_scale)
                    if output_scale.size == 1 else utils.MakeArgument("Y_scales", output_scale),
                    ]
            )
            init_def.op.extend([filler])
            return output_scale

        def quantize_bias(ws, op, init_def, input_scale, weights_scale):
            assert len(op.input) >= 3
            output_scale = weights_scale * input_scale
            output_zero_point = get_zero_point(DATA_TYPE_S32)
            bias = ws.FetchBlob(op.input[2]).astype(np.float32)
            if output_scale.shape[0] == bias.shape[0]:
                assert len(output_scale.shape) == 1
                values = np.rint([bias[i] / output_scale[i]
                                  for i in range(bias.shape[0])]).astype(np.int32)
            else:
                assert output_scale.size == 1
                values = np.rint(bias / output_scale[0]).astype(np.int32)
            filler = core.CreateOperator(
                "Int8GivenIntTensorFill",
                [], [op.input[2]],
                arg=[
                    utils.MakeArgument("shape", bias.shape),
                    utils.MakeArgument("values", values),
                    utils.MakeArgument("Y_zero_point", output_zero_point),
                    utils.MakeArgument("Y_scale", output_scale)
                    if output_scale.size == 1 else utils.MakeArgument("Y_scales", output_scale),
                    ]
            )
            init_def.op.extend([filler])

        def gen_quantized_init_def(ws, predict_def):
            init_def = caffe2_pb2.NetDef()
            init_def.name = predict_def.name + "_weights_bias"
            for i, op in enumerate(predict_def.op):
                if not op.type.startswith("Int8"):
                    continue
                if has_weights(op):
                    calib_euler = self.algo.get_arg(op, "calib_euler")

                    weights_filler = self.algo.get_op_by_output(op.input[1], init_def)
                    if weights_filler is None:
                        if calib_euler and calib_euler.i:
                            float_weights(ws, op, init_def)
                        else:
                            weights_scale = quantize_weights(ws, op, init_def)

                    if has_bias(op):
                        bias_filler = self.algo.get_op_by_output(op.input[2], init_def)
                        if bias_filler is not None:
                            continue
                        if calib_euler and calib_euler.i:
                            float_bias(ws, op, init_def)
                            continue

                        if weights_filler is not None:
                            arg = self.algo.get_arg(weights_filler, "Y_scale")
                            if arg is not None:
                                weights_scale = np.array([arg.f]).astype(np.float32)
                            else:
                                arg = self.algo.get_arg(weights_filler, "Y_scales")
                                assert arg is not None
                                weights_scale = np.array(arg.floats).astype(np.float32)
                        pre_op, _ = self.algo.get_predecessor_op(0, i, predict_def)
                        assert pre_op is not None
                        arg = self.algo.get_arg(pre_op, "Y_scale")
                        assert arg is not None
                        assert weights_scale is not None
                        quantize_bias(ws, op, init_def, arg.f, weights_scale)
            return init_def

        def organize_external_input(ws, predict_def, init_def):
            kTypeNameMapper = {
                np.dtype("float32") : "GivenTensorFill",
                np.dtype("int32")   : "GivenTensorIntFill",
                np.dtype("int64")   : "GivenTensorInt64Fill",
                np.dtype("uint8")   : "GivenTensorStringFill",
            }
            all_existing_inputs = []
            for op in init_def.op:
                all_existing_inputs.append(op.output[0])
            for inp in predict_def.external_input:
                if inp == predict_def.op[0].input[0]:
                    continue
                if inp in all_existing_inputs:
                    continue
                in_data = ws.FetchBlob(inp)
                shape = in_data.shape
                values = in_data
                # pass array of uint8 as a string to save storage
                # storing uint8_t has a large overhead for now
                if in_data.dtype == np.dtype("uint8"):
                    shape = [1]
                    values = [str(in_data.data)]
                op = core.CreateOperator(
                    kTypeNameMapper[in_data.dtype],
                    [], [inp],
                    arg=[
                        utils.MakeArgument("shape", shape),
                        utils.MakeArgument("values", values),
                    ]
                )
                init_def.op.extend([op])

        predict_quantized = copy.deepcopy(predict_def)

        if os.environ.get('DEBUGMODE') == "1":
            for i, op in enumerate(predict_quantized.op):
                if len(op.name) == 0:
                    op.name = op.type.lower() + str(i)

        self.algo.gather_max(predict_quantized)

        self.algo.update_status()

        update_op_type(predict_quantized)

        insert_dequantize(predict_quantized)

        insert_quantize(predict_quantized)

        refine_module_outputs(predict_quantized)

        if os.environ.get('DEBUGMODE') == "1":
            with open("{0}_calib_predict_net.pb".format(predict_quantized.name), "w") as fid:
                fid.write(predict_quantized.SerializeToString())
            with open("{}_calib_predict_net.pbtxt".format(predict_quantized.name), "w") as fid:
                fid.write(str(predict_quantized))

        # DO NOT change the operator order of the module after below line
        output_data_type = predict_output_data_type(predict_quantized)

        add_output_scale(predict_quantized, output_data_type)

        add_storage_order(predict_quantized)

        init_quantized = gen_quantized_init_def(ws, predict_quantized)

        self.algo.remove_max(predict_quantized)

        reset_conv_algorithm(predict_quantized)

        for op in predict_quantized.op:
            if op.type.startswith("Int8"):
                op.engine = str("DNNLOWP")
            #self.algo.remove_arg(op, "fusion_type")
            op.device_option.CopyFrom(caffe2_pb2.DeviceOption())

        organize_external_input(ws, predict_quantized, init_quantized)

        return predict_quantized, init_quantized
