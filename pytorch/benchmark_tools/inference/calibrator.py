"""calibration tool"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
        if op_pos < 0 or op_pos >= (len(predict_def.op) - 1):
            return []
        successors = []
        pos = op_pos + 1
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
                self.remove_arg(op, 'absmax_input'+ '_' + str(i))
            for j in range(len(op.output)):
                self.remove_arg(op, 'absmax_output'+ '_' + str(j))

    def get_max(self, op, blob, max_name, tensor_idx, tensor_name):
        raise Exception("Please add max value computation method!")

    def gather_max(self, predict_def):
        pass

    def update_status(self):
        pass


class KLCalib(algorithm):
    """clibrator of KL"""
    def __init__(self, kl_iter_num_for_range=100):
        self.kl_iter_num_for_range = kl_iter_num_for_range

    def update_status(self):
        global iteration_idx
        iteration_idx += 1

    def get_max(self, op, blob, max_name, tensor_idx, tensor_name):
        global iteration_idx
        name = max_name + "_" + str(tensor_idx)
        op_hist_name = tensor_name + "_" + max_name + "_" + str(tensor_idx)

        arg = self.get_arg(op, name)
        if iteration_idx < self.kl_iter_num_for_range:
            max_min = np.array([np.max(blob), np.min(blob)]).astype(np.float32)
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
            self.get_kl_hist(blob, min_val, max_val, op_hist_name)

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
        layer_max = self.get_optimal_scaling_factor(hist_iter,
                                                    hist_edges_iter, P_sum, max_val, min_val)

        self.remove_arg(op, name)
        max_arg = utils.MakeArgument(name, np.array([layer_max]).astype(np.float32))
        # save max vaules in predict_def as operator arguments
        op.arg.extend([max_arg])

    def gather_max(self, predict_def):
        for op in predict_def.op[0:]:
            for j, input_name in enumerate(op.input):
                max_name = 'absmax_input'
                self.update_max(op, max_name, j, input_name)

            for m, output_name in enumerate(op.output):
                max_name = 'absmax_output'
                self.update_max(op, max_name, m, output_name)

    def get_kl_hist(self, data, min_val, max_val, name):
        hist_iter, hist_edges_iter = np.histogram(data, bins=2048,
                                                  range=(min_val, max_val))
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
        """safe entropy"""
        assert len(reference_distr_P) == len(candidate_distr_Q)
        tmp_sum1 = 0
        tmp_sum2 = 0
        for idx, _ in enumerate(reference_distr_P):
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
    """calibrator for AbsMax"""
    def get_max(self, op, blob, max_name, tensor_idx, tensor_name):
        name = max_name + "_" + str(tensor_idx)
        arg = self.get_arg(op, name)
        absmax = np.array([np.absolute(blob).max()]).astype(np.float32)

        if arg is not None:
            orig_absmax = arg.floats[0]
            absmax = np.array([np.absolute([orig_absmax, absmax]).max()]).astype(np.float32)
            self.remove_arg(op, name)

        max_arg = utils.MakeArgument(name, absmax)
        # save max vaules in predict_def as operator arguments
        op.arg.extend([max_arg])


class EMACalib(algorithm):
    """calibrator for moving average"""
    def __init__(self, ema_alpha=0.5):
        self.ema_alpha = ema_alpha

    def get_max(self, op, blob, max_name, tensor_idx, tensor_name):
        name = max_name + "_" + str(tensor_idx)
        arg = self.get_arg(op, name)
        absmax = np.array([np.absolute(blob).max()]).astype(np.float32)

        if arg is not None:
            orig_absmax = arg.floats[0]
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
        for op in predict_def.op[0:]:
            for j, input_name in enumerate(op.input):
                input_blob = ws.FetchBlob(input_name)
                max_name = 'absmax_input'
                self.algo.get_max(op, input_blob, max_name, j, input_name)

            this_op = copy.deepcopy(op)
            if self.dev_opt is not None:
                this_op.device_option.CopyFrom(self.dev_opt)
            ws.RunOperatorOnce(this_op)

            for m, output_name in enumerate(op.output):
                output_blob = ws.FetchBlob(output_name)
                max_name = 'absmax_output'
                self.algo.get_max(op, output_blob, max_name, m, output_name)

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
                "Conv"          : "Int8Conv",
                "Relu"          : "Int8Relu",
                "Sum"           : "Int8Sum",
                "Add"           : "Int8Add",
                "MaxPool"       : "Int8MaxPool",
                "AveragePool"   : "Int8AveragePool",
                # Int8FC is not supported so far
                #"FC"            : "Int8FC",
            }.get(op_type, None)

        def get_quantized_op_type_by_fusion_type(fusion_type):
            return {
                1 : "Int8ConvRelu",
                2 : "Int8ConvSum",
                3 : "Int8ConvSumRelu",
            }.get(fusion_type, None)

        def get_output_format(op_type):
            if op_type.startswith("Conv"):
                return "NCHW"
            if op_type.startswith("Int8Conv"):
                return "NHWC"
            if op_type.endswith("FC"):
                return "NC"
            return {
                "NCHW2NHWC" :   "NHWC",
                "NHWC2NCHW" :   "NCHW",
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
                if op_type.find(key) != -1:
                    return True
            return False

        def has_weights(op):
            key_type_segment = ["Int8Conv", "Int8FC"]
            for key in key_type_segment:
                if op.type.startswith(key):
                    return True
            return False

        def has_bias(op):
            if op.type.startswith("Int8Conv"):
                if op.type.find("Sum") != -1:
                    if len(op.input) == 4:
                        return True
                elif len(op.input) == 3:
                    return True
                return False
            elif op.type.startswith("Int8FC") and len(op.input) == 3:
                return True
            return False

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
            for _, op in enumerate(predict_def.op):
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
                else:
                    arg_name = "absmax_output" + "_" + str(0)
                    arg = self.algo.get_arg(op, arg_name)
                assert arg is not None
                output_scale = arg.floats[0] / get_abs_max(output_data_type[i])
                self.algo.remove_arg(op, "Y_scale")
                op.arg.extend([utils.MakeArgument("Y_scale", output_scale)])
                self.algo.remove_arg(op, "Y_zero_point")
                output_zero_point = get_zero_point(output_data_type[i])
                op.arg.extend([utils.MakeArgument("Y_zero_point", output_zero_point)])

        def quantize_weights(ws, op, init_def):
            assert len(op.input) >= 2
            weights = ws.FetchBlob(op.input[1]).astype(np.float32)
            if len(weights.shape) == 4:
                weights = np.transpose(weights, (0, 2, 3, 1)).astype(np.float32)
            arg = self.algo.get_arg(op, "absmax_input" + "_" + str(1))
            assert arg is not None
            output_scale = arg.floats[0] / get_abs_max(DATA_TYPE_S8)
            output_zero_point = get_zero_point(DATA_TYPE_S8)
            values = np.rint((weights / output_scale)).astype(np.int8) + output_zero_point
            filler = core.CreateOperator(
                "Int8GivenTensorFill",
                [], [op.input[1]],
                arg=[
                    utils.MakeArgument("shape", weights.shape),
                    utils.MakeArgument("values", values.astype(np.uint8).tobytes()),
                    utils.MakeArgument("Y_zero_point", output_zero_point),
                    utils.MakeArgument("Y_scale", output_scale)])
            init_def.op.extend([filler])
            return output_scale

        def quantize_bias(ws, op, init_def, input_data_type, weights_scale):
            assert len(op.input) >= 3
            bias = ws.FetchBlob(op.input[2]).astype(np.float32)
            arg = self.algo.get_arg(op, "absmax_input" + "_" + str(0))
            assert arg is not None
            input_scale = arg.floats[0] / get_abs_max(input_data_type)
            output_scale = input_scale * weights_scale
            output_zero_point = get_zero_point(DATA_TYPE_S32)
            values = np.rint(bias / output_scale).astype(np.int32)
            filler = core.CreateOperator(
                "Int8GivenIntTensorFill",
                [], [op.input[2]],
                arg=[
                    utils.MakeArgument("shape", bias.shape),
                    utils.MakeArgument("values", values),
                    utils.MakeArgument("Y_zero_point", output_zero_point),
                    utils.MakeArgument("Y_scale", output_scale)])
            init_def.op.extend([filler])

        def gen_quantized_init_def(ws, predict_def, output_data_type):
            init_def = caffe2_pb2.NetDef()
            init_def.name = predict_def.name + "_weights_bias"
            for i, op in enumerate(predict_def.op):
                if not op.type.startswith("Int8"):
                    continue
                if has_weights(op):
                    weights_scale = quantize_weights(ws, op, init_def)
                    if has_bias(op):
                        _, pre_pos = self.algo.get_predecessor_op(0, i, predict_def)
                        assert pre_pos is not None
                        input_data_type = output_data_type[pre_pos]
                        quantize_bias(ws, op, init_def, input_data_type, weights_scale)
            return init_def

        def organize_external_input(ws, predict_def, init_def):
            kTypeNameMapper = {
                np.dtype('float32') : "GivenTensorFill",
                np.dtype('int32')   : "GivenTensorIntFill",
                np.dtype('int64')   : "GivenTensorInt64Fill",
                np.dtype('uint8')   : "GivenTensorStringFill",
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
                if in_data.dtype == np.dtype('uint8'):
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

        self.algo.gather_max(predict_quantized)

        self.algo.update_status()

        update_op_type(predict_quantized)

        insert_dequantize(predict_quantized)

        insert_quantize(predict_quantized)

        refine_module_outputs(predict_quantized)

        # DO NOT change the operator order of the module after below line
        output_data_type = predict_output_data_type(predict_quantized)

        add_output_scale(predict_quantized, output_data_type)

        add_storage_order(predict_quantized)

        init_quantized = gen_quantized_init_def(ws, predict_quantized, output_data_type)

        self.algo.remove_max(predict_quantized)

        for op in predict_quantized.op:
            if op.type.startswith("Int8"):
                op.engine = str("DNNLOWP")
            self.algo.remove_arg(op, "fusion_type")
            op.device_option.CopyFrom(caffe2_pb2.DeviceOption())

        organize_external_input(ws, predict_quantized, init_quantized)

        return predict_quantized, init_quantized
