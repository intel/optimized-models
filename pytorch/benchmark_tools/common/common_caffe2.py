"""common functions for inference and training"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import threading
import itertools
import math
import copy
from collections import defaultdict
import numpy as np
import six
import cv2
import onnx


class ImageProc(threading.Thread):
    """handle dataset preprocess with thread"""
    threads = []
    imgs_one_thread = 8

    def __init__(self, img_paths, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format):
        super(ImageProc, self).__init__()
        self.started = False
        self.reset(img_paths, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format)

    def reset(self, img_paths, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format):
        self.img_paths = img_paths
        self.crop_size = crop_size
        self.rescale_size = rescale_size
        self.mean = mean
        self.imgs = None
        self.forchw = forchw
        self.scale = scale
        self.need_normalize = need_normalize
        self.color_format = color_format

    def run(self):
        self.imgs = ImageProc.PreprocessImages(
            self.img_paths, self.crop_size, self.rescale_size,
            self.mean, self.scale, self.forchw,
            self.need_normalize, self.color_format)

    def kick_off(self):
        if self.started:
            self.run()
        else:
            self.start()
            self.started = True

    @property
    def result(self):
        return self.imgs

    @staticmethod
    def PopThread(img_paths, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format):
        thread = None
        if len(ImageProc.threads) > 0:
            thread = ImageProc.threads.pop()
            thread.reset(img_paths, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format)
        else:
            thread = ImageProc(img_paths, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format)
        return thread

    @staticmethod
    def PushThreads(new_threads):
        ImageProc.threads.extend(new_threads)

    @staticmethod
    def GenSequSplit(iterable, size):
        it = iter(iterable)
        split = list(itertools.islice(it, size))
        while split:
            yield split
            split = list(itertools.islice(it, size))

    @staticmethod
    def ShowImage(img, title):
        # Disabled
        return
        #from matplotlib import pyplot
        #pyplot.figure()
        #pyplot.imshow(img)
        #pyplot.axis('on')
        #pyplot.title(title)

    @staticmethod
    def ShowImageInChannels(img):
        # Disabled
        return
        #from matplotlib import pyplot
        #pyplot.figure()
        #for i in range(3):
            # For some reason, pyplot subplot follows Matlab's indexing
            # convention (starting with 1). Well, we'll just follow it...
            #pyplot.subplot(1, 3, i + 1)
            #pyplot.imshow(img[i])
            #pyplot.axis('off')
            #pyplot.title("RGB channel {}".format(i + 1))

    @staticmethod
    def CropCenter(img, cropx, cropy):
        """center crop the image in dataset"""
        y, x, _ = img.shape
        #startx = x // 2 - (cropx // 2)
        #starty = y // 2 - (cropy // 2)
        startx = int(math.floor(x * 0.5 - (cropx * 0.5)))
        starty = int(math.floor(y * 0.5 - (cropy * 0.5)))

        imgCropped = img[starty : starty + cropy, startx : startx + cropx]
        ImageProc.ShowImage(imgCropped, "Cropped image")
        logging.info("After cropped: {}".format(imgCropped.shape))
        return imgCropped

    @staticmethod
    def Rescale(img, rescale_size):
        """rescale the image with the given size"""
        #if input_height == 299:
        #    input_height = 320
        #    input_width = 320

        #cv2_interpol = cv2.INTER_AREA
        #cv2_interpol = cv2.INTER_CUBIC
        cv2_interpol = cv2.INTER_LINEAR
        logging.info("Original image shape: {} "
                     "and remember it should be in H, W, C!"
                     .format(str(img.shape)))
        logging.info("Model's input shape is {0} x {0}"
                     .format(rescale_size))
        #xaspect = input_width / input_height
        aspect = img.shape[1] / float(img.shape[0])
        logging.info("Orginal aspect ratio: {}".format(str(aspect)))
        if aspect >= 1:
            # landscape orientation - wide image
            res = int(rescale_size * aspect)
            imgScaled = cv2.resize(img, dsize=(res, rescale_size), interpolation=cv2_interpol)
        elif aspect < 1:
            # portrait orientation - tall image
            res = int(rescale_size / aspect)
            #imgScaled = cv2.resize(img, dsize=(input_height, res), interpolation=cv2_interpol)
            imgScaled = cv2.resize(img, dsize=(rescale_size, res), interpolation=cv2_interpol)
        ImageProc.ShowImage(imgScaled, "Rescaled image")
        logging.info("After rescaled in HWC: {}".format(str(imgScaled.shape)))
        return imgScaled

    @staticmethod
    def SaveImage(image, filename):
        """save image"""
        if image.shape[0] != 1:
            logging.error("the shape[0] of the image is not 1")
            return
        img = np.squeeze(image)
        # switch to HWC
        img = img.swapaxes(0, 1).swapaxes(1, 2)
        # switch to RGB
        img = img[:, :, (2, 1, 0)]
        from matplotlib import pyplot
        pyplot.imsave(filename, img)

    @staticmethod
    def PreprocessSingleImage(image_path, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format):
        """preprocess single image"""
        #This is used for cv2.imread
        img = cv2.imread(image_path)
        img = img.astype(np.float32)
        oshape = copy.deepcopy(img.shape)
        img = ImageProc.Rescale(img, rescale_size)
        img = ImageProc.CropCenter(img, crop_size, crop_size)
        if forchw == 1:
        # switch to CHW
            img = img.swapaxes(1, 2).swapaxes(0, 1)
            # switch to RGB
            if color_format == 'RGB':
                img = img[(2, 1, 0), :, :]
        else:
            nimg = np.ndarray((crop_size, crop_size, 4), dtype=float)
            nimg.fill(0)
            for i, img_i in enumerate(img):
                for j, img_j in enumerate(img_i):
                    for k, img_k in enumerate(img_j):
                        nimg[i][j][k] = img_k
            img = nimg
        #logging.info("After switch to bgra {}, type ={}".format(img, type(img[0][0][0])))
        #if need normalize
        if need_normalize:
            img = img/255
        logging.info("scale is {}".format(scale))
        logging.info("mean is {}".format(mean))
        logging.info("image is {}".format(img))
        logging.info("image shape is {}, mean shape is {}".format(img.shape, mean.shape))
        if len(scale) == 1:
            img = (img - mean) * float(scale[0])
        elif len(scale) > 1:
            img[0, :, :] = (img[0, :, :] - mean[0, :, :])*float(scale[0])
            img[1, :, :] = (img[1, :, :] - mean[1, :, :])*float(scale[1])
            img[2, :, :] = (img[2, :, :] - mean[2, :, :])*float(scale[2])
            logging.info("after img is {}".format(img))
        else:
            logging.error("scale = {} is invalid".format(scale))
            exit()

        # add batch size
        if forchw == 1:
            img = img[np.newaxis, :, :, :].astype(np.float32)
        else:
            img = img[np.newaxis, :, :, :].astype(np.uint8)
        ImageProc.ShowImageInChannels(img)
        logging.info("After Preprocessing in NCHW: {}".format(img.shape))
        return img, oshape

    @staticmethod
    def PreprocessImages(img_paths, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format):
        imgs = []
        for i in img_paths:
            img, oshape = ImageProc.PreprocessSingleImage(
                i, crop_size, rescale_size, mean, scale, forchw, need_normalize, color_format)
            imgs.append(img)
        logging.info("oshape={} ".format(oshape))
        return np.concatenate(imgs, 0), oshape

    @staticmethod
    def PreprocessImagesByThreading(img_paths, crop_size, rescale_size,
                                    mean, scale, forchw, need_normalize, color_format):
        """preprocess image using threading, not used now"""
        imgs, oshape = ImageProc.PreprocessImages(
            img_paths, crop_size, rescale_size,
            mean, scale, forchw, need_normalize, color_format)
        if len(img_paths) <= (ImageProc.imgs_one_thread * 2):
            #logging.info("imgs = {0}".format(imgs))
            return imgs, oshape
        cur_thds = []
        imgs = []
        for img_split in ImageProc.GenSequSplit(
                img_paths, ImageProc.imgs_one_thread):
            if img_split is None:
                logging.error("Failed to split image sequence")
                return None
            cur_thds.append(ImageProc.PopThread(
                                img_split, crop_size, rescale_size, mean,
                                scale, forchw, need_normalize, color_format))
        if len(cur_thds) <= 0:
            logging.error("Failed to create threads in image processing")
            return None
        for t in cur_thds:
            t.kick_off()
        while any([t.is_alive() for t in cur_thds]):
            continue
        timg = []
        for t in cur_thds:
            ximg, _ = t.result
            timg.append(ximg)
        imgs = np.concatenate(timg, 0)
        #logging.info("imgs = {0}".format(imgs))
        logging.info("imgs.shape = {0}".format(imgs.shape))
        # imgs = np.concatenate([t.result for t in cur_thds], 0)
        # Reuse python threads leads to bad performance...
        # ImageProc.PushThreads(cur_thds)
        return imgs, oshape

    @staticmethod
    def BatchImages(images_path, batch_size, iterations):
        """batch the dataset"""
        bs = batch_size
        images = []
        image = []
        fnames = []
        fname = []
        it = iterations
        for root, _, files in os.walk(images_path):
            if it == 0:
                break
            for fn in files:
                fp = os.path.join(root, fn)
                bs -= 1
                image.append(fp)
                fname.append(fn)
                if bs == 0:
                    images.append(image)
                    fnames.append(fname)
                    image = []
                    fname = []
                    bs = batch_size
                    it -= 1
                    if it == 0:
                        break
        if len(image) > 0:
            images.append(image)
            fnames.append(fname)
        return (images, fnames)


def ParsePossOutputs(outputs):
    """parse the outputs"""
    total = 0
    parsed_outputs = []
    # logging.info("The output0 is {}".format(outputs[0])
    for i, output in enumerate(outputs):
        for j, o in enumerate(np.split(output, output.shape[0], 0)):
            total += 1
            o = np.squeeze(o)
            index = []
            score = []
            z = 0
            while z < 5:
                z += 1
                index.append(np.argmax(o))
                score.append(o[np.argmax(o)])
                o[np.argmax(o)] = -1
            logging.info("The index is {}".format(index))
            logging.info("The score is {}".format(score))
            # index = 0
            # highest = 0
            # for k, prob in enumerate(o):
            #    if prob > highest:
            #        highest = prob
            #        index = k
            parsed_outputs.append([index, score, (i, j)])
    return (parsed_outputs, total)


def ParsePossResults(results, labels, validation, fnames):
    """parse the result to generate reports"""
    summary = []
    for result in results:
        index = result[0]
        highest = result[1][0]
        file_pos = result[2]
        fname = fnames[file_pos[0]][file_pos[1]]
        if labels:
            if index in labels:
                logging.info("The model infers that the image contains"
                             " {0} with a {1:.5%} probability."
                             .format(labels[index], highest))
                summary.append((fname, labels[index], index, highest))
            else:
                logging.info("No result is found for result index {}!"
                             .format(index))
                summary.append((fname, "UNKNOWN", index, highest))
        elif validation:
            if fname in validation:
                if validation[fname] == index[0]:
                    logging.info("Validation passed for file {0} index[0]"
                                 " {1} with a {2:.5%} probability."
                                 .format(fname, index[0], highest))
                    summary.append((fname, "Pass", index[0], index[0], highest))
                elif validation[fname] in index:
                    logging.info("Validation partially passed for file {0} index[0]"
                                 " {1} with a {2:.5%} top1 probability."
                                 .format(fname, validation[fname], highest))
                    summary.append((fname, "Top5Pass", validation[fname], index[0], highest))
                else:
                    logging.info("Failed in validation for file {0} index"
                                 " {1}. Should be {2}."
                                 .format(fname, index[0], validation[fname]))
                    summary.append(
                        (fname, "Fail", index[0], validation[fname], highest))
            else:
                logging.error("Can NOT find the file {} in validation!"
                              .format(fname))
        else:
            logging.error("No labels and validation is set!")
            return None
    return summary


def LoadLabels(label_file):
    """load labels from file"""
    if not os.path.isfile(label_file):
        logging.error("Can not find lable file {}.".format(label_file))
        return None
    labels = {}
    with open(label_file) as l:
        label_lines = [line.rstrip('\n') for line in l.readlines()]
    for line in label_lines:
        result, code = line.partition(" ")[::2]
        if code and result:
            result = result.strip()
            result = result[result.index("/")+1:]
            if result in labels:
                logging.warning("Repeated name {0} for code {1}in label file. Ignored!"
                                .format(result, code))
            else:
                labels[result] = int(code.strip())
    return labels


def LoadValidation(validation_file):
    """load validation file"""
    if not os.path.isfile(validation_file):
        logging.error("Can not find validation file {}."
                      .format(validation_file))
        return None
    validation = {}
    with open(validation_file) as v:
        validation_lines = [line.rstrip('\n') for line in v.readlines()]
    for line in validation_lines:
        name, code = line.partition(" ")[::2]
        if name and code:
            name = name.strip()
            if name in validation:
                logging.warning("Repeated name {0} for code {1} in"
                                " validation file. Ignored!"
                                .format(name, code))
            else:
                validation[name] = int(code.strip())
    return validation


def SaveOutput(args, summary, accuracy, top5accuracy, total_time, total_image, img_time,
               model_loading_time):
    """save output"""
    fname = args.output_file
    if not fname:
        return
    if fname[-4:] != ".csv":
        fname += ".csv"
    with open(fname, "w") as o:
        imgs_per_sec = total_image / total_time
        info_str = "\nAccuracy,{:.5%}".format(accuracy) if accuracy else ""
        info_str1 = "\nTop5Accuracy,{:.5%}".format(top5accuracy) if top5accuracy else ""
        if accuracy and summary:
            o.write("#,File,Result,Output_Index,Correct_Index,Probability\n")
            for i, r in enumerate(summary):
                o.write("{0},{1},{2},{3},{4},{5:.5%}\n"
                        .format(i + 1, r[0], r[1], r[2], r[3], r[4]))
        elif summary:
            o.write("#,File,Result,Index,Probability\n")
            for i, r in enumerate(summary):
                o.write("{0},{1},\"{2}\",{3},{4:.5%}\n"
                        .format(i + 1, r[0], r[1].strip("\'\", "), r[2], r[3]))
        o.write("Images per second,{0:.10f}\nTotal computing time,{1:.10f}"
                " seconds\nTotal image processing time,{2:.10f} seconds\n"
                "Total model loading time,{3:.10f} seconds\n"
                "Total images,{4}{5}{6}\n{7}"
                .format(imgs_per_sec, total_time, img_time,
                        model_loading_time, total_image, info_str, info_str1, vars(args)))
    logging.warning("Saved the results to {}".format(fname))


def ParsePostOutputs(outputs):
    """parse outputs"""
    total = 0
    parsed_outputs = []
    for i, output in enumerate(outputs):
        for j, o in enumerate(np.split(output, output.shape[0], 0)):
            total += 1
            parsed_outputs.append([o, (i, j)])
    return (parsed_outputs, total)


def SavePostImages(results, path, fnames):
    """save post images"""
    if not os.path.isdir(path):
        return
    for res in results:
        file_pos = res[1]
        fname = fnames[file_pos[0]][file_pos[1]]
        ImageProc.SaveImage(res[0], path + "/" + fname)


def FetchArrayByName(init_def, name):
    """fetch array by name"""
    for index, op in enumerate(init_def.op):
        if op.output[0] != name:
            continue
        if op.type != "GivenTensorFill":
            logging.error("The array {} is not contained"
                          " by GivenTensorFill".format(name))
            return None
        array = [index, None, None]
        for arg in op.arg:
            if arg.name == "shape" and len(arg.ints) > 0:
                if array[1] is not None:
                    logging.error("Duplicated shape definition in array")
                    return None
                array[1] = list(arg.ints)
            elif arg.name == "values" and len(arg.floats) > 0:
                if array[2] is not None:
                    logging.error("Duplicated value definition in array")
                    return None
                array[2] = np.array(list(arg.floats))
        array[2] = array[2].reshape(array[1])
        return array


def FeedArrayByName(init_def, name, array):
    """feed array by name"""
    for op in init_def.op:
        if op.output[0] != name:
            continue
        arg_index = [None, None]
        for i, arg in enumerate(op.arg):
            if arg.name == "shape":
                arg_index[0] = i
            elif arg.name == "values":
                arg_index[1] = i
        if not all(i is not None for i in arg_index):
            logging.error("Incomplete arguments in init_def")
            return False
        if list(op.arg[arg_index[0]].ints) != list(array.shape):
            logging.error("Unmatch shape in init_def")
            return False
        if len(op.arg[arg_index[1]].floats) != array.size:
            logging.error("Unmatch size in init_def")
            return False
        values = array.flatten()
        for index in range(array.size):
            op.arg[arg_index[1]].floats[index] = values[index]
        return True
    return False


def FindInputName(predict_def, index, name):
    """find input name from predict def"""
    for op in predict_def.op[index:]:
        for inp in op.input:
            if inp == name:
                return True
    return False


def UpdateInputName(predict_def, index, from_name, to_name):
    """update inputname in the whole net"""
    if from_name == to_name:
        return
    for op in predict_def.op[index:]:
        for i, inp in enumerate(op.input):
            if inp == from_name:
                for j, outp in enumerate(op.output):
                    if outp == inp:
                        op.output[j] = to_name
                op.input[i] = to_name


def UpdateDeviceOption(dev_opt, net_def):
    """update device options in net_def"""
    # net_def.device_option.CopyFrom(dev_opt)
    # gpufallbackop=['GenerateProposals', 'BoxWithNMSLimit', 'BBoxTransform',
    #     'PackedInt8BGRANHWCToNCHWCStylizerPreprocess', 'BRGNCHWCToPackedInt8BGRAStylizerDeprocess']
    gpufallbackop = ['GenerateProposals', 'BoxWithNMSLimit', 'BBoxTransform']
    # gpufallbackop=[]
    ideepfallbackop = []
    from caffe2.proto import caffe2_pb2
    for eop in net_def.op:
        if (eop.type in gpufallbackop and dev_opt.device_type == caffe2_pb2.CUDA) or (
                eop.type in ideepfallbackop and dev_opt.device_type == caffe2_pb2.IDEEP):
            eop.device_option.device_type = caffe2_pb2.CPU
        elif (
                eop.device_option and
                eop.device_option.device_type != dev_opt.device_type
        ):
            eop.device_option.device_type = dev_opt.device_type


def FillZeroParamsWithOne(net_def):
    """fill zero parameter"""
    for eop in net_def.op:
        if eop.output[0] == 'im_info':
            for earg in eop.arg:
                if earg.name == 'shape':
                    for i in range(len(earg.ints)):
                        if earg.ints[i] == 0:
                            earg.ints[i] = 1
                            logging.warning("find op {0} shape has value 0, replace it, original blob is {1}"
                                            .format(eop.output[0], eop))
                            eop.type = 'ConstantFill'
        if eop.output[0] == 'data':
            for earg in eop.arg:
                if earg.name == 'shape':
                    for i in range(len(earg.ints)):
                        if earg.ints[i] == 0:
                            earg.ints[i] = 1
                            logging.warning("find op {0} shape has value 0, replace it, original blob is {1}"
                                            .format(eop.output[0], eop))
                            eop.type = 'ConstantFill'

    # index=-1
    # for i, earg in enumerate(eop.arg):
    #    if earg.name == 'values':
    #        index = i
    #        break
    # if index != -1:
    #    del eop.arg[index]


def UpdateImgInfo(shape, net_def, predict_def, crop_size):
    """update image info specifically for fastrcnn"""
    im_info_name = 'NA'
    for eop in predict_def.op:
        if eop.type == 'GenerateProposals':
            im_info_name = eop.input[2]
            break
    if im_info_name != 'NA':
        for ep in net_def.op:
            if ep.output[0] == im_info_name:
                ep.type = 'GivenTensorFill'
                im_scale = float(0)
                for earg in ep.arg:
                    if earg.name == 'shape':
                        # for i in range(len(earg.ints)):
                        earg.ints[0] = 1
                        im_size_min = np.min(shape[0:2])
                        im_size_max = np.max(shape[0:2])
                        im_scale = float(crop_size) / float(im_size_min)
                        # Prevent the biggest axis from being more than MAX_SIZE
                        if np.round(im_scale * im_size_max) > 1000:
                            im_scale = float(1000) / float(im_size_max)
                        earg.ints.extend([3])
                        break
                index = -1
                for i, earg in enumerate(ep.arg):
                    if earg.name == 'value':
                        index = i
                        break
                if index != -1:
                    del ep.arg[index]
                index = -1
                for i, earg in enumerate(ep.arg):
                    if earg.name == 'values':
                        index = i
                        break
                if index != -1:
                    del ep.arg[index]
                from caffe2.proto import caffe2_pb2
                narg = caffe2_pb2.Argument()
                narg.name = 'values'
                narg.floats.extend([crop_size, crop_size, im_scale])
                ep.arg.extend([narg])
                logging.warning("update im_info to {}".format(ep))


def CreateIMBlob(shape, predict_def, crop_size):
    """generate im_info blob"""
    im_info_name = 'NA'
    for eop in predict_def.op:
        if eop.type == 'GenerateProposals':
            im_info_name = eop.input[2]
            break
    im_size_min = np.min(shape[0:2])
    im_size_max = np.max(shape[0:2])
    im_scale = float(crop_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > 1000:
        im_scale = float(1000) / float(im_size_max)
    blob = np.ndarray((1, 3), buffer=np.array([crop_size, crop_size, im_scale], dtype=np.float32), dtype=np.float32)
    logging.warning("im_info name is {0}, and blob is {1} ".format(im_info_name, blob))
    return im_info_name, blob


def assert_allclose(x, y, atol=1e-5, rtol=1e-4, verbose=True):
    """Asserts if some corresponding element of x and y differs too much.

    This function can handle both CPU and GPU arrays simultaneously.

    Args:
        x: Left-hand-side array.
        y: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.

    """
    try:
        np.testing.assert_allclose(
            x, y, atol=atol, rtol=rtol, verbose=verbose)
        return True
    except AssertionError as e:
        f = six.StringIO()
        f.write(str(e) + '\n\n')
        f.write(
            'assert_allclose failed: \n' +
            '  shape: {} {}\n'.format(x.shape, y.shape) +
            '  dtype: {} {}\n'.format(x.dtype, y.dtype))
        if x.shape == y.shape:
            xx = x if x.ndim != 0 else x.reshape((1,))
            yy = y if y.ndim != 0 else y.reshape((1,))
            err = np.abs(xx - yy)
            i = np.unravel_index(np.argmax(err), err.shape)
            f.write(
                '  i: {}\n'.format(i) +
                '  x[i]: {}\n'.format(xx[i]) +
                '  y[i]: {}\n'.format(yy[i]) +
                '  err[i]: {}\n'.format(err[i]))
        opts = np.get_printoptions()
        try:
            np.set_printoptions(threshold=10000)
            f.write('x: ' + np.array2string(x, prefix='x: ') + '\n')
            f.write('y: ' + np.array2string(y, prefix='y: ') + '\n')
        finally:
            np.set_printoptions(**opts)
            #raise AssertionError(f.getvalue())
            logging.warning(f.getvalue())
            return False

def assert_compare(x, y, atol=1e-5, method='ALL'):
    """method can be MSE, MAE and RMSE"""
    mae = 0
    mse = 0
    rmse = 0
    result = 0
    if method == 'MAE':
        mae = np.abs(x-y).mean()
        result = mae
    elif method == 'RMSE':
        rmse = np.sqrt(np.square(x - y).mean())
        result = rmse
        #result=np.sqrt(((x - y) ** 2).mean())
    elif method == 'MSE':
        mse = np.square(x - y).mean()
        result = mse
        #result=((x - y) ** 2).mean()
    else:
        mae = np.abs(x-y).mean()
        rmse = np.sqrt(np.square(x - y).mean())
        mse = np.square(x - y).mean()

    if result > atol or (method == 'ALL' and (mae > atol or rmse > atol or mse > atol)):
        f = six.StringIO()
        f.write(
            'assert_compare failed: \n' +
            '  atol: {} \n'.format(atol) +
            '  method: {}\n'.format(method) +
            '  MAE: {}\n'.format(mae) +
            '  MSE: {}\n'.format(mse) +
            '  RMSE: {}\n'.format(rmse) +
            '  shape: {} {}\n'.format(x.shape, y.shape) +
            '  dtype: {} {}\n'.format(x.dtype, y.dtype))
        if x.shape == y.shape:
            xx = x if x.ndim != 0 else x.reshape((1,))
            yy = y if y.ndim != 0 else y.reshape((1,))
            err = np.abs(xx - yy)
            i = np.unravel_index(np.argmax(err), err.shape)
            f.write(
                '  i: {}\n'.format(i) +
                '  x[i]: {}\n'.format(xx[i]) +
                '  y[i]: {}\n'.format(yy[i]) +
                '  err[i]: {}\n'.format(err[i]))
        opts = np.get_printoptions()
        try:
            np.set_printoptions(threshold=10000)
            f.write('x: ' + np.array2string(x, prefix='x: ') + '\n')
            f.write('y: ' + np.array2string(y, prefix='y: ') + '\n')
        finally:
            np.set_printoptions(**opts)
            logging.warning(f.getvalue())
            return False
    else:
        return True

def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        logging.error("the array shape should be (1,4)")
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def prepare_and_compute_map_data(outputs, fnames, path):
    """calculate ap for ssd"""
    return 1


def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function evaluates predicted bounding boxes obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **ap** (*numpy.ndarray*): An array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **map** (*float*): The average of Average Precisions over classes.

    """

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.

    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value..

    Returns:
        tuple of two lists:
        This function returns two lists: :obj:`prec` and :obj:`rec`.

        * :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
            set to :obj:`None`.
        * :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`gt_labels`, :obj:`rec[l]` is \
            set to :obj:`None`.

    """

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
        six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):
        print("the bbox is ", pred_bbox)
        print("the gt bbox is ", gt_bbox)
        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)
        #print(pred_label)
        #print(gt_label)
        #print(gt_bbox)
        #print(pred_bbox)
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
         #   print(l)
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.

    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def AddTensor(init_net, name, blob):
    ''' Create an operator to store the tensor 'blob',
        run the operator to put the blob to workspace.
        uint8 is stored as an array of string with one element.
    '''
    from caffe2.python import core, utils
    kTypeNameMapper = {
        np.dtype('float32'): "GivenTensorFill",
        np.dtype('int32'): "GivenTensorIntFill",
        np.dtype('int64'): "GivenTensorInt64Fill",
        np.dtype('uint8'): "GivenTensorStringFill",
    }

    shape = blob.shape
    values = blob
    # pass array of uint8 as a string to save storage
    # storing uint8_t has a large overhead for now
    if blob.dtype == np.dtype('uint8'):
        shape = [1]
        values = [str(blob.data)]

    op = core.CreateOperator(
        kTypeNameMapper[blob.dtype],
        [], [name],
        arg=[
            utils.MakeArgument("shape", shape),
            utils.MakeArgument("values", values),
        ]
    )
    init_net.op.extend([op])


def AddTensors(workspace, init_net, blob_names):
    """add tensors"""
    for blob_name in blob_names:
        blob = workspace.FetchBlob(str(blob_name))
        AddTensor(init_net, blob_name, blob)


def AllBlobNamesInInit(init_net):
    """collect all blob names in weight file"""
    from caffe2.proto import caffe2_pb2
    names = []
    if init_net is None or not isinstance(init_net, caffe2_pb2.NetDef):
        return names
    for op in init_net.op:
        assert(len(op.output) == 1)
        names.append(str(op.output[0]))
    return names


def ExportModel(workspace, net, blobs):
    """Returns init_net and predict_net suitable for writing to disk
       and loading into a Predictor"""

    from caffe2.proto import caffe2_pb2
    from caffe2.python import core, workspace, utils

    proto = net if isinstance(net, caffe2_pb2.NetDef) else net.Proto()
    predict_net = caffe2_pb2.NetDef()
    predict_net.CopyFrom(proto)
    ssa, blob_versions = core.get_ssa(net)

    # Populate the init_net.
    init_net = caffe2_pb2.NetDef()

    inputs = []
    for versioned_inputs, _ in ssa:
        inputs += [name for name, _ in versioned_inputs]

    input_blobs = [blob_name for blob_name, version in
                   blob_versions.items()
                   if version == 0 and blob_name not in blobs]
    # Blobs that are never used as an input to another layer,
    # i.e. strictly output blobs.
    output_blobs = [blob_name for blob_name, version in
                    blob_versions.items()
                    if version != 0 and blob_name not in inputs]

    AddTensors(workspace, init_net, blobs)
    # We have to make sure the blob exists in the namespace
    # and we can do so with fake data. (Which is immediately overwritten
    # by any typical usage)
    for blob_name in input_blobs:
        init_net.op.extend(
            [
                core.CreateOperator(
                    "GivenTensorFill", [], [blob_name],
                    arg=[
                        utils.MakeArgument("shape", [1, 1]),
                        utils.MakeArgument("values", [0.0])
                    ]
                )
            ]
        )

    for op in init_net.op:
        if op.type != 'ConstantFill':
            continue
        for arg in op.arg:
            if arg.name != 'shape':
                continue
            if len(arg.ints) != 0:
                continue
            arg.ints.extend([1])

    # Now we make input/output_blobs line up with what Predictor expects.
    del predict_net.external_input[:]
    predict_net.external_input.extend(input_blobs)
    # For populating weights
    external_input = [e for e in proto.external_input if e not in input_blobs]
    predict_net.external_input.extend(external_input)
    # Ensure the output is also consistent with what we want
    del predict_net.external_output[:]
    predict_net.external_output.extend(output_blobs)

    return init_net, predict_net


def SaveModel(init_file, init_net, predict_file, predict_net):
    """save model in pb"""
    with open(init_file, "wb") as i:
        i.write(init_net.SerializeToString())
    with open(predict_file, "wb") as p:
        p.write(predict_net.SerializeToString())


def SaveModelPtxt(predict_file, predict_net):
    """save model in ptxt"""
    with open(predict_file, "wb") as p:
        p.write(str(predict_net))


def SaveAsOnnxModel(init_net, predict_net, data_shape, onnx_file):
    """save model in onnx format"""
    onnx_model = Caffe2ToOnnx(init_net, predict_net, data_shape)
    with open(onnx_file, "wb") as i:
        i.write(onnx_model.SerializeToString())


def SetOpName(predict_def):
    """set op name"""
    for i, op in enumerate(predict_def.op):
        if len(op.name) == 0:
            op.name = op.type.lower() + str(i)


def MergeScaleBiasInBN(predict_def):
    """
    For models converted from legacy caffe pre-trained, BN is split into 3 ops,
    e.g. SpatialBN + Mul + Add, for BN, shfit and bias, respectively.
    Here, merge all 3 ops into 1 BN op.
    """
    bn_index = -100
    bn_indexes = []
    for i, op in enumerate(predict_def.op):
        if op.type == "SpatialBN":
            bn_index = i + 1
        elif op.type == "Mul":
            bn_index = (i + 1) if (bn_index == i) else -100
        elif op.type == "Add":
            if (bn_index == i):
                bn_indexes.append(i - 2)
            bn_index = -100
    rm_cnt = 0
    for j in bn_indexes:
        index = j - rm_cnt
        bn_op = predict_def.op[index]
        mul_op = predict_def.op[index + 1]
        add_op = predict_def.op[index + 2]
        if (
                bn_op.type != "SpatialBN" or
                mul_op.type != "Mul" or
                add_op.type != "Add"
        ):
            logging.info("Found error in BN compatibility!")
            continue
        scale_index = -1
        if bn_op.output[0] == mul_op.input[0]:
            scale_index = 1
        elif bn_op.output[0] == mul_op.input[1]:
            scale_index = 0
        else:
            logging.info("Fail to find scale in BN compatibility!")
            continue
        bias_index = -1
        if mul_op.output[0] == add_op.input[0]:
            bias_index = 1
        elif mul_op.output[0] == add_op.input[1]:
            bias_index = 0
        else:
            logging.info("Fail to find bias in BN compatibility!")
            continue
        # Set scale and bias blobs as BN inputs
        if len(bn_op.input) != 5:
            logging.warning("Cannot find BN scale and bias inputs")
            continue
        bn_op.input[1] = mul_op.input[scale_index]
        bn_op.input[2] = add_op.input[bias_index]
        # Do NOT allow InPlace in BN
        if bn_op.output[0] == add_op.output[0]:
            bn_op.output[0] = bn_op.output[0] + "_bn_" + str(index)
        UpdateInputName(predict_def, index + 3, add_op.output[0],
                        bn_op.output[0])
        # Delete Mul op
        del predict_def.op[index + 1]
        # Delete Add op
        del predict_def.op[index + 1]
        rm_cnt += 2
    logging.warning("[OPT] Merged {} scale and bias ops into BN ops"
                    .format(rm_cnt // 2))


def ApplyBnInPlace(init_def, predict_def, model_info):
    """do bn inplace optimize"""
    bn_indexes = []
    for i, op in enumerate(predict_def.op):
        if op.type == "SpatialBN":
            bn_indexes.append(i)
    ip_cnt = 0
    for index in bn_indexes:
        bn_op = predict_def.op[index]
        if bn_op.type != "SpatialBN":
            logging.error("Found error in BN InPlace!")
            continue
        if bn_op.input[0] == bn_op.output[0]:
            logging.error("Found error in BN InPlace!")
            continue
        if FindInputName(predict_def, index + 1, bn_op.input[0]):
            logging.error("Found error in BN InPlace!")
            continue
        UpdateInputName(predict_def, index + 1, bn_op.output[0],
                        bn_op.input[0])
        bn_op.output[0] = bn_op.input[0]
        ip_cnt += 1
    logging.warning("[OPT] Enabled {} BN InPlace".format(ip_cnt))


def ApplyBnFolding(init_def, predict_def, model_info):
    """do bn folding optimize"""
    conv_index = -100
    conv_indexes = []
    for i, op in enumerate(predict_def.op):
        if op.type == "Conv":
            conv_index = i + 1
        elif op.type == "SpatialBN":
            if (conv_index == i):
                conv_indexes.append(i - 1)
            conv_index = -100
    rm_cnt = 0
    for j in conv_indexes:
        index = j - rm_cnt
        conv_op = predict_def.op[index]
        bn_op = predict_def.op[index + 1]
        if (
                conv_op.type != "Conv" or
                bn_op.type != "SpatialBN" or
                conv_op.output[0] != bn_op.input[0] or
                len(bn_op.input) != 5 or
                (
                    conv_op.output[0] != bn_op.output[0] and
                    FindInputName(predict_def, index + 2, conv_op.output[0])
                )
        ):
            logging.error("Inputs error in BN folding")
            continue
        if model_info["model_type"] != "prototext":
            conv_w = FetchArrayByName(init_def, conv_op.input[1])
            conv_b = None
            if len(conv_op.input) >= 3:
                conv_b = FetchArrayByName(init_def, conv_op.input[2])
                if conv_b is None:
                    logging.error("Failed to fetch conv bias in BN folding")
                    continue
            bn_scale = FetchArrayByName(init_def, bn_op.input[1])
            bn_bias = FetchArrayByName(init_def, bn_op.input[2])
            bn_mean = FetchArrayByName(init_def, bn_op.input[3])
            bn_var = FetchArrayByName(init_def, bn_op.input[4])
            if not all([conv_w, bn_scale, bn_bias, bn_mean, bn_var]):
                logging.error("Shape error in BN folding")
                continue
            if (
                    conv_w[1][0] != bn_scale[1][0] or
                    bn_scale[1] != bn_bias[1] or
                    bn_scale[1] != bn_mean[1] or
                    bn_scale[1] != bn_var[1] or
                    (
                        conv_b is not None and
                        bn_scale[1] != conv_b[1]
                    )
            ):
                logging.error("Shape unmatch error in BN folding")
                continue
            bn_is_test = None
            bn_epsilon = 1e-5
            # bn_momentum = 0.9
            # Use default value as in caffe model
            bn_momentum = 1.0
            for arg in bn_op.arg:
                if arg.name == "is_test":
                    if arg.i == 1:
                        bn_is_test = arg.i
                    else:
                        logging.error("The BN is not for inference")
                        break
                elif arg.name == "momentum":
                    bn_momentum = arg.f
                elif arg.name == "epsilon":
                    bn_epsilon = arg.f
            if not all([bn_is_test, bn_momentum, bn_epsilon]):
                logging.error("Found error in BN arguments")
                continue
            bn_alpha = (bn_scale[2] / np.sqrt(bn_var[2] * bn_momentum
                                              + bn_epsilon))
            if conv_b is not None:
                conv_bias = bn_alpha * conv_b[2]
                bias_name = conv_op.input[2]
            else:
                conv_bias = np.zeros(bn_bias[2].shape)
                bias_name = bn_op.input[2]
                conv_op.input.append(bias_name)
            conv_bias = conv_bias + (
                bn_bias[2] - (bn_alpha * bn_momentum * bn_mean[2]))
            if not FeedArrayByName(init_def, bias_name, conv_bias):
                logging.error("Failed to feed Conv bias")
                continue
            for i in range(conv_w[2].ndim - 1):
                bn_alpha = bn_alpha[:, np.newaxis]
            conv_weight = bn_alpha * conv_w[2]
            if not FeedArrayByName(init_def, conv_op.input[1], conv_weight):
                logging.error("Failed to feed Conv weight")
                continue
        else:
            pass
        if conv_op.output[0] != bn_op.output[0]:
            UpdateInputName(predict_def, index + 2, bn_op.output[0],
                            conv_op.output[0])
        # Remove BN op
        del predict_def.op[index + 1]
        rm_cnt += 1
    logging.warning("[OPT] Merged {} BN ops into Conv ops by BN folding"
                    .format(rm_cnt))


class FusionType(object):
    UNKNOWN = 0
    CONV_RELU = 1
    CONV_SUM = 2
    CONV_SUM_RELU = 3
    MAX = CONV_SUM_RELU + 1
    ARG_NAME = "fusion_type"
    OP_TYPE = "ConvFusion"


def ApplyFusionConvSum(init_def, predict_def, model_info):
    """fuse conv and sum"""
    conv_index = -100
    conv_indexes = []
    for i, op in enumerate(predict_def.op):
        if op.type == "Conv":
            conv_index = i + 1
            for arg in op.arg:
                if arg.name == "group" and arg.i > 1:
                    conv_index = -100
        elif op.type == "Sum" and len(op.input) == 2:
            if (conv_index == i):
                conv_indexes.append(i - 1)
            conv_index = -100
    from caffe2.proto import caffe2_pb2
    rm_cnt = 0
    for j in conv_indexes:
        index = j - rm_cnt
        conv_op = predict_def.op[index]
        sum_op = predict_def.op[index + 1]
        if (
                conv_op.type != "Conv" or
                sum_op.type != "Sum" or
                (
                    conv_op.output[0] != sum_op.input[0] and
                    conv_op.output[0] != sum_op.input[1]
                )
        ):
            logging.error("Inputs error in Conv Sum fusion")
            continue
        sum_input = (sum_op.input[0]
                     if sum_op.input[1] == conv_op.output[0] else sum_op.input[1])
        if FindInputName(predict_def, index + 2, sum_input):
            logging.error("Inputs error in Conv Sum fusion")
            continue
        if conv_op.output[0] != sum_op.output[0]:
            if FindInputName(predict_def, index + 2, conv_op.output[0]):
                logging.error("Inputs error in Conv Sum fusion")
                continue
        fusion_arg = caffe2_pb2.Argument()
        fusion_arg.name = FusionType.ARG_NAME
        fusion_arg.i = FusionType.CONV_SUM
        conv_op.type = FusionType.OP_TYPE
        conv_op.arg.extend((fusion_arg,))
        conv_op.input.extend((sum_input,))
        conv_op.output[0] = sum_input
        UpdateInputName(predict_def, index + 2, sum_op.output[0], sum_input)
        # Remove Sum op
        del predict_def.op[index + 1]
        rm_cnt += 1
    logging.warning("[OPT] Fused {} Sum ops to Conv ops".format(rm_cnt))


def ApplyFusionConvReLU(init_def, predict_def, model_info):
    """fuse conv and relu"""
    def IsConvOp(op):
        if op.type == "Conv":
            for arg in op.arg:
                if arg.name == "group" and arg.i > 1:
                    return False
            return True
        if op.type == FusionType.OP_TYPE:
            for arg in op.arg:
                if (
                        arg.name == FusionType.ARG_NAME and
                        arg.i == FusionType.CONV_SUM
                ):
                    return True
        return False

    conv_index = -100
    conv_indexes = []
    for i, op in enumerate(predict_def.op):
        if IsConvOp(op):
            conv_index = i + 1
        elif op.type == "Relu":
            if (conv_index == i):
                conv_indexes.append(i - 1)
            conv_index = -100
    from caffe2.proto import caffe2_pb2
    rm_cnt = 0
    for j in conv_indexes:
        index = j - rm_cnt
        conv_op = predict_def.op[index]
        relu_op = predict_def.op[index + 1]
        if (
                (
                    conv_op.type != "Conv" and
                    conv_op.type != FusionType.OP_TYPE
                ) or
                relu_op.type != "Relu" or
                conv_op.output[0] != relu_op.input[0]
        ):
            logging.error("Inputs error in Conv ReLU fusion")
            continue
        if conv_op.output[0] != relu_op.output[0]:
            if FindInputName(predict_def, index + 2, conv_op.output[0]):
                logging.error("Inputs error in Conv ReLU fusion")
                continue
        if conv_op.type == FusionType.OP_TYPE:
            for arg in conv_op.arg:
                if arg.name == FusionType.ARG_NAME:
                    assert arg.i == FusionType.CONV_SUM, "Error in Conv ReLU"
                    arg.i = FusionType.CONV_SUM_RELU
                    break
            UpdateInputName(predict_def, index + 2, relu_op.output[0],
                            conv_op.output[0])
        else:
            fusion_arg = caffe2_pb2.Argument()
            fusion_arg.name = "fusion_type"
            fusion_arg.i = FusionType.CONV_RELU
            conv_op.output[0] = relu_op.output[0]
            conv_op.type = FusionType.OP_TYPE
            conv_op.arg.extend((fusion_arg,))
        # Remove ReLU op
        del predict_def.op[index + 1]
        rm_cnt += 1
    logging.warning("[OPT] Fused {} ReLU ops to Conv ops".format(rm_cnt))


def ApplyRemoveDropout(init_def, predict_def, model_info):
    """remove dropout"""
    dropout_indexes = []
    for i, op in enumerate(predict_def.op):
        if op.type == "Dropout":
            dropout_indexes.append(i)
    rm_cnt = 0
    for j in dropout_indexes:
        index = j - rm_cnt
        dropout_op = predict_def.op[index]
        if dropout_op.input[0] != dropout_op.output[0]:
            UpdateInputName(predict_def, index + 1, dropout_op.output[0],
                            dropout_op.input[0])
        # Remove Dropout op
        del predict_def.op[index]
        rm_cnt += 1
    logging.warning("[OPT] Removed {} dropout ops".format(rm_cnt))


def ApplyInt8Mode(init_def, predict_def, model_info):
    """int8 mode"""
    pass
    # logging.warning("[OPT] Applied Int8 mode for {} ops".format(total))


def ApplyOptimizations(init_def, predict_def, model_info, optimization):
    """do all optimization"""
    if not optimization:
        return
    if ("all" in optimization) or ("bn_folding" in optimization):
        ApplyBnFolding(init_def, predict_def, model_info)
    if ("all" in optimization) or ("bn_inplace" in optimization):
        ApplyBnInPlace(init_def, predict_def, model_info)
    if ("all" in optimization) or ("fusion_conv_sum" in optimization):
        ApplyFusionConvSum(init_def, predict_def, model_info)
    if ("all" in optimization) or ("fusion_conv_relu" in optimization):
        ApplyFusionConvReLU(init_def, predict_def, model_info)
    if ("all" in optimization) or ("remove_dropout" in optimization):
        ApplyRemoveDropout(init_def, predict_def, model_info)
    if ("all" in optimization) or ("int8_mode" in optimization):
        ApplyInt8Mode(init_def, predict_def, model_info)

def Caffe2ToOnnx(init_def, predict_def, data_shape):
    """transfer caffe2 to onnx"""
    from caffe2.proto import caffe2_pb2
    from caffe2.python.onnx import frontend
    from caffe2.python import workspace

    old_ws_name = workspace.CurrentWorkspace()
    workspace.SwitchWorkspace("_onnx_porting_", True)

    data_type = onnx.TensorProto.FLOAT
    value_info = {
        str(predict_def.op[0].input[0]) : (data_type, data_shape)
    }

    device_opts_cpu = caffe2_pb2.DeviceOption()
    device_opts_cpu.device_type = caffe2_pb2.CPU
    UpdateDeviceOption(device_opts_cpu, init_def)
    UpdateDeviceOption(device_opts_cpu, predict_def)

    onnx_model = frontend.caffe2_net_to_onnx_model(
        predict_def,
        init_def,
        value_info
    )

    onnx.checker.check_model(onnx_model)
    workspace.SwitchWorkspace(old_ws_name)
    return onnx_model


def OnnxToCaffe2(model_file):
    """transfer onnx to caffe2"""
    from caffe2.python.onnx import backend
    model = onnx.load(model_file)
    return backend.Caffe2Backend.onnx_graph_to_caffe2_net(model)

def RemoveUselessExternalInput(predict_net):
    """remove useless external input"""
    from caffe2.proto import caffe2_pb2
    if predict_net is None or not isinstance(predict_net, caffe2_pb2.NetDef):
        return
    allInputs = []
    for op in predict_net.op:
        for inp in op.input:
            if inp not in allInputs:
                allInputs.append(inp)

    external_input = [i for i in predict_net.external_input if i in allInputs]
    del predict_net.external_input[:]
    predict_net.external_input.extend(external_input)

def CalMAE(img, predict):
    """calculate mae"""
    errors = np.absolute(img - predict)
    return np.sum(errors) / errors.size

def CalMSE(img, predict):
    """calculate mse"""
    return np.square(np.subtract(img, predict)).mean()

def ConvertText2PB(pbtxt_file, to_save):
    """convert model from txt to pb"""
    with open(pbtxt_file) as t:
        from caffe2.proto import caffe2_pb2
        import google.protobuf.text_format as ptxt
        m = ptxt.Parse(t.read(), caffe2_pb2.NetDef())
        with open(to_save, "w") as s:
            s.write(m.SerializeToString())

def ConvertPB2Text(pb_file, to_save):
    """convert model from pb to txt"""
    with open(pb_file) as p:
        from caffe2.proto import caffe2_pb2
        m = caffe2_pb2.NetDef()
        m.ParseFromString(p.read())
        with open(to_save, "w") as s:
            s.write(str(m))


if __name__ == '__main__':
    logging.critical("Do not run this script independently!")
    exit()
