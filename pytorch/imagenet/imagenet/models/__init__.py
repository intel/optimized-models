"""parse model def"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

class Models:
    """def model class"""
    model_def = {}
    model_def_file = "__model_def"

    def __init__(self):
        if len(self.model_def) == 0:
            self._ParseAllModels()

    def _ParseAllModels(self):
        path = os.path.dirname(__file__)
        dirs = os.listdir(path)
        for d in dirs:
            dir_path = os.path.join(path, d)
            full_path = os.path.join(dir_path, self.model_def_file)
            if os.path.isfile(full_path):
                with open(full_path, 'r') as mf:
                    lines = [line.rstrip('\n') for line in mf.readlines()]
                    self._LoadModelDef(lines, dir_path)

    def _LoadModelDef(self, defs, path):
        model_name = defs[defs.index("[Model Name]") + 1].lower()
        # model type in lowercase, e.g. "caffe legacy", "normal", "prototext"
        model_type = defs[defs.index("[Model Type]") + 1].lower()
        # output type in lowercase, e.g. "possibility", "segmentation", "post image"
        output_type = defs[defs.index("[Output Type]") + 1].lower()
        groups = defs[defs.index("[Groups]") + 1]
        layers = defs[defs.index("[Layers]") + 1]
        width_per_group = defs[defs.index("[Width_per_group]") + 1]
        float_Model = defs[defs.index("[Float Model]") + 1]
        quantized_Model = defs[defs.index("[Quantized Model]") + 1]
        scripted_Quantized_Model = None
        if "[Scripted Quantized Model]" in defs:
            scripted_Quantized_Model = defs[defs.index("[Scripted Quantized Model]") + 1]
        quantized_Model_State_Dict = None
        if "[Quantized Model State Dict]" in defs:
            quantized_Model_State_Dict = defs[defs.index("[Quantized Model State Dict]") + 1]
        onnx_model = None
        if "[Onnx Model]" in defs:
            onnx_model = path, defs[defs.index("[Onnx Model]") + 1]
        crop_size = defs[defs.index("[Crop Size]") + 1]
        image_mean = defs[defs.index("[Image Mean]") + 1]
        scale = 1
        if "[Scale]" in defs:
            scale = defs[defs.index("[Scale]") +1]
        rescale_size = 256
        if "[ReScale Size]" in defs:
            rescale_size = defs[defs.index("[ReScale Size]") +1]
        if len(image_mean) > 0:
            image_mean = os.path.join(path, image_mean)
        else:
            image_mean = None
        allow_device_override = True
        need_normalize = False
        if "[Allow Device Override]" in defs:
            allow_device_override = defs[defs.index("[Allow Device Override]") +1].lower() in ('yes', 'true', 't', '1')
        if "[Need Normalize]" in defs:
            need_normalize = defs[defs.index("[Need Normalize]") +1].lower() in ('yes', 'true', 't', '1')
        color_format = None
        if "[Color Format]" in defs:
            color_format = defs[defs.index("[Color Format]") + 1]
        if "[Model Source]" in defs:
            model_source = defs[defs.index("[Model Source]") + 1]

        if model_name in self.model_def:
            logging.warning("Already has model: {}. Ignored!"
                            .format(model_name))
        else:
            self.model_def[model_name] = {
                "model_name" : model_name,
                "model_dir" : path,
                "model_type" : model_type,
                "output_type" : output_type,
                "groups" : groups,
                "layers" : layers,
                "width_per_group" : width_per_group,
                "float_Model" : float_Model,
                "quantized_Model" : quantized_Model,
                "scripted_Quantized_Model" : scripted_Quantized_Model,
                "quantized_Model_State_Dict" : quantized_Model_State_Dict,
                "onnx_model" : onnx_model,
                "crop_size" : crop_size,
                "image_mean" : image_mean,
                "scale" : scale,
                "rescale_size" : rescale_size,
                "allow_device_override": allow_device_override,
                "need_normalize" : need_normalize,
                "color_format" : color_format,
                "model_source" : model_source
            }


def ShowModels():
    models = Models()
    logging.critical("All supported models for inference:\n{}"
                     .format([str(s) for s in models.model_def]))

def IsSupported(model):
    models = Models()
    return (model.lower() in models.model_def)

def GetModelInfo(model):
    models = Models()
    return models.model_def[model.lower()]
