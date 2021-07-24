from . import my_model_8_relu_kaiming
from . import my_model_8_swish_kaiming
from . import my_model_8_relu_kaiming_DW


def get_MyModel(mode, activation_func, dimension_choice, theta_stride):
    if dimension_choice == '3D':
        out_channel = int(360 / theta_stride)
    else:
        out_channel = 1
    if mode == "Model8" and activation_func == "swish":
        return my_model_8_swish_kaiming.get_model(out_channel)
    elif mode == "Model8" and activation_func == "ReLU":
        return my_model_8_relu_kaiming.get_model(out_channel)
    elif mode == "Model8_DW" and activation_func == "ReLU":
        return my_model_8_relu_kaiming_DW.get_model(out_channel)
