import torch
import torch.onnx
#import onnx
#import onnxruntime

import pvnet


model_path = 'models/299.pth'
dummy_input = torch.randn(1, 3, 480, 640)
net = pvnet.Resnet18_8s(18, 2)

torch_model = torch.load(model_path)
net.load_state_dict(torch_model)
#net = net.cuda()

# 设置模型为评估模式
net.eval()

# 转换为onnx模型
torch.onnx.export(net, dummy_input, 'models/pvnet-299.onnx', verbose=False, input_names=['input'], output_names=['output'], opset_version=11)


# check ONNX model with ONNX's API
#onnx_model = onnx.load('models/pvnet-299.onnx')
#onnx.checker.check_model(onnx_model)

# run the model with ONNX Runtime
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# ort_session = onnxruntime.InferenceSession("models/pvnet-299.onnx")
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)
# print(ort_outs)
