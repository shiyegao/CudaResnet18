import numpy as np
import onnx
from onnx import numpy_helper


nodes = [
    'conv1',
    'layer1.0.conv1', 'layer1.0.conv2',
    'layer1.1.conv1', 'layer1.1.conv2',
    'layer2.0.conv1', 'layer2.0.conv2',
    'layer2.0.downsample.0',
    'layer2.1.conv1', 'layer2.1.conv2',
    'layer3.0.conv1', 'layer3.0.conv2',
    'layer3.0.downsample.0',
    'layer3.1.conv1', 'layer3.1.conv2',
    'layer4.0.conv1', 'layer4.0.conv2',
    'layer4.0.downsample.0',
    'layer4.1.conv1', 'layer4.1.conv2',
    'fc'
]

dic = {
    '193': 'conv1.weight', '194': 'conv1.bias',
    '196': 'layer1.0.conv1.weight', '197': 'layer1.0.conv1.bias',
    '199': 'layer1.0.conv2.weight', '200': 'layer1.0.conv2.bias',
    '202': 'layer1.1.conv1.weight', '203': 'layer1.1.conv1.bias',
    '205': 'layer1.1.conv2.weight', '206': 'layer1.1.conv2.bias',
    '208': 'layer2.0.conv1.weight', '209': 'layer2.0.conv1.bias',
    '211': 'layer2.0.conv2.weight', '212': 'layer2.0.conv2.bias',
    '214': 'layer2.0.downsample.0.weight', '215': 'layer2.0.downsample.0.bias',
    '217': 'layer2.1.conv1.weight', '218': 'layer2.1.conv1.bias',
    '220': 'layer2.1.conv2.weight', '221': 'layer2.1.conv2.bias',
    '223': 'layer3.0.conv1.weight', '224': 'layer3.0.conv1.bias',
    '226': 'layer3.0.conv2.weight', '227': 'layer3.0.conv2.bias',
    '229': 'layer3.0.downsample.0.weight', '230': 'layer3.0.downsample.0.bias',
    '232': 'layer3.1.conv1.weight', '233': 'layer3.1.conv1.bias',
    '235': 'layer3.1.conv2.weight', '236': 'layer3.1.conv2.bias',
    '238': 'layer4.0.conv1.weight', '239': 'layer4.0.conv1.bias',
    '241': 'layer4.0.conv2.weight', '242': 'layer4.0.conv2.bias',
    '244': 'layer4.0.downsample.0.weight', '245': 'layer4.0.downsample.0.bias',
    '247': 'layer4.1.conv1.weight', '248': 'layer4.1.conv1.bias',
    '250': 'layer4.1.conv2.weight', '251': 'layer4.1.conv2.bias',
    'fc.weight': 'fc.weight', 'fc.bias': 'fc.bias'
}

root = './'
onnx_path = root + 'build/resnet18.onnx'  # change to where .onnx is
weights_path = root + 'python/weights/'

model = onnx.load(onnx_path)


def write_npy(root):
    for t in model.graph.initializer:
        weight = numpy_helper.to_array(t)
        dest = root + dic[t.name] + '.npy'
        np.save(dest, weight)
        print('Write', t.name, 'to', dest)


def judge(root):
    for t in model.graph.initializer:
        weight = numpy_helper.to_array(t)
        dest = root + dic[t.name] + '.npy'
        weight_ = np.load(dest)
        print(t.name, (weight_ == weight).all())


write_npy(weights_path)
judge(weights_path)
