import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import collections
from typing import Dict, OrderedDict, List, Union
import pycuda.autoinit

import torch


def load_engine(path):
    logger = trt.Logger(trt.Logger.INFO)
    logger.log(trt.Logger.INFO, f'Loading engine from file {path}')  # Logging at INFO level
    runtime = trt.Runtime(logger)
    with open(path, 'rb') as f:
        _engine = runtime.deserialize_cuda_engine(f.read())
    return _engine


def get_input_tensor_names(_engine: trt.ICudaEngine) -> list[str]:
    input_tensor_names = []
    for binding in _engine:
        if _engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            input_tensor_names.append(binding)
    return input_tensor_names


def get_output_tensor_names(_engine: trt.ICudaEngine) -> list[str]:
    output_tensor_names = []
    for binding in _engine:
        if _engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
            output_tensor_names.append(binding)
    return output_tensor_names


class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        super().__init__()
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name: str, memory: int, size: int, alignment: int) -> int:
        print("[MyOutputAllocator::reallocate_output] TensorName=%s, Memory=%s, Size=%d, Alignment=%d" % (
            tensor_name, memory, size, alignment))
        if tensor_name in self.buffers:
            del self.buffers[tensor_name]

        address = cuda.mem_alloc(size)
        self.buffers[tensor_name] = address
        return int(address)

    def notify_shape(self, tensor_name: str, shape: trt.Dims):
        # print("[MyOutputAllocator::notify_shape] TensorName=%s, Shape=%s" % (tensor_name, shape))
        self.shapes[tensor_name] = tuple(shape)


class ProcessorV3:
    def __init__(self, _engine: trt.ICudaEngine):
        self.engine = _engine
        self.output_allocator = OutputAllocator()
        self.context = _engine.create_execution_context()
        self.input_tensor_names = get_input_tensor_names(_engine)
        self.output_tensor_names = get_output_tensor_names(_engine)

        self.stream = cuda.Stream()

        self.start_event = cuda.Event()
        self.end_event = cuda.Event()

    def get_last_inference_time(self):
        return self.start_event.time_till(self.end_event)

    def infer(self, _inputs: Union[Dict[str, np.ndarray], List[np.ndarray], np.ndarray]) -> OrderedDict[str, np.ndarray]:
        if isinstance(_inputs, np.ndarray):
            _inputs = [_inputs]
        if isinstance(_inputs, dict):
            _inputs = [inp if name in self.input_tensor_names else None for (name, inp) in _inputs.items()]
        if isinstance(_inputs, list):
            for name, arr in zip(self.input_tensor_names, _inputs):
                self.context.set_input_shape(name, arr.shape)

        buffers_host = []
        buffers_device = []
        for name, arr in zip(self.input_tensor_names, _inputs):
            host = cuda.pagelocked_empty(arr.shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            device = cuda.mem_alloc(arr.nbytes)

            host[:] = arr
            cuda.memcpy_htod_async(device, host, self.stream)
            buffers_host.append(host)
            buffers_device.append(device)

        for name, buffer in zip(self.input_tensor_names, buffers_device):
            self.context.set_tensor_address(name, int(buffer))

        for name in self.output_tensor_names:
            self.context.set_tensor_address(name, 0)  # set nullptr
            self.context.set_output_allocator(name, self.output_allocator)

        self.start_event.record(self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        self.end_event.record(self.stream)

        output_buffers = collections.OrderedDict()
        for name in self.output_tensor_names:
            arr = cuda.pagelocked_empty(self.output_allocator.shapes[name],
                                        dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            cuda.memcpy_dtoh_async(arr, self.output_allocator.buffers[name], stream=self.stream)
            output_buffers[name] = arr

        self.stream.synchronize()

        return output_buffers


if __name__ == "__main__":
    model_path = r"D:\pycharm_projects\MachineLearning\transformer\tensorRT\engine.trt"
    engine = load_engine(model_path)

    input_tokens_ids = [1, 57, 16, 6, 4517, 3158, 40, 9039, 14770, 22,
                        112, 13547, 1961, 3403, 22, 7946, 151, 3037, 59, 2857,
                        21, 3871, 375, 5427, 253, 660, 2806, 364, 8053, 351,
                        4, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0]

    true_tokens_ids = [1, 111, 7, 32, 14323, 12, 1944, 296, 42, 142, 2452, 2452,
                       2452, 65, 2782, 10, 8, 2862, 78, 2329, 10, 2694, 10, 2694,
                       286, 1745, 415, 8, 201, 3145, 1491, 10326, 1755, 4, 2]

    inputs = {
        'encoder_input': np.array(input_tokens_ids, dtype=np.int64).reshape(1, 128),
        'decoder_input': np.array([1, 111, 7, 32], dtype=np.int64).reshape(1, -1),
    }

    processor = ProcessorV3(engine)

    for _ in range(128):
        outputs = processor.infer(inputs)
        top_token_id = torch.softmax(torch.tensor(outputs["output"]), dim=-1)[:, -1, :].flatten()
        top_token_id = torch.argsort(top_token_id)[-1:]
        inputs["decoder_input"] = np.append(inputs["decoder_input"], np.int64(top_token_id.item())).reshape(1, -1)

        if inputs["decoder_input"][0][-1] == 2:
            break

    print(inputs["decoder_input"])
