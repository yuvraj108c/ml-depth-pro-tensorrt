import torch
from torch.cuda import nvtx
from collections import OrderedDict
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy import util
from polygraphy.backend.trt import ModifyNetworkOutputs, Profile
from polygraphy.backend.trt import (
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER
import tensorrt as trt
from logging import error, warning
from tqdm import tqdm
import copy
from cuda import cudart

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
G_LOGGER.module_severity = G_LOGGER.ERROR

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}

# https://github.com/Jeff-LiangF/streamv2v/blob/18c1a3bd56ff348d54a3300605936980bb13b03c/src/streamv2v/acceleration/tensorrt/utilities.py
def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None

class TQDMProgressMonitor(trt.IProgressMonitor):
    def __init__(self):
        trt.IProgressMonitor.__init__(self)
        self._active_phases = {}
        self._step_result = True
        self.max_indent = 5

    def phase_start(self, phase_name, parent_phase, num_steps):
        leave = False
        try:
            if parent_phase is not None:
                nbIndents = (
                    self._active_phases.get(parent_phase, {}).get(
                        "nbIndents", self.max_indent
                    )
                    + 1
                )
                if nbIndents >= self.max_indent:
                    return
            else:
                nbIndents = 0
                leave = True
            self._active_phases[phase_name] = {
                "tq": tqdm(
                    total=num_steps, desc=phase_name, leave=leave, position=nbIndents
                ),
                "nbIndents": nbIndents,
                "parent_phase": parent_phase,
            }
        except KeyboardInterrupt:
            # The phase_start callback cannot directly cancel the build, so request the cancellation from within step_complete.
            _step_result = False

    def phase_finish(self, phase_name):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    self._active_phases[phase_name]["tq"].total
                    - self._active_phases[phase_name]["tq"].n
                )

                parent_phase = self._active_phases[phase_name].get("parent_phase", None)
                while parent_phase is not None:
                    self._active_phases[parent_phase]["tq"].refresh()
                    parent_phase = self._active_phases[parent_phase].get(
                        "parent_phase", None
                    )
                if (
                    self._active_phases[phase_name]["parent_phase"]
                    in self._active_phases.keys()
                ):
                    self._active_phases[
                        self._active_phases[phase_name]["parent_phase"]
                    ]["tq"].refresh()
                del self._active_phases[phase_name]
            pass
        except KeyboardInterrupt:
            _step_result = False

    def step_complete(self, phase_name, step):
        try:
            if phase_name in self._active_phases.keys():
                self._active_phases[phase_name]["tq"].update(
                    step - self._active_phases[phase_name]["tq"].n
                )
            return self._step_result
        except KeyboardInterrupt:
            # There is no need to propagate this exception to TensorRT. We can simply cancel the build.
            return False

class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def reset(self, engine_path=None):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors
        self.engine_path = engine_path

        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.inputs = {}
        self.outputs = {}

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        update_output_names=None,
    ):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = [Profile()]
        if input_profile:
            p = [Profile() for i in range(len(input_profile))]
            for _p, i_profile in zip(p, input_profile):
                for name, dims in i_profile.items():
                    assert len(dims) == 3
                    _p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        network = network_from_onnx_path(
            onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
        )
        if update_output_names:
            print(f"Updating network outputs to {update_output_names}")
            network = ModifyNetworkOutputs(network, update_output_names)

        builder = network[0]
        config = builder.create_builder_config()
        config.progress_monitor = TQDMProgressMonitor()

        config.set_flag(trt.BuilderFlag.FP16) if fp16 else None
        config.set_flag(trt.BuilderFlag.REFIT) if enable_refit else None

        profiles = copy.deepcopy(p)
        for profile in profiles:
            # Last profile is used for set_calibration_profile.
            calib_profile = profile.fill_defaults(network[1]).to_trt(
                builder, network[1]
            )
            config.add_optimization_profile(calib_profile)

        try:
            engine = engine_from_network(
                network,
                config,
            )
        except Exception as e:
            error(f"Failed to build engine: {e}")
            return 1
        try:
            save_engine(engine, path=self.engine_path)
        except Exception as e:
            error(f"Failed to save engine: {e}")
            return 1
        return 0

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
        #    self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        nvtx.range_push("allocate_buffers")
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]["shape"]
            else:
                shape = self.context.get_tensor_shape(name)

            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(
                tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]
            ).to(device=device)
            self.tensors[binding] = tensor
        nvtx.range_pop()

    def infer(self, feed_dict, stream, use_cuda_graph=False):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        if use_cuda_graph:
            if self.cuda_graph_instance is not None:
                CUASSERT(cudart.cudaGraphLaunch(self.cuda_graph_instance, stream.ptr))
                CUASSERT(cudart.cudaStreamSynchronize(stream.ptr))
            else:
                # do inference before CUDA graph capture
                noerror = self.context.execute_async_v3(stream.ptr)
                if not noerror:
                    raise ValueError("ERROR: inference failed.")
                # capture cuda graph
                CUASSERT(
                    cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                )
                self.context.execute_async_v3(stream.ptr)
                self.graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                self.cuda_graph_instance = CUASSERT(cudart.cudaGraphInstantiate(self.graph, 0))
        else:
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors