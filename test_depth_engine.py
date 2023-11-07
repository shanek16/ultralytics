import tensorrt as trt
import pycuda.driver as cuda
import torch
torch.cuda.set_device(0)
trt.init_libnvinfer_plugins(None, "")

cuda.init()
# Get the default CUDA device
cuda_device = cuda.Device(0)
# Create a context for the device
context = cuda_device.make_context()

logger = trt.Logger(trt.Logger.VERBOSE) #VERBOSE
runtime = trt.Runtime(logger)
# with open("/home/swkim/Project/weights/depth/engine/midas_dpt_hybrid_simplified.engine", "rb") as f:
with open("/workspace/weights/depth/midas_dpt_hybrid_simplified.engine", "rb") as f:
    engine_data = f.read()
engine = runtime.deserialize_cuda_engine(engine_data)
try:
    midas = engine.create_execution_context()
except:
    context.pop()
    exit()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform