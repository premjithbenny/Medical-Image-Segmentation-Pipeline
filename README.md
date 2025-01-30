# Medical-Image-Segmentation-Pipeline

🚀 End-to-End AI Pipeline for 3D Medical Imaging on Jetson Orin Nano

Goal:
We are building a complete AI pipeline that:
✅ Loads 3D medical images (DICOM/NIfTI)
✅ Runs AI-based segmentation using a pretrained MONAI model
✅ Uses TensorRT to optimize inference for Jetson Orin Nano
✅ Deploys the model with Triton Inference Server
✅ Visualizes segmentation using ViST3D + VTK
✅ Supports real-time medical imaging workflows using Holoscan

🔹 Step 1: Set Up Jetson Orin Nano

📌 1.1 Install JetPack SDK (Linux OS for Jetson)

💡 Why?

JetPack includes CUDA, cuDNN, TensorRT, and other AI libraries optimized for Jetson.
Install JetPack
1️⃣ Update the system:

sudo apt update && sudo apt upgrade -y
2️⃣ Install JetPack SDK (if not installed already):

sudo apt install -y nvidia-jetpack
3️⃣ Check CUDA & GPU acceleration:

nvcc --version         # Check CUDA version
tegrastats             # Monitor GPU usage
sudo jetson_clocks     # Maximize GPU performance
🔹 Step 2: Install AI Libraries

📌 2.1 Install PyTorch, MONAI & Medical Image Processing Tools

💡 Why?

PyTorch: ML framework for training & inference
MONAI: Medical AI framework (DICOM/NIfTI processing)
nibabel & pydicom: Libraries to handle medical imaging formats
Install Dependencies
pip3 install torch torchvision torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v51
pip3 install monai[all] nibabel pydicom matplotlib
Verify PyTorch is using Jetson's GPU
import torch
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
💡 If output shows True & Jetson GPU name, PyTorch is correctly installed!

🔹 Step 3: Deploy a Pretrained Medical AI Model

📌 3.1 Download a Pretrained Medical AI Model (MONAI UNet-3D)

💡 Why?

A 3D U-Net model is commonly used for organ/tumor segmentation in CT/MRI scans
Download Model (Brain Tumor Segmentation - MONAI Zoo)
python3 -c "from monai.apps import load_from_mmar; load_from_mmar('clara_pt_brain_mri_segmentation_t1c', './models', progress=True)"
📂 This creates a folder models/ with a trained model checkpoint.

📌 3.2 Convert AI Model to ONNX (for TensorRT Optimization)

💡 Why?

TensorRT only supports ONNX models for inference optimization.
Convert PyTorch Model to ONNX
import torch
import monai
from monai.networks.nets import UNet

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
)

dummy_input = torch.randn(1, 1, 128, 128, 128)  # Example 3D medical image
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
📂 This saves model.onnx, which we will optimize using TensorRT.

🔹 Step 4: Optimize the AI Model with TensorRT

💡 Why?

TensorRT accelerates model inference by using FP16 (half-precision) calculations optimized for Jetson’s GPU.
Install TensorRT & Optimize ONNX Model
sudo apt install -y python3-libnvinfer libnvinfer-bin
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
📂 This converts model.onnx into a TensorRT engine (model.trt), optimized for Jetson.

🔹 Step 5: Run Inference on a Medical Image

📌 5.1 Load a DICOM/NIfTI Scan & Run AI Model

import monai
import torch
import nibabel as nib

# Load medical scan (NIfTI format)
image = nib.load("sample.nii.gz").get_fdata()

# Preprocess (normalize intensity)
image = (image - image.min()) / (image.max() - image.min())
image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Add batch & channel dims

# Load Optimized Model (TensorRT)
model = torch.jit.load("model.trt")
output = model(image.cuda())  # Run AI model
📌 This outputs a 3D segmentation mask for the scan.

🔹 Step 6: Deploy Model with Triton Inference Server

💡 Why?

Triton allows AI models to be served via API (ideal for large-scale AI applications).
Install Triton Inference Server on Jetson
wget https://developer.download.nvidia.com/compute/tritonserver/repos/jetpack5.1/tritonserver-2.34.0_arm64.deb
sudo dpkg -i tritonserver-2.34.0_arm64.deb
Set Up Triton Model Repository
mkdir -p ~/triton_models/unet_3d/1/
mv model.trt ~/triton_models/unet_3d/1/model.plan
📂 Create config.pbtxt to define the model for Triton:

echo '
name: "unet_3d"
platform: "tensorrt_plan"
max_batch_size: 1
input [ { name: "INPUT__0", data_type: TYPE_FP32, dims: [ 1, 128, 128, 128 ] } ]
output [ { name: "OUTPUT__0", data_type: TYPE_FP32, dims: [ 1, 128, 128, 128 ] } ]
' > ~/triton_models/unet_3d/config.pbtxt
Start Triton Server
tritonserver --model-repository=~/triton_models --strict-model-config=false
Run Inference via API
import tritonclient.grpc as grpcclient
triton_client = grpcclient.InferenceServerClient("localhost:8001")
inputs = grpcclient.InferInput("INPUT__0", [1, 128, 128, 128], "FP32")
inputs.set_data_from_numpy(image.numpy())
results = triton_client.infer("unet_3d", model_inputs=[inputs])
output_data = results.as_numpy("OUTPUT__0")
🔹 Step 7: Visualize the 3D Segmentation with ViST3D

Install ViST3D & VTK
pip3 install vist3d vtk
Render the 3D Segmentation
import vtk
from vist3d import Renderer, Volume

volume = Volume(output_data.squeeze())  # Convert segmentation mask to 3D object
renderer = Renderer()
renderer.add_volume(volume)
renderer.show()
📌 This renders the segmentation as a 3D model! 🎉

🚀 Final Thoughts

This Jetson-powered pipeline enables:
✅ Fast 3D medical AI inference with TensorRT
✅ Scalable model deployment via Triton Server
✅ Real-time visualization with ViST3D & VTK
