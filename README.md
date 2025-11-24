나에게
이 메일은 영어로 작성된 것 같습니다
K-VLM

Real-Time Webcam Vision-Language Model Demo (VILA-based)

This repository contains a real-time webcam VLM (Vision-Language Model) demo script built on the NVILA Korean model family.
It receives live webcam frames and answers user queries using nvila_ko_chat_vector_1.5B and the nvila_ko_vlm_lora_20 LoRA checkpoint.

    Note: Model weights are not included in this repository due to size limits.

Features

    Real-time webcam capture + multimodal Q&A

    NVILA-based LlavaLlama vision-language pipeline

    Custom multimodal projector

    LoRA-based fine-tuning support

    On-screen GUI overlay with live answers

    Multi-threaded frame processing for smooth performance

Repository Contents
vila_environment.yml

Conda environment specification including:

    CUDA 12.x

    PyTorch 2.3.0

    Transformers 4.46.0

    Flash-Attn 2.5.8

    VILA 2.0.0

    And other dependencies required for VLM execution

live_test_vila_20_fast_wiondow.py

Main demo script that:

    Loads the NVILA-based LlavaLlama model

    Applies a custom multimodal projector

    Loads and applies the LoRA adapter

    Processes real-time webcam frames

    Performs multimodal inference and displays results

Requirements

    OS: Linux (tested using cv2.CAP_V4L2)

    GPU: NVIDIA GPU with CUDA 12.x

        bfloat16 support strongly recommended

        At least 12GB vRAM recommended

    Python: 3.10 (installed automatically via Conda)

Installation
1. Create the Conda Environment

conda env create -f vila_environment.yml
conda activate vila

2. Prepare Model & Checkpoints
nvila_ko_chat_vector_1.5B

Place in your BASE_MODEL directory, e.g.:

/home/user/Desktop/vlm/model/nvila_ko_chat_vector_1.5B

nvila_ko_vlm_lora_20 (LoRA checkpoint)

Place in:

/home/user/Desktop/vlm/model/nvila_ko_vlm_lora_20/checkpoint-epoch3

mmprojector.pt

Place inside the same LoRA checkpoint directory:

/home/user/Desktop/vlm/model/nvila_ko_vlm_lora_20/checkpoint-epoch3/mmprojector.pt

3. Update Paths in the Script

Edit the top of live_test_vila_20_fast_wiondow.py:

BASE_MODEL     = "/path/to/nvila_ko_chat_vector_1.5B"
CHECKPOINT_DIR = "/path/to/nvila_ko_vlm_lora_20/checkpoint-epoch3"
MMPROJECTOR_PT = os.path.join(CHECKPOINT_DIR, "mmprojector.pt")

Running the Demo

python live_test_vila_20_fast_wiondow.py

Runtime Behavior

    Loads VILA-based LlavaLlama model and vision tower

    Loads LoRA adapter and applies it to the LLM

    Searches camera IDs (0, 1, 2) and opens at 1280×720

    Every 2 seconds (FRAME_INTERVAL = 2.0) a frame is selected

    Image + text prompt is sent to model.generate()

    On-screen overlay displays:

        Frame/Inference count

        Last answer (up to 60 characters)

    Fullscreen window with real-time updates

Exit Controls

    Press ESC

    Or terminate with Ctrl+C

Implementation Details
🔹 Custom Multimodal Projector

class CustomMMProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1152, 1344),
            nn.GELU(),
            nn.Linear(1344, 1536),
            nn.LayerNorm(1536),
        )
    def forward(self, x):
        return self.proj(x)

    The script loads mmprojector.pt and overwrites model.mm_projector.

🔹 Tokenizer Extension

    Uses AutoTokenizer.from_pretrained(BASE_MODEL/llm, trust_remote_code=True)

    Adds the "image" token if missing

    Calls resize_token_embeddings()

    Sets model.media_token_ids & config.media_token_ids for multimodal input

🔹 Webcam & GUI Processing

    Opens webcam using cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    Korean fonts auto-detected (NanumGothic, NanumBarunGothic, DejaVuSans, LiberationSans)

    Rendering done via PIL.ImageDraw

    Frame capture and inference run on separate threads to minimize latency

Tips & Troubleshooting
⚠️ Slow first launch

Model, LoRA, and projector loading may take time initially.
⚠️ GPU memory issues?

Try:

    Lower webcam resolution

    Increase FRAME_INTERVAL

    Reduce max_new_tokens
