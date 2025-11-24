# K-VLM
VILA 기반 실시간 웹캠 VLM 데모

이 리포지토리는 NVILA 계열 한국어 모델을 이용해 웹캠 영상을 실시간으로 받아 질문에 답하는 VLM(Webcam Vision-Language Model) 데모 스크립트를 포함한다.
모델 파일은 용량 문제로 포함하지 않았으며, nvila_ko_chat_vector_1.5B와 nvila_ko_vlm_lora_20 LoRA 체크포인트를 사용한다.
구성 요소

    vila_environment.yml

        CUDA 12.x 및 PyTorch 2.3.0, Transformers 4.46.0, Flash-Attn 2.5.8, VILA 2.0.0 등 VLM 구동에 필요한 패키지들을 포함한 Conda 환경 정의 파일.

    live_test_vila_20_fast_wiondow.py

        NVILA 기반 LlavaLlama 모델을 로드하고, 커스텀 MM 프로젝터와 LoRA 어댑터를 적용해 실시간 웹캠 입력에 대해 질의응답을 수행하는 메인 데모 스크립트.

요구 사양

    OS: Linux (테스트 기준, cv2.CAP_V4L2 사용)

    GPU: NVIDIA GPU(CUDA 12.x, bfloat16 지원 권장), 최소 12GB 이상 vRAM 권장

    Python: 3.10 (Conda 환경에서 자동 설치)

설치 방법

    Conda 환경 생성

bash
conda env create -f vila_environment.yml
conda activate vila

    모델 및 체크포인트 준비

    nvila_ko_chat_vector_1.5B

        BASE_MODEL 디렉토리에 배치 (예: /home/user/Desktop/vlm/model/nvila_ko_chat_vector_1.5B).

    nvila_ko_vlm_lora_20

        LoRA 체크포인트 디렉토리(예: /home/user/Desktop/vlm/model/nvila_ko_vlm_lora_20/checkpoint-epoch3).

    mmprojector.pt

        위 LoRA 체크포인트 디렉토리 안에 mmprojector.pt 파일 위치.

스크립트 내 상단의 다음 상수를 실제 경로에 맞게 수정한다.

python
BASE_MODEL   = "/path/to/nvila_ko_chat_vector_1.5B"
CHECKPOINT_DIR = "/path/to/nvila_ko_vlm_lora_20/checkpoint-epoch3"
MMPROJECTOR_PT = os.path.join(CHECKPOINT_DIR, "mmprojector.pt")

실행 방법

bash
python live_test_vila_20_fast_wiondow.py

실행 시 동작:

    VILA 기반 LlavaLlama 모델 및 비전 타워, 커스텀 MM Projector 로드.

    LoRA 어댑터(nvila_ko_vlm_lora_20)를 LLM에 적용 후 GPU(bfloat16)로 이동.

    사용 가능한 카메라 ID(기본 0,1,2) 탐색 후 1280×720 해상도로 웹캠 오픈.

    2초 간격(FRAME_INTERVAL=2.0)으로 프레임을 선택하여 이미지 + 질문 프롬프트를 생성하고 model.generate로 응답 생성.

    화면 좌측 상단 오버레이에 프레임/추론 횟수와 최근 답변(최대 60자)을 출력하고 전체 화면 윈도우로 표시.

종료:

    ESC 키 또는 Ctrl+C 입력 시 추론 스레드를 정리하고 카메라/윈도우를 종료한다.

주요 구현 디테일

    커스텀 MM Projector

python
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

    mmprojector.pt 가중치를 로드해 model.mm_projector에 덮어쓰는 방식으로 사용한다.

    토크나이저 확장

        AutoTokenizer.from_pretrained(BASE_MODEL/llm, trust_remote_code=True)로 로드하고, 이미지 토큰 "image"가 없을 경우 특별 토큰으로 추가 후 resize_token_embeddings 호출.

        model.media_token_ids 및 config.media_token_ids에 이미지 토큰 ID를 세팅해 멀티모달 입력을 처리한다.

    웹캠 및 GUI

        cv2.VideoCapture(camera_id, cv2.CAP_V4L2)로 카메라를 연 뒤, PIL.ImageDraw를 사용해 한국어 폰트(NanumGothic, NanumBarunGothic, DejaVuSans, LiberationSans)를 우선 탐색하여 오버레이 텍스트를 렌더링한다.

        프레임 처리와 모델 추론은 별도 스레드로 분리하여, 프레임 드롭을 최소화하면서 실시간 GUI를 유지한다.

주의 사항 및 팁

    최초 실행 시 모델/LoRA 로딩과 mmprojector 로딩에 시간이 다소 소요될 수 있다.

    GPU 메모리가 부족한 경우:

        해상도를 낮추거나, FRAME_INTERVAL을 늘리고, max_new_tokens를 줄이는 식으로 튜닝할 수 있다.

    카메라 권한 문제가 있는 경우, 스크립트 출력의 안내대로 현재 유저를 video 그룹에 추가해야 할 수 있다.
