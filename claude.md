프로젝트: Mobilint NPU 성능·메모리 프로파일링 (claude.md)

목표
- Jetson Orin (JetPack 6)에서 Mobilint NPU로 제공된 `.mxq` 모델들을 실행하여 동작 검증, 추론 성능(지연시간/throughput) 측정, 메모리 사용량 프로파일링 및 상세 분석을 수행한다.

첫 번째 단계
- Model Zoo에 있는 다음 모델들을 NPU에서 실행하고 정상 동작 확인 및 기본 성능 측정
  - DeiT_Tiny_Patch16_224
  - ViT_Tiny_Patch16_224
  - ViT_Small_Patch16_224
  - DeiT_Small_Patch16_224

요구 사항(요약)
- Jetson Orin, JetPack 6
- Python 3 (시스템 기본) + 가상환경(.env)
- Mobilint 런타임/드라이버 (qbruntime 또는 vendor가 제공한 NPU 런타임)
- pip 패키지: `mblt-model-zoo`, `opencv-python-headless` 등 (아래 `requirements.txt` 참조)

환경 설정 (빠른 예)
1. 가상환경 만들기 및 활성화

```bash
python3 -m venv .env
source .env/bin/activate
```

2. 필수 파이썬 패키지 설치

```bash
pip install -r requirements.txt
```

3. (필요시) Mobilint NPU 런타임 설치
- NPU 공급사에서 제공하는 `qbruntime` 또는 vendor 패키지를 설치해야 합니다. 시스템 제공 설치 가이드를 따르세요.

첫 NPU 테스트 실행 방법
- 제공된 파이썬 스크립트 `scripts/run_npu_test.py`를 사용합니다. 예:

```bash
source .env/bin/activate
export MBLT_MODEL_ZOO_VERBOSE=true
python3 scripts/run_npu_test.py --models DeiT_Tiny_Patch16_224 ViT_Tiny_Patch16_224 ViT_Small_Patch16_224 DeiT_Small_Patch16_224
```

스크립트 기능 요약
- 각 모델을 순차적으로 로드하여 지정한 이미지(기본: `imagenet_val/ILSVRC2012_val_00000001.JPEG`)에 대해 전처리→추론→후처리를 수행합니다.
- 모델 로딩/추론 소요 시간(초)을 출력합니다.
- `--mxq-path` 옵션으로 로컬 MXQ 파일 지정 가능.
- `--core-mode` 옵션으로 NPU 실행 모드 지정 (single/multi/global4/global8).

기본 프로파일링/로그 수집
- NPU/시스템 측 메트릭 수집
  - `tegrastats` 실행(또는 `jtop`)으로 GPU/CPU/메모리/온도 추적
  - `nvpmodel` 설정 확인
  - Mobilint의 verbose 모드(`MBLT_MODEL_ZOO_VERBOSE=true`)로 모델 크기/해시 확인

- 예시: 추론 실행 전후로 시스템 모니터링

```bash
sudo tegrastats --interval 1000 &> tegrastats.log &
export MBLT_MODEL_ZOO_VERBOSE=true
python3 scripts/run_npu_test.py --models DeiT_Tiny_Patch16_224
pkill -f tegrastats
```

향후 계획
- Reduced(경량화) 및 커스텀 모델에 대해 동일한 파이프라인으로 성능·메모리 분석
- (옵션) 원격 서버에서 대량 벤치마크 자동화 및 로그 수집

참고
- 모델 이름/구성은 `mblt-model-zoo` 레지스트리와 일치해야 합니다. 로컬 `.mxq`를 우선 사용하려면 `--mxq-path`로 지정하세요.

문제가 생기면 `scripts/run_npu_test.py` 실행 로그와 `tegrastats.log`를 공유해 주세요.
