# 자동화용 환경 준비 기록

이 디렉터리는 설치·검증·재현에 필요한 **비밀정보 없는 환경 자료**를 보존한다.
이 파일은 최초 환경 준비 기록이다. 현재 CLI·skill·에이전트 구현과 검증 상태는 [프로젝트 README](../README.md)와 [구현 현황](../tasks/implementation-status.md)을 따른다.
날짜별 상태는 해당 `bootstrap-status.json` 및 준비 보고서와 함께 확인한다.
모델 로그인 cache, API key, browser session, venv·패키지 cache는 포함하지 않는다.

## 2026-11-13 배치

- Brev: `robocurve-automation-256`, ID `th9ih02z4`.
- Crusoe `l40s-48gb.1x`, L40S 48GB, 실제 메모리 약 144GiB.
- `/data`: `/dev/vdb`, **274877906944 bytes = 256GiB**. OS disk는 별도다.
- 작업 공간: `/data/robocurve-automation`.
- VS Code용 진입 경로: `~/workspace/robocurve-automation` → 위 작업 공간.
- CPU venv: `/data/robocurve-automation/envs/control`.
- GPU baseline: 고정 Arena/YAM의 **native uv 환경**으로 준비한다.
  Docker GPU 검증은 별도다. Isaac Sim이 들어간 Docker 이미지까지 완성했다고
  해석하지 않는다. 자동 배포용 이미지 구성은 후속 구현에서 정한다.

## 정확한 인스턴스 생성

CLI의 명시적인 `--type`과 `--min-disk` 조합만으로는 스토리지 크기가 전달되지
않는다. 실제 v0.6.337의 dry-run과 block device 결과로 확인했다.
CLI의 JSON stdin 계약에서 디스크를 직접 지정한다.

```python
import json
import subprocess

subprocess.run(
    ["brev", "create", "robocurve-automation-256", "--stdin",
     "--jupyter=false", "--timeout", "600"],
    input=json.dumps([{
        "type": "l40s-48gb.1x",
        "provider": "crusoe",
        "target_disk_gb": 256,
    }]), text=True, check=True,
)
```

새 이름/빈 환경에서만 사용한다. 기존 인스턴스 재개에는 create를 사용하지 않는다.
생성 뒤 `lsblk -b`·`findmnt -T /data`로 크기와 mount를 확인한다.

## 도구와 인증

OS 도구는 `python3-venv`, `python3-pip`, `ffmpeg`, `gh`, `jq`, `cmake`,
`build-essential`, `pkg-config`, `libegl1`, `libgl1`, `libvulkan1`을 준비했다.
CPU 패키지는 날짜별 `control-requirements.txt`에 기록한다. simulator와 같은
Python 환경에 섞지 않는다.

Codex CLI는 로컬과 같은 0.147.0의 공식 Linux musl 배포본을 사용했다.
archive SHA-256은 `0246e2e773834e07f0fb5249ed6ebad12e4591e608f8c7bb97dd6a9690544c36`이다.
설치 경로는 `~/.local/bin/codex`다.

기기 코드 로그인은 계정 설정 때문에 사용할 수 없어 일반 브라우저 로그인으로
완료했다. 새 환경에서 로그인할 때에는 로컬 callback을 SSH로 전달할 수 있다.

```sh
ssh -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:1455:127.0.0.1:1455 robocurve-automation-256 \
  '/home/ubuntu/.local/bin/codex login'
```

명령이 출력한 **현재 실행의 URL**로 로그인한다. 예전 OAuth URL을 보존/재사용하지
않는다. `codex login status`와 최소 schema 응답 시험의 성공을 각각 확인했다.

## GPU Docker의 CDI 수정

새 이미지의 Container Toolkit 1.20.1이 driver 565.57.01의
`nvSandboxUtilsShutdown()`에서 SIGSEGV/SIGABRT로 종료했다. 따라서 CDI 파일이
생성되지 않아 GPU container를 시작할 수 없었다.

이 조합에서 공식 기능 플래그 `disable-nvsandboxutils`로 NVML 기반 장치 탐색을
사용하니 CDI 생성과 실제 GPU container 실행이 성공했다. 이는 장치 정보 검색
경로의 변경이며 kernel/container 격리를 끄는 설정이 아니다.

기본 Brev service는 Docker 뒤에서 실행하고 Docker를 다시 시작하도록 되어
있었는데, 새 Toolkit drop-in은 반대로 Docker 이전 실행을 요구했다. 준비한
service는 Docker에 대한 `After`·재시작 단계를 제거하고 환경 파일을 읽게 해
부팅 순서를 맞췄다. 날짜별 service와 `.env.example`을 비교용으로 보존한다.
다른 driver/image에 무조건 적용하지 말고 같은 오류인지 먼저 확인한다.

검증 명령:

```sh
nvidia-ctk cdi list
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 \
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

사용한 CUDA image digest:
`sha256:a10f328daee4e6797838a34d5eab69618b907f6313f5e6370dc6ea0105ae9628`.
컨테이너 GPU 인식과 Isaac Sim 물리·렌더의 성공은 별도로 검증한다.

## simulator 소스 준비

Arena `c7edf3a9d5d8276565208f72422f8f8a0806d6cf`, Isaac Lab
`91a0906f93802eb859192dcb0bca879f891e92d6`, YAM
`0059d02a1a91bc6834c1d0846022891df0c78730`을 사용한다. Dream policy 설치
소스도 `b5e936a201cec66c54c297e522c920d98445e339`로 지정하되 평가를 실행하지 않는다.

YAM에 불필요한 GR00T 외부 시뮬레이터 다운로드는 로컬 git config의
`submodule.submodules/Isaac-GR00T.update=none`으로 제외했다. 기본 uv group에는
선택적인 `gr00t-client`가 없다. Isaac Lab은 pinned commit의 source/scripts/apps와
루트 파일을 sparse checkout해 대형 문서·예제 자료의 다운로드를 줄였다.
원본 소스 commit을 변경하거나 성공 검사를 완화한 것은 아니다.

자세한 설치·실행 결과와 stop/start 검증 여부는 날짜별 준비 기록을 따른다.
패키지 설치 완료만으로 모델 평가·대회 제출까지 검증했다고 간주하지 않는다.

`simulator-requirements.txt`는 설치 결과의 기록이다. 단독 `pip install -r`로
모든 source package·uv override·driver·asset을 복구하는 파일은 아니다.
고정한 source checkout과 해당 setup/lock을 먼저 준비해야 한다.

## 검증 결과와 재개

2026-11-13 기본 카메라 검사는 8 pass / 0 fail이었다. 기존 `stack_cubes.py`를
`yam_scripted`로 실행해 재시작 전후 두 번 모두 `task_score=1.0`을 확인했다.
3개 카메라 영상은 H.264, 320×176, 20fps, 189프레임(9.45초)이다.
새 산업 태스크나 Dream 평가로 계산하지 않는다.

실제 STOPPED 상태에서 다시 start한 뒤 root/data marker의 hash, Codex 로그인,
GPU container·simulator package·source revision이 유지됐고 예제도 성공했다.
이는 **stop/start 한 사이클**의 검증이다. 인스턴스 삭제 후 복구나 workflow의
외부 요청 중복 방지·승인·제출 복구까지 검증한 것은 아니다.

로컬에서 다음과 같이 재개한다.

```sh
brev start robocurve-automation-256
brev open robocurve-automation-256 code --dir /data/robocurve-automation
```

원격 simulator shell에서는 다음 환경을 활성화한다. CPU 조정기용 control
venv와 구별하며, 설치 cache/venv를 Git에 넣지 않는다.

```sh
source /data/robocurve-automation/sources/IsaacLab-Arena/yam.env
```

작업 종료 후 로컬의 `brev stop robocurve-automation-256`을 사용하고, 명령 반환과
실제 STOPPED를 구분한다. 준비 시험에서는 STOPPING이 몇 분 유지됐다.

Inspect Robots 1.0의 기본 record는 이벤트·종료/guard 결과를 남기지만 이 파일을
전체 물리 상태·접촉 trace나 scorer replay의 검증으로 취급하지 않는다.
또한 record의 `approval` 이벤트는 action guard의 수정 판정이며 **사용자의 제출
승인과 무관하다**. 향후 pipeline은 별도 계측 recorder와 승인 계약을 사용해야 한다.

## 출처

- [Brev CLI create 구현](https://github.com/brevdev/brev-cli/blob/main/pkg/cmd/gpucreate/gpucreate.go)
- [OpenAI headless 인증](https://learn.chatgpt.com/docs/auth#login-on-headless-devices)
- [NVIDIA CDI 절차](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
- [장치 검색 기능의 공식 소스](https://github.com/NVIDIA/nvidia-container-toolkit/blob/v1.20.1/pkg/nvcdi/lib.go)
- [고정 YAM 설치 절차](https://github.com/kairosiann/yam-isaac-lab-arena/blob/0059d02a1a91bc6834c1d0846022891df0c78730/setup.sh)
