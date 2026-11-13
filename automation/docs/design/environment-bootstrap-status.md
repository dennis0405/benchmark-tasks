# 전용 Brev 환경 준비 기록

2026-11-13 (Asia/Seoul). 자동화 코드는 구현하지 않았으며 환경 준비만 진행한다.
아래 상태는 점검 당시 관측이며 향후 상태를 보증하지 않는다.

## 확정한 준비 방향

- 기존 실험과 별도인 전용 Brev 인스턴스를 생성한다.
- Codex는 기존 ChatGPT 계정의 원격 로그인부터 검증한다.
- 결과는 benchmark fork의 새 작업 브랜치에 보존하고 origin에 push한다.
- 실제 제출 PR은 최종 결과물 승인 후에만 생성한다.

## 로컬에서 확인

| 항목 | 관측 |
|---|---|
| Brev CLI | v0.6.337, 로그인되어 인스턴스 목록 조회 성공 |
| VS Code | CLI와 Remote SSH extension 있음 |
| GitHub | gh의 활성 계정 dennis0405, fork 읽기 확인 |
| Codex | CLI 0.147.0, 로컬 ChatGPT 로그인 상태 |
| benchmark origin | git@github.com:dennis0405/benchmark-tasks.git |
| benchmark upstream | git@github.com:robocurve/benchmark-tasks.git |
| 기존 checkout | feature/close-snack-bag. 전환/수정하지 않음 |

기존 `robocurve-snack-film`에는 실제 GPU 작업이 있었다. 읽기 점검만 했으며
stop·재설치·GPU 시험·프로세스 종료를 수행하지 않았다.

## 새 인스턴스

| 항목 | 값 |
|---|---|
| 이름 | robocurve-automation |
| Brev ID | tkweq1ez1 |
| provider/type | nebius / gpu-l40s-d.1gpu-16vcpu-96gb |
| 검색 당시 사양 | L40S 48GB 1개, 16 vCPU, RAM 96GiB |
| 요청 조건 | stop 지원, 디스크 최소 200GB, VM mode, Jupyter 설치 비활성화 |
| 검색 표시 가격 | 약 $2.18/hour. 실제 청구/할당 디스크와 별도 구분 |
| 최초 관측 상태 | DEPLOYING / PENDING / SHELL NOT READY |

후속 관측: SSH 준비 전에 `FAILURE`로 전환했다. 생성 명령은
`internal server error`로 종료했고 `brev stop robocurve-automation`도
`Internal Server Error`를 반환했다. 이 인스턴스를 정상 실행/정상 stop으로
기록하지 않는다. 데이터 업로드·패키지 설치·로그인은 수행하지 않았다.

## AWS 대체 환경

동일 작업의 fallback으로 다음 전용 인스턴스를 요청했다.

```sh
brev create robocurve-automation-aws --type g6e.2xlarge --provider aws --stoppable --min-disk 200 --jupyter=false --timeout 600
```

검색 사양은 L40S 48GB 1개·8 vCPU·RAM 64GiB, 표시 가격은 약 $2.69/hour다.
생성 접수와 SSH/런타임 준비 완료를 구분한다. 이 대체 시도가 실패하면 추가
provider를 계속 생성하지 않고 원인을 확인한다.

후속 관측: AWS는 RUNNING/COMPLETED/READY가 됐고 Brev 관리 SSH alias로
접속했다. L40S·driver 595.112.07·Python 3.10.12·Docker 접근을 확인했다.
실제 root는 117GiB이며 별도 412GiB ephemeral mount도 보였다. 요청한 min-disk
조건을 실제 영속 디스크 200GiB로 검증한 것은 아니다. 패키지 설치·Codex 로그인·
Isaac Sim 실행은 아직 수행하지 않았다.

사용자 지적 후 기존 Crusoe `l40s-48gb.1x` 생성 페이지를 직접 확인했다.
UI에는 L40S 48GB·8 CPU·RAM 147GB·256GiB와 stop/start 지원이 명시됐고,
총 $1.77/hr, stop 중 storage $0.03/hr로 표시됐다. 현재 CLI 검색에서 이 사양이
누락돼도 생성 불가/stop 미지원으로 판단하면 안 된다. 실제 새 할당은 아직 하지 않았다.
[확인한 생성 설정](https://brev.nvidia.com/environment/new?gpu=L40S&instanceType=l40s-48gb.1x&storage=256&location=any).

AWS의 추가 환경 작업을 멈추고 stop을 요청했으며 최초 확인은 STOPPING이다.
기존 과자봉지 실험 인스턴스는 중지하지 않았다. 기존 사양으로 신규 생성하는
동작도 아직 수행하지 않았다.

## GitHub 보존

기존 task checkout을 전환하지 않고 별도 임시 checkout에서
`automation/bootstrap` 브랜치를 만들었다. 설계·미착수 계획·이 준비 기록을
fork origin에 push했다. 대회 PR이나 미디어 업로드는 수행하지 않았다.
[보존 브랜치](https://github.com/dennis0405/benchmark-tasks/tree/automation/bootstrap).

생성에 사용한 실제 명령:

```sh
brev create robocurve-automation --type gpu-l40s-d.1gpu-16vcpu-96gb --provider nebius --stoppable --min-disk 200 --jupyter=false --timeout 600
```

## 다음 검증

실제 SSH·GPU/driver·mount/디스크·Docker, 격리한 CPU 환경과 Codex device-code
로그인을 확인한다. 이후 선택한 Isaac Sim/Arena/YAM 버전을 준비하고 작은 실제
렌더/제어 시험을 진행한다. stop/start 후 보존·재접속은 이 전용 instance에서만
검증한다. CLI·skill 구현 완료 또는 실제 GPU 데모·Dream 검증으로 기록하지 않는다.

로그인 cache·토큰·browser session은 Git 보존 대상이 아니다. 인증 진행을 위해
사용자가 브라우저에서 직접 로그인해야 할 수 있다.


## 사용자 지정 Crusoe 환경으로 전환

사용자가 Crusoe L40S와 256GiB 설정을 지정했고 이전 대체 인스턴스는 직접
삭제했다고 알렸다. `l40s-48gb.1x`는 검색 목록에 없어도 CLI `--type` 지정으로
생성 가능했다. 다만 CLI 소스와 실제 디스크 확인 결과, 명시적인 `--type`에서는
`--min-disk`가 allocation 크기로 전달되지 않는다. 이 방식의 첫 빈 VM
`powkoyto9`는 설치/자료 저장 전에 CLI로 정리했다.

스토리지는 공식 CLI의 JSON stdin 계약 `target_disk_gb`를 직접 사용한다.
dry-run 출력에서 `l40s-48gb.1x (256GB disk)`를 확인한 뒤 다음과 같이 요청했다.

```python
import json
import subprocess

spec = [{"type": "l40s-48gb.1x", "provider": "crusoe", "target_disk_gb": 256}]
subprocess.run(
    ["brev", "create", "robocurve-automation-256", "--stdin",
     "--jupyter=false", "--timeout", "600"],
    input=json.dumps(spec), text=True, check=True,
)
```

이 코드는 환경 생성 명령의 기록이며 자동화 CLI 구현이 아니다. 새 instance의
실제 block device·mount·stop/start 보존을 다시 확인한다.
참고: [공식 CLI 소스](https://github.com/brevdev/brev-cli/blob/main/pkg/cmd/gpucreate/gpucreate.go).

## 현재 사용하는 최종 Crusoe 환경

| 항목 | 실제 확인 결과 |
|---|---|
| 이름 / ID | robocurve-automation-256 / th9ih02z4 |
| GPU / driver | L40S 46068MiB / 565.57.01 |
| RAM | 약 144GiB |
| data disk | /dev/vdb, 274877906944 bytes = 256GiB, /data에 ext4 mount |
| 작업 공간 | /data/robocurve-automation, ~/workspace/robocurve-automation으로 연결 |
| CPU 환경 | envs/control, HTTPS 200·SQLite FTS5·필수 import 통과 |
| Codex | CLI 0.147.0, ChatGPT 일반 브라우저 로그인 성공·최소 JSON 응답 성공 |
| GPU Docker | CDI 호환성 수정 후 실제 container에서 L40S 인식 |
| simulator | Isaac Sim 6.1.0.0·Arena 0.3.0·Isaac Lab 17.0.2·YAM 0.1.0 native uv 설치 완료 |
| 실제 렌더·관절 시험 | upstream stage5 카메라/그리퍼 추종 검사 8 pass / 0 fail |
| 실행기·데모·채점 | upstream stack_cubes, yam_scripted, task_score 1.0, 1 trial 성공 |
| stop/start 보존 | 아직 미검증 |
| 자동화 구현·Dream·제출 | 수행하지 않음 |

Codex 기기 코드 인증은 계정 설정으로 제한돼 있었으므로 보안 설정을 변경하지
않고 SSH callback 포워딩으로 로그인했다. OAuth URL·인증 cache는 저장소에 넣지 않는다.

Container Toolkit 1.20.1의 장치 검색이 driver 565.57.01의 sandboxutils 종료
함수에서 충돌했다. NVML 탐색을 사용하는 공식 기능 플래그로 CDI 생성에
성공했고, 새 Toolkit과 기존 Brev unit의 Docker 시작 순서도 정리했다.
실제 설정·버전·진단 결과는 [환경 자료](../../environment/README.md)에 보존한다.

기본 YAM 검증은 새 산업 태스크가 아니다. 기존 upstream 예제를 사용했고 Dream
호출·새 scorer·과자봉지 실험·대회 제출은 하지 않았다. 생성 영상은 3개 카메라
각 189프레임·20fps이며, 원본 결과 JSON과 함께 보존한다. driver 565에서는
DLSS Responsive Denoising이 preset D로 fallback된 경고가 있었다. 실제 사용한
driver/render 경로를 기록하며 최신 공식 driver 검증을 대신했다고 주장하지 않는다.
