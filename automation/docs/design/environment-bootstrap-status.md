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
