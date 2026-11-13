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
