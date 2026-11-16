# RoboCurve automation

프롬프트로 산업 태스크 후보를 조사하고 하나를 제작하여 **제출 전 검토**까지 진행하는 CLI MVP다.
실제 제출은 사용자가 결과물을 검토하고 승인한 뒤 별도 명령으로 수행한다.
2026-11-14 기준 CLI·skill·Brev 실행 경로를 구현했다. 첫 산업 파일럿은 후보군이 신규성 검사에서
차단되어 종료했다. [실행에서 발견한 문제와 결과](docs/design/first-industrial-pilot.md)를 남겼다.
모의 실행·기존 YAM 예제 성공과 신규 태스크 완성·채택을 구분한다.
후속 HELICOIL 파일럿은 물리·YAM 검증 전이다. Brev 시작 실패 뒤 새 L40S에 환경을
복원했고, 사용자 요청에 따라 작업군을 넓혀 실제 시제품 실행을 먼저 확인하고 있다.
[측정 결과·보존 및 복구 기록](docs/design/native-pilot-recovery.md)을 참고한다.

```text
사용자 프롬프트 → 실행 기록·목록 갱신 → 지정 Brev start
 → 산업 원문·에셋 조사 → 후보 다수 비교·선정·사실 검증
 → 6개 벤치마크 원문 비교 보고서 → 전수 제출 목록 비교·독립 검토
 → 계약 고정 → 환경·물리 검사 → 데모·채점·대조군 → 이식 검사
 → Dream 최종 episode 1회 → 로컬 보존·fork push → Brev stop
 → 최신 중복 재검사·보고서 → 사용자 검토
     ├ 수정 요청: 같은 run·브랜치로 재개
     └ 승인: 최신 중복 재검사 → 미디어 업로드·PR → 접수 확인
```

## 사용

현재 Mac에 `rcauto`와 `robocurve-workflow` skill을 설치했다. CLI는 이 소스에 editable로 연결된다.
다른 Mac에는 다음과 같이 설치한다. Brev CLI·GitHub CLI 로그인과 ffmpeg/ffprobe도 필요하다.

```sh
uv tool install --editable /Users/dennis0405/Workspace/SNU-PI/robocurve/automation
rcauto doctor
```

초기 인증은 **사용자 터미널에서 직접** 입력한다. 토큰을 채팅에 보내지 않는다.
Dream 인증은 키를 저장할 뿐 추론을 보내지 않는다. 팀 인증은 공식 로그인만 확인한다.
인증은 프로젝트 밖 로컬 state의 `private/credentials.json`에 권한 0600으로 보관한다.

```sh
rcauto auth board
rcauto auth dream
```

여러 줄·특수문자를 포함한 요청은 파일로 전달한다.

```sh
rcauto start --prompt-file /absolute/path/request.txt --detach
rcauto status RUN_ID
rcauto resume RUN_ID --request '접촉 형상을 이 조건에 맞춰 수정해줘' --from-stage environment --detach
rcauto stop RUN_ID
```

`--detach`에서는 터미널을 닫아도 로컬 supervisor가 계속 실행되며 macOS caffeinate를 사용한다.
**Mac 전원·네트워크는 유지해야 한다.** 단절 뒤에도 stop됐다고 가정하지 않는다. 같은 run으로
`resume`하면 실행 중인 자기 worker에 연결하거나 보존한 단계부터 재개한다.
`stop`은 단계 경계에서 보존하는 요청이며 즉시 VM 종료가 아니다.

## 검토·승인·제출

기본 state는 `~/.local/state/rcauto`다. `--state PATH`로 분리할 수 있다.
각 run의 `report.html`에서 영상·초기 장면·근거·계약·검토 결과를 확인한다.
`release-manifest.json`은 검토한 파일들의 해시와 digest다. 파일이나 보고서가 바뀌면 승인은 무효다.

```sh
rcauto approve RUN_ID --digest RELEASE_DIGEST --actor dennis0405 --statement '보고서와 이 버전의 제출을 승인함'
rcauto submit RUN_ID --approval APPROVAL_ID
rcauto receipt RUN_ID
```

승인·제출은 GPU를 켜지 않는다. 제출 직전에도 최신 목록과 최종 목적·물리 기술·성공 조건을
비교한다. 한 task 폴더만 별도 checkout에서 발행한다. 미디어 업로드·PR 생성 전에 journal을
남기며 응답 유실을 새 요청으로 덮어쓰지 않는다. PR 생성·접수·채택은 별도 상태다.

## Dream와 복구

- 환경·데모·채점·대조군·이식 검사가 끝난 뒤 태스크당 최종 episode 한 번만 실행한다.
- episode 안의 여러 요청과 SDK 재시도도 기록한다. 실패·결과 불명이어도 새 episode를 자동 생성하지 않는다.
- 키가 없으면 제작물을 보존하고 GPU를 stop한 뒤 평가 대기 사유를 보고한다.
- `rcauto evaluate RUN_ID`는 아직 시도하지 않은 최종 평가만 수행한다.
- 평가 후 문서 수정은 입력 digest가 같은 경우 기존 결과를 재사용한다. 물리·관측·지시·채점 입력이
  바뀌면 기존 결과를 유효하게 처리하지 않고 예외 검토로 보낸다.
- `rcauto finalize RUN_ID`는 완료한 평가의 중복 검사·보고·Git 보존을 로컬에서 복구한다.

누적 예산은 $20 ledger로 관리한다. 실제 과금 API를 검증하지 않은 상태에서 보고된 $0.5를
실제 청구액으로 확정하지 않는다. 기본값은 비용 불명에 **전체 $20를 보수적으로 예약**하며,
확인 전에는 다음 유료 평가를 막는다. 이는 제공자 측 결제 한도를 대신하지 않는다.
실제 비용을 확인한 경우 검토 과정에서 다음과 같이 반영한다. 실패도 0달러로 추정하지 않는다.

```sh
rcauto reconcile RUN_ID --dream-actual-usd ACTUAL_USD --billing-reference PROVIDER_USAGE_REFERENCE
```

## 검증

```sh
uv run pytest -q
uv run ruff check src tests
rcauto selftest
rcauto selftest --remote
rcauto doctor --remote --agent --sim
rcauto catalog-sync
```

`selftest`는 합성 fixture다. `--remote`는 실제 Brev start·SSH·별도 worker·7단계 실행·파일 반출·
stop을 검증하지만 모델·Dream·대회 제출을 호출하지 않는다. fixture는 승인·평가·제출할 수 없다.
`doctor --sim`은 고정된 upstream 블록 쌓기 예제이며 산업 태스크 제작 실적으로 세지 않는다.

## 저장과 실행 경계

| 위치 | 내용 |
|---|---|
| `src/rcauto` | CLI·기록·수집·에이전트·검사·worker·평가·발행 |
| `config/production.json` | 등록 instance ID·fork·runtime pin·반복 한도 |
| `skills/robocurve-workflow` | CLI를 호출하는 운영 skill |
| 로컬 state의 `runs/RUN_ID` | 보고·영상·원시 기록·승인·외부 요청 이력 |
| `/data/robocurve-automation/worker-state` | 원격 worker journal과 작업 공간 |
| fork의 `automation/RUN_ID` | `tasks/SLUG`와 `automation-results/RUN_ID` 보존 |
| fork의 `submit/RUN_ID` | 승인 후 대회 PR에 쓸 task 폴더만 발행 |

현재 검증한 **native uv + Isaac Sim 6.1 + Arena/YAM**을 사용한다. Docker GPU 접근도
확인했지만 범용 Isaac Sim 배포 이미지를 완성했다고 주장하지 않는다. 태스크마다 깨끗한 환경
이식 검사를 실행하고 실제 검증 범위를 적는다. 50MB 초과 파일 또는 100MB 초과 공개 checkpoint는
업로드 전에 보관 방식 검토로 보낸다. 산업 매뉴얼 전체·인증 cache는 공개 보존하지 않는다.
공개 checkpoint의 1MiB 이상 원시 JSON·JSONL·텍스트 로그는 gzip으로 손실 없이 압축한다.
압축 후에도 한도를 넘는 **실패한 native 사전 시험의 접촉 상세 기록**은 사용자가 보관 방식을
정한 경우에만 별도로 처리한다. run 밖으로 원본을 삭제하지 않고 Mac의
`private-experimental-evidence/`와 Brev에 보존하며, SHA-256으로 확인한 원본·실패 보고서·
전체 스텝의 경량 동작 기록을 `external-evidence-policy.json`에 연결한다.
GitHub 복사본에서는 승인한 해시의 상세 기록만 제외하고 `external-evidence.json`에
누락 경로와 원본 해시를 남긴다. 이 경우 GitHub만으로 전체 원시 접촉 기록을 복원할 수 없다.
새 실험으로 해시가 바뀐 기록이나 성공·qualification 결과에는 이 예외를 적용하지 않는다.
실험 작업 공간의 원본은 그대로 유지한다. 다운로드한 checkpoint에서
`python3 restore-evidence.py`를 실행하면 `evidence-archive.json`의 크기·SHA-256을 검증하고
기존 경로로 복원한다. 압축 후에도 기존 용량 제한을 적용하며, 데모 영상·작은 채점 보고서와
제출할 task 파일은 압축하지 않는다.
실험 중인 native prototype의 형상·물성 기준은 `frozen-inputs/`에도 원본 바이트 그대로
보존한다. `frozen-inputs-manifest.json`에 복구 위치와 SHA-256이 기록된다.
네이티브 진단 실행 전에는 코드·에셋 원본을 `measurements/native-source-blobs/`에
해시별로 중복 없이 저장한다. `native-source-snapshots/`의 매핑으로 당시 파일 경로와
바이트를 복구할 수 있으며, 이후 수정·삭제되거나 검사가 실패해도 원본이 남는다.
이 스냅샷 자체는 실행 성공이나 물리 검증 통과를 뜻하지 않는다.

기본 도형 DSL로 표현할 수 없는 다관절 기구에는
[task 전용 네이티브 YAM 어댑터](docs/design/native-runtime-adapter.md)를 명시적으로 사용한다.
기존 YAM 동작·관측·채점 메서드를 유지하며, GPU 검증과 데모를 별도로 통과해야 한다.

수집원은 GitHub 전체 PR이다. 모든 PR의 제목·본문을 남기고 open·merged 및 등록 제출자의
명세를 SHA별로 확보한다. 모든 open을 보수적으로 비교하며 pending으로 단정하지 않는다.
분야·작성자로 비교 대상을 제한하지 않는다. board 직접 동기화·dense embedding은 필수 의존성이
아니며 전수 batch 비교와 원문 반박 검토를 우선한다. 필수 자료가 빠지면 이전 완성 snapshot을
유지하고 신규성 검사를 통과시키지 않는다.

팀 인증을 저장한 경우 공식 보드의 accepted·pending·재검토 보유 이력과 우리 팀 전체 이력을
함께 동기화한다. 제목·지시문이 정확히 일치하는 GitHub 명세만 연결하고 다른 버전은 보존한다.
원문에 없는 pilot 명세 항목은 빈 항목으로 표시하며 새로 만들지 않는다. 철회 이력을 보드가
숨기면 `board_history_receipts`에 등록한 PR의 공식 접수 댓글과 접수 시점 커밋으로 복구한다.
태스크 ID·접수 작성자·제목을 검증하지 못하면 통과시키지 않는다.

닫힌 미병합 타 팀 PR도 카탈로그 원문과 기존 비교 기록을 보존한다. 완전한 최신 보드 동기화로 현재 접수·예약 대상이
아님을 명시적으로 확인한 타 팀 종료 기록은 `archival_records`에 범위 근거를 남기는 참고 자료다.
유효한 기존 판정이 있으면 그대로 유지하고 중복·불확실 판정은 `inactive_history`에도 남긴다. 이를 `distinct`로 바꾸지 않는다. `inactive_uncertainty`는
호환용 부분 목록이다. Accepted·merged·Pending·open·보드상 예약/재검토·우리 팀 이력과
상태 불명 기록은 중복 또는 불확실이면 계속 차단한다. 공개 벤치마크도 동일하게 차단한다. `required_total`·`required_compared`는 필수 대상의 전수 비교를,
`compared`는 실제 비교된 기록 수를, `catalog_total`은 종료 이력을 포함한 전체 수를 뜻한다.

비교 응답에서 일부 ID가 빠지면 후보를 탈락시키지 않고 누락된 항목만 최대 두 번 추가 요청한다.
확인된 결과는 후보·요청 맥락·비교 지침·원문 레코드의 해시로 보존한다. 재개와 목록 갱신에서는
이 입력이 모두 같은 결과만 재사용하며, 바뀐 기록은 다시 검사한다. 누락이 끝내 해결되지 않으면
신규성 판정과 구분한 `CoverageError`로 중지한다.

## 설계·완료 범위

- [현재 구현·검증 현황](tasks/implementation-status.md)

- [사용자가 확정한 제작 의도](docs/intent/robocurve-automation.md)
- [선택한 방향과 MVP 범위](docs/ideas/industrial-task-pipeline.md)
- [운영 설계 초안: 단계별 산출물·통과 조건·실패 처리](docs/design/mvp-workflow.md)
- [기술 설계 초안: 에이전트·도구·Brev 실행 위치](docs/design/technical-architecture.md)
- [상세 기술 설계: 자료·에셋 수집, 제출 동기화와 중복 검사](docs/design/sources-and-novelty.md)
- [산업 원문·형상·실행기 사전 점검에서 확인한 사항](docs/design/industrial-preflight-lessons.md)
- [상세 기술 설계: 환경·에셋 제작과 물리 검증](docs/design/environment-and-validation.md)
- [상세 기술 설계: 데모·채점·Dream·승인 후 제출](docs/design/demo-evaluation-and-submission.md)
- [MVP 구현 순서·단계별 검증 계획](tasks/plan.md)
- [MVP 상세 검증 체크리스트](tasks/todo.md)
- [Brev 사전 준비: 도구·인증·사용자 결정](docs/design/brev-readiness.md)
- [CLI·전용 skill·Brev start/stop·fork 결과 보존](docs/design/cli-skill-and-lifecycle.md)
- [전용 Brev 환경 준비 점검 기록](docs/design/environment-bootstrap-status.md)
- [실제 환경 설정·의존성·검증 자료](environment/README.md)

2026-11-16 사용자 확정 범위는 현장 조립·정비(Impact on jobs)와 로봇 제작·데이터센터
(Impact on AI timelines)다. 실제 산업 근거에 기반한 후보 다수 검토·단일
태스크 제작·승인 후 제출이다. 강체를 기본으로 하되 검증 가능한 기존 연성
에셋을 허용한다. [실행 명령 근거](https://learn.chatgpt.com/docs/developer-commands#codex-exec),
[권한 profile](https://learn.chatgpt.com/docs/permissions),
[대회 규칙](https://github.com/robocurve/benchmark-tasks/blob/main/AGENTS.md)을 참고한다.


환경 제작은 작은 native `preflight`와 전체 `qualification`을 구분한다. 사전 시험 결과는
다음 구현에 반영되며 환경 통과로 계산하지 않는다. 실행 전 소스 검토는 상수·대용 수치를
실제 측정으로 오인하지 않게 확인한다. 전체 통과에는 동결된 물리 기준과 에셋 검증,
실행 후 물리 검토가 모두 필요하다.
