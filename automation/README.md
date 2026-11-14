# RoboCurve automation

프롬프트로 산업 태스크 후보를 조사하고 하나를 제작하여 **제출 전 검토**까지 진행하는 CLI MVP다.
실제 제출은 사용자가 결과물을 검토하고 승인한 뒤 별도 명령으로 수행한다.
2026-11-14 기준 CLI·skill·Brev 실행 경로를 구현했다. 새 산업 태스크의 파일럿은 다음 사용자
프롬프트를 기다린다. 모의 실행·기존 YAM 예제 성공과 신규 태스크 채택을 구분한다.

```text
사용자 프롬프트 → 실행 기록·목록 갱신 → 지정 Brev start
 → 산업 원문·에셋 조사 → 후보 다수 비교·선정 → 전수 중복 검사
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

닫힌 미병합 타 팀 PR도 비교 기록에 남긴다. 다만 내용 없는 테스트·철회 기록의 정보 부족은
현재 접수된 태스크의 중복 주장으로 취급하지 않는다. 이 불확실성은 `inactive_uncertainty`에
별도로 남긴다. 실제 중복 판정은 닫힌 기록도 차단하며, merged·open·우리 팀·상태 불명의
정보 부족은 계속 차단한다.

## 설계·완료 범위

- [현재 구현·검증 현황](tasks/implementation-status.md)

- [사용자가 확정한 제작 의도](docs/intent/robocurve-automation.md)
- [선택한 방향과 MVP 범위](docs/ideas/industrial-task-pipeline.md)
- [운영 설계 초안: 단계별 산출물·통과 조건·실패 처리](docs/design/mvp-workflow.md)
- [기술 설계 초안: 에이전트·도구·Brev 실행 위치](docs/design/technical-architecture.md)
- [상세 기술 설계: 자료·에셋 수집, 제출 동기화와 중복 검사](docs/design/sources-and-novelty.md)
- [상세 기술 설계: 환경·에셋 제작과 물리 검증](docs/design/environment-and-validation.md)
- [상세 기술 설계: 데모·채점·Dream·승인 후 제출](docs/design/demo-evaluation-and-submission.md)
- [MVP 구현 순서·단계별 검증 계획](tasks/plan.md)
- [MVP 상세 검증 체크리스트](tasks/todo.md)
- [Brev 사전 준비: 도구·인증·사용자 결정](docs/design/brev-readiness.md)
- [CLI·전용 skill·Brev start/stop·fork 결과 보존](docs/design/cli-skill-and-lifecycle.md)
- [전용 Brev 환경 준비 점검 기록](docs/design/environment-bootstrap-status.md)
- [실제 환경 설정·의존성·검증 자료](environment/README.md)

범위는 실제 산업 근거에 기반한 AI timelines 태스크의 후보 다수 검토·단일
태스크 제작·승인 후 제출이다. 강체를 기본으로 하되 검증 가능한 기존 연성
에셋을 허용한다. [실행 명령 근거](https://learn.chatgpt.com/docs/developer-commands#codex-exec),
[권한 profile](https://learn.chatgpt.com/docs/permissions),
[대회 규칙](https://github.com/robocurve/benchmark-tasks/blob/main/AGENTS.md)을 참고한다.
