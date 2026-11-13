# CLI·전용 skill·Brev 생명주기와 결과 보존

상태: 실행 기준과 생명주기·fork 보존 방향은 사용자 확정. 명령 문법은 설계 초안.
최초 설계: 2026-11-13 (Asia/Seoul). 2026-11-14 CLI·skill을 구현했다. 현재 실제 명령과 제약은 [README](../../README.md)를 따른다.

## 1. 확정한 동작

```text
로컬 skill/대화 → 로컬 CLI
  → 지정한 기존 Brev 인스턴스 start·ready 확인
  → 원격 제작·시험·Dream 최종 1회
  → 검토 자료 + 일관된 실행 snapshot + fork 작업 브랜치 origin push 확인
  → 원격 작업 종료 확인 → 해당 인스턴스 stop·상태 확인
  → 로컬 자료로 제출 전 사용자 검토
       ├ 승인 → 로컬에서 최신 중복/규칙 확인 → 실제 제출·접수 확인
       └ 수정 요청 → 같은 run/branch로 instance start → 필요한 단계만 재개
```

검토 대기 중 GPU를 유지하지 않는다. 작업 결과의 fork push와 대회 upstream
PR 제출은 다른 동작이다. 사용자가 승인한 것은 제출 전 결과 보존용 origin
push까지 포함한다. 대회 PR·미디어 업로드는 최종 승인 뒤에 수행한다.
수정 요청이 최종 Dream 이후라면 기존 평가 이력을 유지하고 자동 재평가하지 않는다.

## 2. CLI 명령 계약 제안

`rcauto`는 제안한 실행 파일 이름이며 현재 사용할 수 있는 명령이 아니다.
명령은 Mac에서 실행하고 GPU 작업만 원격 worker로 전달한다.

| 명령 예시 | 책임 | Brev 동작 |
|---|---|---|
| `rcauto start --instance NAME --scope SCOPE` | 새 run 예약, 인스턴스/소유권 확인, 제작 시작 | stopped이면 start, 이미 running이면 다른 작업 점유 확인 후 연결 |
| `rcauto status RUN_ID` | 마지막 checkpoint·현재 작업·예외·보존 SHA·승인·접수 출력 | 기본은 로컬 기록만 조회. 갱신해도 자동 start하지 않음 |
| `rcauto resume RUN_ID --request FILE` | 중단 복구 또는 사용자 수정 요구를 기록하고 관련 단계 재개 | 같은 instance start, 환경/asset/hash·기존 외부 요청 상태 재검사 |
| `rcauto approve RUN_ID --digest DIGEST` | 사용자의 제출 승인을 정확한 ReleaseManifest에 기록 | GPU 시작 없음. 승인 자체는 미디어 업로드/PR 생성과 구분 |
| `rcauto submit RUN_ID --approval APPROVAL_ID` | 승인·최신 중복·공식 규칙 확인 후 업로드·fork 발행·PR·접수 | 기본은 로컬 CPU. GPU instance를 제출만을 위해 켜지 않음 |
| `rcauto stop RUN_ID` | 제어된 checkpoint·보존 후 해당 run 자원 stop | 즉시 kill이 아님. 불명확한 Dream/외부 요청은 unknown으로 보존 |

`start`는 사전 등록한 instance name/ID만 받는다. Brev가 start 인수를 URL로
해석해 새 VM을 만드는 경로를 자동으로 사용하지 않는다. VM 신규 생성은 별도
환경 준비 작업이다. `--all` stop이나 다른 run/기존 실험 인스턴스 stop은 없다.

각 명령은 run ID, task lineage, 상태, checkpoint, instance ID/상태, 최신
commit/보존 상태, 승인 상태, 다음 행동을 출력한다. JSON 출력도 같은 계약을
사용한다. `awaiting-review`, `blocked`, `unknown`은 성공 완료와 구분한다.
같은 명령 재실행이 새 Dream·새 브랜치·새 PR을 만드는 트리거가 되지 않게 한다.

## 3. 전용 skill의 역할

전용 skill은 **로컬에서 발견·호출되는 얇은 CLI 진입점**이다. 대화에서 시작,
상태 조회, 수정 요구, 승인·제출 의도를 구분해 CLI 인수로 전달하고 결과·보고서를
보여준다. skill 이름은 `robocurve-workflow`를 제안한다.

- 비즈니스 상태·재시도·GPU 관리·Git 규칙을 skill 안에 중복 구현하지 않는다.
- 사용자 수정 요구는 파일/구조화 입력으로 남겨 `resume`에 전달한다.
- 최종 보고 digest를 보여준 뒤 명시적으로 승인한 경우 `approve`를 호출한다.
  사용자가 그 결과의 제출을 승인했다면 이어서 `submit`까지 진행할 수 있다.
- 설계에 대한 긍정·skill 호출·작업 시작을 최종 결과물 승인으로 대체하지 않는다.
- 예외를 요약하고 멈춘 상태를 표시한다. 에이전트가 임의로 승인 값을 작성하지 않는다.

지금 사용하는 idea-refine/interview-me와 이 운영 skill은 다른 목적이다.
운영 CLI가 검증된 뒤 skill을 작성·설치한다. 현재 SKILL.md를 만들지 않는다.

## 4. 로컬 control plane과 원격 worker

Mac에는 run index·instance binding·승인·발행 journal·수신한 ReleaseBundle을
보존한다. 시작·상태·승인·제출은 VM이 꺼져 있어도 사용할 수 있게 한다.
원격에는 제작/물리 실행 worker와 현재 run 기록을 둔다. checkpoint snapshot의
revision과 소유권을 대조하고 양쪽이 같은 단계를 동시에 실행하지 않게 한다.

검토 대기 전 최신 원문/중복 카탈로그와 portable report도 내려받는다. 제출 직전
신규성 갱신은 로컬 CPU adapter/판단 도구로 수행할 수 있어야 한다. 필요한 자료나
인증이 로컬에 없으면 준비/보류 상태를 표시하고 GPU VM을 몰래 start하지 않는다.

로컬 생명주기 supervisor는 터미널 연결과 분리한다. **초기 MVP에서는 Mac이
네트워크 연결 상태로 supervisor를 실행해야 자동 stop을 완료할 수 있다.**
Mac 절전/전원 종료까지 견디는 원격 종료 guard는 Brev의 지원·인증/권한을 확인한
후 보완할 항목이다. 로컬 단절 후에도 이미 stop됐다고 표시하지 않는다.
재접속 시 ready checkpoint·실제 worker/instance 상태를 대조해 stop을 마무리한다.

## 5. 검토 대기 전 checkpoint와 stop

1. worker의 안전한 단계 경계에서 기록·영상·trace를 flush하고 실행 중 자식 작업을 정리한다.
2. DB의 일관된 snapshot, 소스/asset/runtime manifest, 보고서·평가/요청 이력을 보존한다.
3. fork 작업 브랜치에 허용 자료를 commit·origin push하고 remote SHA를 확인한다.
4. 로컬 독립 보존본의 hash·필수 파일과 Git에 없는 자료의 참조도 확인한다.
5. 정확한 instance ID에 stop을 요청하고 provider의 실제 stopped 상태를 확인한다.
6. 결과물은 `awaiting-review`, compute는 `stopped`로 각각 기록한다.

`stop-requested`, `stopping`, `stopped`, `stop-failed/unknown`을 구분한다.
작업 완료·보고서 생성만으로 GPU가 꺼졌다고 보고하지 않는다. stop 지원 여부와
stop/start 후 mount·환경 재현을 실제 새 전용 인스턴스에서 검증한다.

실제 Crusoe 준비 시험에서 stop 명령은 먼저 반환했지만 STOPPING은 몇 분
유지됐다. 생명주기 adapter는 이 구간을 명시적으로 표시하고 bounded polling과
재접속 시 상태 대조를 지원해야 한다. 대기 시간이 끝나도 완료로 바꾸거나
동일 stop 요청·새 start/새 VM 생성을 무작정 반복하지 않는다.

origin push가 실패했어도 일관된 로컬 보존본을 확보했다면 GPU를 stop하고 CPU에서
보존 실패를 수리할 수 있다. 이 경우 제출 준비 완료로 표시하지 않는다. 보존 자체가
실패하면 제한된 복구 후 예외를 보고하고 임의로 데이터를 버리지 않는다.
예외로 사람이 판단하기를 기다리는 동안도 안전한 checkpoint/보존이 가능하면 stop한다.

## 6. fork 작업 브랜치와 제출 브랜치

현재 로컬 benchmark fork의 origin은 `git@github.com:dennis0405/benchmark-tasks.git`,
upstream은 `git@github.com:robocurve/benchmark-tasks.git`으로 읽어서 확인했다.
현재 `feature/close-snack-bag`는 별도 실험 브랜치이므로 전환하거나 수정하지 않는다.

- 새 task lineage마다 `automation/<task-slug>-<short-id>` 같은 새 작업 브랜치를 만든다.
  이는 명명 제안이며 다른 기존 branch를 재사용/덮어쓰지 않는다.
- 해당 브랜치에 task 환경·에셋·controller·scorer·지시문·라이선스·재현 방법,
  환경 lock·실제 결과 요약과 필요한 최종 미디어를 보존한다.
- 수정/재개는 같은 브랜치에 후속 commit으로 남기며 task/run/commit SHA를 연결한다.
- token·browser session·Codex 로그인 cache·개인 검사 로그는 Git에 포함하지 않는다.
- 큰 영상·raw trace·CAD 원본은 파일/총량을 먼저 측정한다. GitHub 한도에 걸리거나
  보존량이 큰 경우 push 전에 사용자에게 알리고 원본 저장 경로·hash를 manifest에 남긴다.
  Git LFS/외부 저장소 지원·권한을 확인하지 않고 자동 의존성으로 만들지 않는다.

작업 브랜치는 보존 목적이므로 내부 보고서가 있을 수 있다. 실제 upstream 제출은
별도 깨끗한 checkout/제출 브랜치에 승인한 task 폴더와 허용 자료만 구성한다.
그 브랜치도 fork origin에 push하되 내부 실행 DB·사적 보고·다른 task를 넣지 않는다.
두 브랜치와 RunInputManifest/ReleaseManifest·최종 SHA의 대응을 기록한다.

## 7. 수정과 재개

수정 요청은 대상 report digest·요청 내용과 함께 기록한다. instance start 후
mount·runtime·source/asset hash를 확인하고 checkpoint를 복원한다. stale 승인·
의존 검증을 무효화하고 필요한 단계만 다시 수행한다. 새 실행 이름으로 Dream
평가 제한을 초기화하지 않는다. 작업 완료 후 다시 보존→stop→사용자 검토로 돌아간다.

여기서 사용자의 승인/반려는 **제출 전 검토**다. 실제 대회에서 이미 pending인
PR의 철회·재제출이나 공식 심사 반려 처리는 기존 외부 제출 정책을 별도로 따른다.

## 8. 추가 검증 항목

stopped start·이미 running·다른 작업 점유·ready 지연·IP 변경·mount 누락,
검토 보고서 수신 전후 단절·origin push 실패·stop 요청 응답 유실·Mac 절전,
꺼진 VM에서 status/approve/submit·resume 시 기존 branch 유지·최종 평가 이력 복구를
시험한다. 실제 start/stop 시험은 새 전용 인스턴스에만 수행한다.

참고: [Brev 인스턴스 관리](https://docs.nvidia.com/brev/cli/instance-management),
[CLI 연결·복사](https://docs.nvidia.com/brev/cli/connectivity).
설치된 CLI v0.6.337의 help도 대조했다. 일부 provider는 start/stop 지원이 다르다.
