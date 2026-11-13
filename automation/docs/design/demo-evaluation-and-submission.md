# 기술 설계 초안: 데모·채점·Dream·승인 후 제출

상태: 검토 초안, 구현 전. 작성일: 2026-11-13 (Asia/Seoul).
[환경 검증 계약](environment-and-validation.md)을 통과한 한 태스크를 입력으로 한다.
실제 에이전트·GPU·Dream 호출·업로드·PR 생성은 수행하지 않았다.

## 1. 역할·도구와 권한

| 역할/도구 | 책임 | 주요 도구·입력 | 산출물·변경 범위 |
|---|---|---|---|
| demo-developer | 목표를 물리적으로 수행하는 제어 개발 | YAM FK/IK·joint controller·환경 관측·영상·실패 기록 | demo controller만 기본 변경 |
| scoring-author | TaskContract의 성공·부분 성공·실패를 계측으로 구현 | 검사 관측·이벤트·JSON 계약 | scorer와 관측 요구 |
| scoring-reviewer | 편법·누락·과도한 조건과 구현 오류 검토 | 계약·scorer·대조군·재채점 | 수정 의견과 유효성 보고 |
| run-reviewer | 동작·점수·문서·fidelity의 일치 검토 | 렌더·원시 궤적·물리 검사 | 데모/최종 평가 검토 |
| submission-reviewer | 산업 가치·신규성·블라인드·재현·라이선스 점검 | 동결 결과물과 앞단 보고 | 사용자용 제출 전 보고서 |
| sim-runner / scorer-replay | rollout·기록·재채점 실행 | 고정 runtime, task와 정책 | 재현 가능한 실행 결과 |
| dream-evaluator | 최종 평가를 한 번 실행하고 API 이력 보존 | 동결 task, 공식 입력, 기존 예산 기록 | Dream 결과·종료 이유 |
| report-builder / approval-handler | 보고서 생성·사용자 결정과 대상 해시 확인 | ReleaseBundle, 로컬 검토 경로 | 승인/수정/보류 기록 |
| publisher / receipt-monitor | 승인 후 업로드·Git·PR·접수 상태 확인 | 승인 기록·고정 자료·제출 credentials | 외부 요청 이력·접수 결과 |

판단 역할은 앞단과 동일한 Codex 실행 어댑터를 우선 제안한다. 실행·재채점·
평가·승인·업로드·Git·PR은 조정기가 호출하는 일반 프로그램이다. LLM이 성공
수치나 사용자 승인을 직접 만들어 다음 단계로 보내지 못하게 한다.

Dream credentials는 최종 평가 도구, 제출 credentials는 publisher에서만
사용한다. 데모 제작 역할에는 임의의 유료 평가/제출 도구를 제공하지 않는다.

## 2. 데모 제어와 채점 인터페이스

초기 기준은 기존 양팔 YAM이다. 고정한 어댑터의 14차원 absolute joint/gripper
targets와 limit·그리퍼 정규화를 확인한다. 제어 loop는 단순 waypoint부터 시작할
수 있으나 task가 접촉 피드백·힘 제한·양팔 협응을 요구하면 해당 controller를
구현·검증해야 한다. waypoint 생성이나 IK 수렴만으로 실제 수행을 인정하지 않는다.

단일 demo-developer가 실패 원인에 근거해 제어를 수정하고, sim-runner가 새
run ID에서 시험한다. 결과를 읽고 실패 분류를 기록한 뒤 관련 단계만 수정한다.
환경과 채점까지 무제한으로 바꾸는 하나의 성공 최적화 loop를 만들지 않는다.

| 경계 | 입력 | 출력·기록 |
|---|---|---|
| controller | 현재 step, 관측, controller state | 로봇 action, 계획 단계·제어 종료 이유 |
| environment | 검증된 action | 새 정책 관측·검사 관측·물리 이벤트 |
| scorer | step별 검사 관측·누적 이벤트 | score, success, valid, 부분 조건, 위반, 종료 근거 |
| recorder | action·관측·metrics·영상 | step/time 기준으로 연결된 궤적·영상·run result |

정책 관측과 검사 관측은 분리한다. 스크립트는 simulator state를 사용할 수 있지만
실행 도중 물체 pose/joint를 직접 써서 성공시키거나 센서를 가짜 값으로 채우면 안 된다.
객체 초기화·진단 fixture와 실제 rollout을 명시적으로 구분한다.

## 3. 채점 개발·검증

scoring-author는 TaskContract를 바탕으로 상태와 전이·부분 점수를 작성한다.
과제에 따라 최종 상태 조건과 전 구간 유지 조건을 구분한다. 무조건 일정 시간
유지나 일정한 힘 임계값을 붙이지 않고 실제 결과를 측정할 수 있는 기준을 둔다.

필수 검사 묶음:

1. **순수 로직 검사**: 경계값, 단계 순서, 누적 실패, 유지 시간, step 중복/누락,
   입력 무효의 처리. 작은 synthetic state는 로직 시험이며 물리 데모가 아니다.
2. **물리 대조군**: 무행동, 근접하지만 미완료, 정상 성공, 관련 파손/금지 행동,
   관측 누락이 의도한 판정을 받는지 실제/진단 환경에서 확인한다.
3. **재채점**: 동일 raw trace와 계측 이벤트를 같은 버전의 scorer에 넣어 실행 시
   score·위반·성공 시점과 일치하는지 검사한다. 재채점이 실패하면 진실을 다시
   생성한 것이 아니라 관측이 부족할 수도 있으므로 원인을 분리한다.
4. **영상 대조**: 관측과 렌더가 가리키는 상태가 맞고 최종 결과가 식별되는지 본다.
   시각 검토만으로 접촉력·짧은 위반·관통이 없었다고 판정하지 않는다.

관측 불명은 `valid=false`로 기록한다. 제출할 숫자를 만들기 위해 임의의 0이나
1로 바꿔 유효한 점수처럼 다루지 않는다. SDK가 숫자를 강제하는 경우 내부 validity와
오류 상태를 반드시 함께 보존하고 최종 검증에서 차단한다.

정상 demo는 full score를 run limit 안에서 달성해야 한다. 최종 고정 구성의
깨끗한 reset 재실행을 확인하는 안을 제안한다. 구체적인 반복 수와 물리 허용
오차는 후속 검토로 정한다. Dream 성능에 맞춰 scorer를 완화하지 않는다.

## 4. 실행 기록과 변경 영향

RunRecord에는 task lineage ID, run ID, source/asset/config/scorer/controller 해시,
EnvironmentLock, seed, 정책, step/time, 상태·metrics·유효성·이벤트, 기록 파일,
실제 종료 이유를 담는다. 영상과 trajectory의 시작·끝 및 마지막 step을 연결한다.
멀티 카메라 중 손목 영상 누락 등을 성공한 영상 한 개로 숨기지 않는다.

| 변경 | 무효화·다시 검증할 대상 |
|---|---|
| 장비 형상·물리·배치·센서·runtime | 환경 준비·대조군·데모 및 해당 신규성/재현성 근거 |
| controller | 데모·행동 한도·실패 검사; objective 변경 여부 확인 |
| scorer | 로직·대조군·trace 재채점, 필요하면 관측을 갖춘 새 demo |
| objective·성공 조건 | 산업 근거·중복·계약 및 의존하는 모든 결과 |
| 영상 인코딩·결과 수치의 문서 기입 | 원시 기록 보존, media·문서 일치·블라인드 검사 |

scorer 재채점은 저장된 과거 상태의 평가다. 수정된 환경에서 로봇을 다시 움직인
것과 혼동하지 않는다. Dream 이후 의미 변경이 생기면 해당 결과와의 관계를
예외 보고하고 자동 재평가하지 않는다.

## 5. 최종 평가 전 스냅샷

Dream 전에 task의 실행 입력을 고정하고 실제 사용할 portable runtime까지
검증한다. private skill, 원래 서버의 Python 파일·절대 경로에 의존하지 않게 한다.
공통 runtime을 task 패키지에 export할 경우 그 export본으로 검증해야 한다.

두 manifest를 구분한다.

- **RunInputManifest**: 코드·에셋·계약·카메라·정책 관측·제어/채점 설정과 runtime.
  demo와 Dream이 어떤 task/관측/채점을 사용했는지 비교하는 기준이다.
- **ReleaseManifest**: 최종 제출 문서·영상·실제 결과·라이선스까지 포함한 사용자
  승인 대상. 추가된 결과/미디어 참조와 실행 입력을 구분해서 검증한다.

동결 이후 허용되는 자동 변경은 실제 결과의 기입, 검증된 원본을 유지한 인코딩,
media ID·PR URL 등 제출 메타데이터 연결뿐이다. setup·scoring·fidelity의 의미를
바꾸는 수정은 단순 문서 변경이라고 자동 허용하지 않는다.

## 6. Dream 최종 평가 도구

첫 후보는 고정한 `inspect-robots-dream` 정책 어댑터와 기존 YAM/공용 recorder의
결합이다. 각 어댑터가 실제로 지원하는 reset, camera calibration, action, 종료,
per-step logging을 시험한 뒤 선택한다. upstream Inspect Robots의 기본 로그가
task의 모든 계측을 자동 보존한다고 가정하지 않는다.

확인한 고정 revision의 YAM 경로는 20 Hz, 3개 RGB 카메라와 실제 14 joint/gripper
값을 사용한다. 정확한 키·크기·모델 옵션은 고정 어댑터에서 검증한다. 추가 overview
카메라는 보고서용으로 쓸 수 있지만 모델 입력에 자동 추가하지 않는다.
object pose·정답 위치·점수·스크립트 계획·원문 매뉴얼을 Dream에 주지 않는다.

### 실행 1회 계약

```text
offline-preflight → ready → attempt-reserved → running
                                      ├→ completed: 성공 또는 성능 실패
                                      ├→ interrupted: API/기록/실행 중단
                                      └→ unknown: 외부 처리 여부 불명
```

- stable task lineage에 평가 attempt를 원자적으로 예약하고 중복 시작을 막는다.
  DB unique 제약과 실행 소유권을 사용한다. 새 task 이름/버전으로 제한을 초기화하지 않는다.
- offline-preflight에서는 파일·설정·로컬 패키지·카메라 기록 가능성을 확인한다.
  모델에 실제 요청하는 warmup·dry-run·SDK 초기화는 무료 사전 검사라고 가정하지 않는다.
- 한 attempt 안에서 여러 action chunk 요청은 허용한다. 요청별 상태·응답·소요량,
  내부 retry와 실제 네트워크 시도를 기록한다. 반환 후 실제 수행한 로봇 action도 남긴다.
- 기존 누적 사용액과 진행 중 예약을 포함한 단일 권위 있는 비용 기록을 사용한다.
  새 VM마다 별도 초기 잔액을 만들지 않는다. 여러 실행자가 있으면 같은 예약
  권위로 모아 관리한다. provider 사용액이 불명확하면 보수적으로 보류한다.
- timeout/세션 만료 시 SDK가 숨겨서 새 평가를 만드는 동작을 확인한다. 같은
  논리 요청의 중복 외부 처리 여부를 모르면 무한 재시도하지 않고 중단·예외 보고한다.
- 전체 평가의 성능 실패·중단·unknown 어느 경우에도 자동으로 두 번째 평가를
  시작하지 않는다. 이미 진행 중인 프로세스에 재접속하는 것은 새 실행과 구분한다.

상태 불명 후 사용자가 예외적으로 추가 실행을 결정할 여지는 자동화가 추정하지
않는다. 그 시점의 명시적인 지시와 원래 시도 이력이 필요하다. 기본 정책은 재평가 금지다.

결과는 실제 score·종료 이유·영상과 함께 남긴다. 모델 거부·행동 전 실패 등은
실제 에러를 보존한다. 예외 제출 경로의 공식 지원 여부가 불명확하면 수동 검토로
보내며, 실행하지 않은 결과를 만들어 넣거나 성공 demo를 모델 영상으로 대체하지 않는다.

## 7. 사용자 보고와 승인

MVP 제안은 **정적 HTML 보고서 + 기계 판독용 JSON + 명시적 승인 도구**다.
로컬 Brev CLI/SSH의 포트 전달 또는 결과 파일 반출 후 VS Code/브라우저로 본다.
이것은 구현할 인터페이스 제안이며 현재 보고 서버나 승인 CLI는 없다.
검토 자료·일관된 복구 snapshot·fork 작업 브랜치 origin push를 확인한 뒤
instance를 stop하고 로컬 자료로 검토한다. 승인·제출은 로컬 CLI, 수정 요청은
instance start·resume으로 처리한다. [생명주기 설계](cli-skill-and-lifecycle.md)를 따른다.
Slack/메일 발송·외부 공개 대시보드를 기본으로 요구하지 않는다.

ReportBundle에는 산업 근거, 장비 setup, 조작 난도, 최신 중복 검사, demo와 Dream
전체 영상·최종 상태, 실제 점수, 대조군·재현 결과, fidelity/미해결 한계, 출처·
라이선스, 제출 목적지와 제출 예정 자료를 포함한다. 작업자의 검사 영상과
블라인드 제출용 순수 camera 영상을 분리한다.

승인 기록은 decision, 사용자, 시각, task lineage, ReleaseManifest digest,
관련 NoveltyReport ID, 제출 대상 repo/base/branch와 수행할 동작을 참조한다.
에이전트나 report-builder는 승인 값을 생성할 권한을 갖지 않는다.

환경 준비 중 확인한 Inspect Robots 1.0의 record에도 `approval` 이벤트가 있지만,
이는 action guard의 수정/허용 판정이다. 제출 승인으로 읽지 않는다. 제출용 승인
계약은 목적·작성 주체·사용자·ReleaseManifest digest를 검사하며 SDK 이벤트나
모델 출력의 같은 이름 필드로 대체할 수 없다.

- 대화의 일반적인 긍정 표현을 미래의 모든 결과에 대한 제출 승인으로 해석하지 않는다.
- 사용자가 본 결과물 digest가 변경되면 의미 변화와 허용 메타데이터 변경을 구분한다.
- 새로운 중복 차단 사유가 발견되면 기존 승인이 있어도 제출 대기/재검토한다.
- 승인 대기가 길어졌다면 카탈로그를 갱신한다. task 내용이 그대로이고 차단 사유가
  없다면 단순 자료 갱신만으로 불필요하게 내용을 수정하지 않는다.

제출용 인코딩과 파일 검사는 승인 전에 마친다. 승인 후 허용하는 변경은 미리
명시한 media ID·접수/PR 참조 필드의 연결로 한정하고, 업로드할 파일 해시는
승인 당시와 같아야 한다. 최종 커밋은 원래 승인 digest와 이 메타데이터 변경의
대응표로 검증한다. 이를 이용해 소스·scorer·영상 내용을 함께 바꾸지 못하게 한다.

## 8. 승인 후 제출의 실행 순서

1. **최종 사전 검사**: 승인 digest, 최신 중복, 현재 공식 schema·규칙, 등록된 제출자,
   run score·run limit·라이선스·필수 파일·비밀정보/블라인드 여부를 확인한다.
2. **미디어 검증**: ffprobe로 codec·container·크기·duration·재생 가능성을 검사하고
   setup/최종 프레임·전체 동작과 보고한 score를 대조한다. 수정 인코딩은 원본을 보존한다.
3. **업로드**: 고정·검토한 공식 `tools/upload_media.py`를 사용한다. media ID와
   원본/업로드 파일 해시, 요청 전후 상태를 기록한다. uploader는 task.json의 media를
   변경하므로 허용한 메타데이터 외 변경이 없는지 확인한다.
4. **제출 checkout 준비**: 별도의 제출 작업 공간에서 최신 upstream/main 기준으로
   한 task 폴더와 필요한 runtime·에셋을 넣는다. 진행 중인 제작 repo의 branch를
   전환하거나 다른 task를 stage하지 않는다. 승인한 내용이 diff에 정확히 들어가는지 본다.
5. **commit·push**: 승인된 fork origin을 대상으로 수행하고 SHA를 기록한다.
6. **PR 생성/확인**: GitHub CLI 또는 REST로 base/head·task ID의 기존 PR을 먼저
   확인한다. 최초 제출은 하나의 PR을 만들고 응답 URL·number·head SHA를 기록한다.
7. **접수 확인**: 공식 bot 댓글/board에서 접수 ID와 심사 대기를 확인한다.
   PR 생성 성공만으로 접수를 완료했다고 보고하지 않는다.

현재 확인한 공식 uploader는 H.264 MP4, 영상당 최대 50 MB, screenshot 최대
10 MB를 안내하고, 모델이 만점이 아니면 full-score demo를 요구한다. 제한과
필드·category·제출 빈도는 발행 시 공식 버전을 확인한다. task run limit는
실제 조작에 맞게 정하며 예제의 2,400 step을 공통 필수값으로 고정하지 않는다.

미디어 업로드나 PR 생성의 서버 측 idempotency key 지원을 확인하지 않았으므로
지원하는 것처럼 설계하지 않는다. 작업 journal에 payload digest와 호출 의도를
먼저 남기고, 중단 후 외부 상태를 조회한다. media ID를 복구할 공식 조회 경로가
없으면 `unknown`으로 멈춰 사용자에게 알린다. PR은 base/head와 기존 응답을
확인한 뒤 이어가며, 재개했다고 새 PR을 만들지 않는다.

## 9. 제출 상태와 피드백

```text
awaiting-approval → approved → uploading → pushed → pr-created
                                                   ↓
                                 received / intake-blocked / unknown
                                                   ↓
                             pending → accepted / revision-needed / rejected
```

일일 한도·인증·JSON 오류 등 접수 차단은 심사 탈락과 구분한다. 현재 공식 규칙에
맞춰 수정·재시도·대기할 수 있지만, 의미 있는 제출물 변경은 새 검증/승인을 거친다.
심사 대기 중 push가 새 자료로 반영된다고 가정하지 않는다.

수정 요청에는 피드백과 수정의 대응표를 만들고 검증·사용자 승인을 거친다.
pending task 철회나 close/reopen·새 PR로 재제출하는 동작은 최초 제출 승인의
범위에 자동 포함하지 않는다. 해당 외부 동작을 명시한 사용자 지시가 필요하다.
접수 확인·상태 조회는 가능하지만 자기 PR을 자동 merge하지 않는다.

심사 의견은 출처·시각·task revision·문제·수정·검증 결과에 연결해 앞단 후보
선별과 회귀 검사에 반영한다. 한 번의 채택으로 비슷한 task를 대량 생성하거나,
단일 심사 의견을 모든 task의 보편적인 물리 규칙으로 바꾸지 않는다.

## 10. 구현 전 시험할 실패 상황

관측 누락을 성공으로 처리, 초기 상태만 성공, 중간 위반 후 원상 복구, 제어 action의
NaN/범위 이탈, 이미지·trace 누락, 영상 인코딩 실패, scorer와 replay 불일치,
Dream constructor/transport의 숨은 요청·retry, attempt 예약 후 프로세스 중단,
부분 응답 후 접속 단절, 승인 후 source 변경, 새 중복 발견, 업로드 응답 유실,
push 후 중단, PR 생성 응답 유실, 접수 차단, pending 중 수정 요청을 검증한다.

시험은 fake transport/fixture로 먼저 수행할 수 있으나 실제 GPU 데모·Dream·
승인 후 제출 경로를 검증했다는 뜻은 아니다. 이후 실제 파일럿 결과를 별도로 남긴다.

## 확인한 출처와 한계

- [현재 공식 제출 가이드](https://github.com/robocurve/benchmark-tasks/blob/main/AGENTS.md)
- [공식 uploader](https://github.com/robocurve/benchmark-tasks/blob/main/tools/upload_media.py)
- [고정 YAM revision](https://github.com/kairosiann/yam-isaac-lab-arena/blob/0059d02a1a91bc6834c1d0846022891df0c78730/README.md)
- [고정 Dream 어댑터](https://github.com/kairosiann/inspect-robots-dream/blob/b5e936a201cec66c54c297e522c920d98445e339/README.md)
- [ffprobe](https://www.ffmpeg.org/ffprobe.html)
- [GitHub PR API](https://docs.github.com/en/rest/pulls/pulls)

YAM/Dream 어댑터는 커뮤니티 구현이며 Dream Labs의 공식 SDK 계약 자체와
동일하지 않다. 실제 설치 SDK의 세션·재시도·계측을 확인해야 한다. 이 문서의
도구 설계만으로 실제 평가·과금·접수 복구가 검증됐다고 주장하지 않는다.
