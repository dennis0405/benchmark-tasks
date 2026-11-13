# MVP 구현 체크리스트

상태: **미착수 계획**. 모든 체크박스는 실제 구현·검증 완료 때만 변경한다.
순서·공통 실패 정책은 [plan.md](plan.md), 준비 조건은
[Brev 준비](../docs/design/brev-readiness.md)를 따른다.
예상 파일은 automation 아래의 제안 경로이며 아직 생성하지 않는다.
S는 1~2개, M은 3~5개 파일 안팎을 뜻한다. 초과할 작업은 구현 전에 분할한다.

## P0. Brev 사전 준비

### T01. 접속·스토리지 준비 보고 — S

로컬 CLI/VS Code와 원격 host/container·작업 공간·보존 목적지를 확인한다.

- [ ] 새 환경에 접속되고 GPU·mount·디스크 여유·실행 중 작업을 기록한다.
- [ ] 기존 main-thread 작업과 분리한 automation 작업 공간·보존 목적지를 특정한다.
- [ ] 원격 CLI 연결과 결과 파일 왕복을 확인한다.

검증: 비밀정보 없는 ReadinessReport와 임시 결과 파일의 hash 대조.
의존: 사용자 환경 지정. 예상 파일: readiness 보고 문서·비밀정보 없는 환경 manifest.

### T02. CPU 도구·인증 smoke test — M

선택한 에이전트·검색·GitHub 읽기 경로와 CPU 환경을 확인한다.

- [ ] 선택한 Codex 인증·모델로 schema를 따르는 작은 결과를 얻는다.
- [ ] 검색 원문 확보·GitHub 읽기를 실행하고 권한·rate limit 상태를 기록한다.
- [ ] 에이전트에 Dream/제출 credentials가 전달되지 않는 것을 시험한다.

검증: 최소 입력의 성공/만료 인증 실패 fixture. 실제 Dream·제출 요청은 없음.
의존: T01, 인증 방식 결정. 예상 파일: 도구 lock·인증 probe·probe 검사.

### T03. 고정 YAM GPU 런타임 smoke test — M

기존 작은 예제로 렌더·제어·관측 경로를 확인한다. 제출용 task 제작은 아니다.

- [ ] 선택한 버전에서 container GPU·reset·짧은 joint action이 동작한다.
- [ ] 필수 카메라·joint/gripper 상태·step/time과 짧은 영상을 기록한다.
- [ ] 이미지 digest·driver·Arena/YAM revision을 EnvironmentLock으로 남긴다.

검증: 실제 headless 렌더와 제어 추종·관측 shape 확인. Dream 호출 없음.
의존: T01. 예상 파일: runtime lock·diagnostic runner·관측 검사·보고서.

### Checkpoint A

- [ ] T01~T03 완료. 접근 가능, 실행 가능, 복구 목적지를 각각 확인했다.

## P1. 기록과 최소 전체 흐름

### T04. 실행 기록·소유권·일관된 backup — M

run/stage/attempt/journal을 저장하고 같은 실행의 중복 소유를 막는다.

- [ ] 중단 후 완료·진행·unknown 상태를 구분해 복구한다.
- [ ] 같은 run의 동시 시작을 거부하고 외부 요청 의도를 호출 전에 보존한다.
- [ ] DB·원본 참조의 일관된 backup/restore를 확인한다.

검증: 동시 시작·프로세스 종료·WAL을 포함한 새 공간 restore 시험.
의존: T01. 예상 파일: run_store·state_contracts·focused tests·backup adapter.

### T05. 에이전트 실행 어댑터 — M

역할별 명세와 입력 hash를 전달하고 구조화 결과를 검증한다.

- [ ] CLI/model/prompt/input 버전과 stdout·종료 상태를 보존한다.
- [ ] schema 오류·없는 evidence 참조·timeout을 통과시키지 않는다.
- [ ] 역할별 작업 공간·도구·환경변수 전달 범위를 검사한다.

검증: 정상 JSON, 비정상 JSON, 참조 누락, 제한 밖 실행 fixture.
의존: T02, T04. 예상 파일: agent_adapter·role contracts·tests.

### T06. fake 산출물로 승인 대기까지 — M

고정 fixture 하나를 조정기에 넣어 단계 경계와 무효화를 확인한다.

- [ ] 선택부터 fake 평가·보고·승인 대기까지 연결된다.
- [ ] 필수 증거 누락·입력 hash 변경·위조 승인으로 진행하지 않는다.
- [ ] 중단 후 완료 단계를 무작정 반복하지 않고 이어간다.

검증: 네트워크 없는 fake E2E와 단계별 중단/재개. 실제 작업 성공으로 보고하지 않음.
의존: T04~T05, T24. 예상 파일: coordinator·gate rules·fake stages·E2E tests.

### Checkpoint B

- [ ] 기록 복구·필수 gate·승인 대기까지 연결됐다. 외부 유료/쓰기 요청은 없었다.

## P2. 자료와 후보

### T07. GitHub 제출 목록 동기화 — M

전체 PR·공식 댓글·head revision을 수집하고 성공 snapshot만 확정한다.

- [ ] 전수 페이지·fork head·여러 task revision을 보존한다.
- [ ] 중간 페이지 실패 시 watermark를 앞으로 옮기지 않는다.
- [ ] 증분 재개·ETag·rate limit을 처리하고 PR 상태를 심사 상태와 구분한다.

검증: 다중 페이지·403·삭제 head·댓글만 변경된 fixture와 실제 읽기 표본 대조.
의존: T02, T04. 예상 파일: github collector·normalizer·fixtures·tests.

### T08. board 대조와 benchmark 범위 기록 — M

공식 board와 GitHub 관계·자기 팀·비교할 기존 benchmark를 카탈로그에 연결한다.

- [ ] 필요한 모든 상태/페이지의 coverage와 갱신 시각을 남긴다.
- [ ] 출처 충돌·매핑 실패·접근 실패를 unknown/incomplete로 남긴다.
- [ ] 누락된 필수 비교 범위로 novelty 통과를 내지 않는다.

검증: 실제 표본 원문과 정규화 결과 대조, parser 변경·인증 만료 fixture.
의존: T07, 비교 benchmark 범위 결정. 예상 파일: board adapter·reconciler·tests·scope manifest.

### T09. 산업 근거·에셋 수집의 한 경로 — M

선택한 조사 영역에서 매뉴얼/작업 근거와 관련 에셋을 함께 확보한다.

- [ ] 작업·산업 가치·조작 난도 주장을 원문 위치와 연결한다.
- [ ] 에셋 원본/수정본 hash·라이선스·실제 대응 장비·치수를 기록한다.
- [ ] 스캔/수집 실패·추정·재배포 불가를 누락 없이 표시한다.

검증: 산업 자료 한 묶음의 원문·페이지/프레임·AssetRecord 교차 확인.
의존: T02, T04~T05, 조사 범위 결정. 예상 파일: evidence ingest·card schemas·role prompt·tests.

### T10. 중복 검사와 단일 후보 선정 — M

전수 signature 대조·혼합 검색·원문 비교를 거쳐 여러 후보 중 하나를 고른다.

- [ ] 레이아웃/부품만 다른 중복과 물리 능력이 다른 사례를 구분한다.
- [ ] 모든 범주의 coverage·검색 누락·critic 이견을 보고하고 불확실 시 막는다.
- [ ] 여러 후보의 근거·난도·신규성·제작 위험과 선정/탈락 이유를 남긴다.

검증: 알려진 중복/비중복 개발 쌍과 별도 검증 쌍, top-k 밖 사례·자기 팀 사례.
의존: T05, T07~T09. 예상 파일: novelty pipeline·comparison contract·candidate selector·tests.

### Checkpoint C

- [ ] 근거·자료 범위·차별성이 확인된 후보 하나만 제작 대상으로 선정됐다.

## P3. 환경 준비

### T11. TaskContract와 계측 명세 — S

선정 후보를 관측 가능한 성공·실패·근사·시험 계획으로 고정한다.

- [ ] objective와 현실 작업의 대응·허용 근사·run limit가 설명된다.
- [ ] 파손·금지 행동·부분 점수와 필요한 센서/이벤트가 정해져 있다.
- [ ] 핵심 물리 허용 오차의 출처·가정과 검증 방식을 기록한다.

검증: 계약의 각 판정 조건을 계측·대조군 항목에 연결한 completeness 검사.
의존: T10. 예상 파일: 선택 task contract·measurement plan.

### T12. 첫 산업 장비 에셋 패키지 — M

한 후보에 필요한 핵심 장비를 실제 자료에 맞게 제작/변환한다.

- [ ] 치수·관절·질량·collision의 핵심 형상과 상대 참조가 유효하다.
- [ ] 형상 근사·원본/변환 hash·재사용 조건·수정 이력을 보존한다.
- [ ] 개발 서버 절대 경로 없이 패키지를 열 수 있다.

검증: 정적 검사·visual/collision 렌더와 실제 장비 도면 대조.
의존: T09, T11. 예상 파일: 해당 task assets·asset manifest·asset inspection 결과.

### T13. 환경 어댑터·구성요소 물리 검증 — M

고정 runtime에 장비·YAM·센서·reset을 연결하고 필요한 상호작용부터 확인한다.

- [ ] reset·무행동·핵심 접촉/관절 시험에서 안정성과 실제 상태를 기록한다.
- [ ] 정책 관측과 검사 관측을 분리하고 누락·stale·좌표/단위 오류를 검출한다.
- [ ] 필수 시험에 근거한 EnvironmentReadyReport를 얻는다.

검증: 산업 task의 구성요소 fixture·negative controls. 아직 전체 로봇 데모 아님.
의존: T03, T11~T12. 예상 파일: task adapter·measurement adapter·component tests·ready report.

### Checkpoint D

- [ ] 환경·핵심 물리·관측이 준비됐다. 이 판정 전에는 해당 task 데모를 개발하지 않았다.

## P4. 데모·채점과 재현

### T14. 채점 로직과 대조군 — M

계약의 상태·이벤트를 채점하고 편법·중간 위반·입력 무효를 검증한다.

- [ ] 경계값·시간·순서·위반 누적과 validity가 계약을 따른다.
- [ ] 무행동·미완료·관련 파손/금지 행동·관측 누락을 구분한다.
- [ ] 필요한 계측을 담은 raw trace로 점수를 재현한다.

검증: 로직 fixture와 실제/진단 환경의 negative controls. 정상 로봇 대조군은 T15에서 보완.
의존: T11, T13. 예상 파일: scorer·replay·scoring tests·control results.

### T15. 실제 YAM 성공 데모 — M

환경/채점의 계약을 유지하면서 실패 원인에 따라 controller를 개선한다.

- [ ] 로봇 action으로 run limit 내 full score를 달성한다.
- [ ] 깨끗한 reset 반복에서 영상·계측·점수와 종료 상태가 일치한다.
- [ ] object pose 주입 없이 실행하고 정상 trace를 재채점한다.

검증: 최종 구성 3회 성공을 기본 제안으로 검토. 미충족 시 원인별 수리·예외 정책 적용.
의존: T13~T14. 예상 파일: demo controller·최종 run manifests·demo/replay 보고.

### T16. standalone export와 새 컨테이너 재현 — M

제출할 파일 묶음만으로 동일 runtime에서 실행되는지 확인한다.

- [ ] private helper·숨은 asset·원래 경로 없이 export본이 실행된다.
- [ ] demo·scorer·카메라/trace를 재현하고 source/asset hash를 연결한다.
- [ ] 라이선스·블라인드 자료·미디어 형식 검사를 마친다.

검증: 새 컨테이너/작업 공간에 패키지와 선언된 runtime 의존성만 제공해 실행.
의존: T15. 예상 파일: export adapter·package manifest·reproduction test·report.

### Checkpoint E

- [ ] 환경·데모·채점·이식 검증이 끝났다. 실제 Dream은 아직 호출하지 않았다.

## P5. 평가·승인·제출의 모의 검증

### T17. Dream attempt 관리와 fake transport — M

기존 사용 이력·한 평가 실행·SDK 요청을 연결하고 자동 재평가를 차단한다.

- [ ] constructor/warmup/retry를 포함한 실제 요청 위치를 확인한다.
- [ ] 동시 예약·중단·timeout·세션 만료·새 이름으로 평가 제한을 우회하지 못한다.
- [ ] 새 VM 복구 후에도 기존 attempt·누적 사용 기록을 유지한다.

검증: 실제 키 없는 transport 대체·중단 주입. 진짜 요청을 무료 probe로 보내지 않음.
의존: T03~T04, 입력 어댑터 규약. 예상 파일: dream evaluator·attempt repository·fake transport tests.

### T18. 검토 보고와 명시적 승인 — M

실제/fixture 결과로 읽을 수 있는 보고서를 만들고 정확한 버전에 결정을 연결한다.

- [ ] 영상·실제 점수·근거·중복·한계·예정 제출 파일을 열람할 수 있다.
- [ ] digest·대상 repo/branch·허용 동작에 연결된 사용자 결정만 인정한다.
- [ ] 승인 후 소스/영상 변경·새 중복·에이전트 위조 승인을 차단한다.

검증: fake 결과 보고서의 로컬 검토와 승인/수정/보류·digest 변경 fixture.
의존: T06, 결과 manifest 규약. 예상 파일: report builder·HTML template·approval handler·tests.

### T19. 승인 후 제출·접수의 fake 경로 — M

외부 쓰기를 fake adapter로 바꿔 journal·업로드·PR·접수 복구를 검증한다.

- [ ] 승인 없음/불일치 시 업로드·push·PR 호출이 없다.
- [ ] 업로드/PR 응답 유실·push 후 중단에서 중복 요청을 막는다.
- [ ] pending 수정과 최초 접수를 구분하고 승인 파일 외 diff를 차단한다.

검증: local bare repo와 fake media/GitHub/board. 실제 대회 테스트 PR·미디어 업로드 없음.
의존: T04, T18. 예상 파일: publisher·receipt monitor·fake adapters·recovery tests.

### Checkpoint F

- [ ] 유료 평가·승인·외부 제출의 실패 방어를 실제 비용/접수 없이 검증했다.

## P6~P7. 실제 파일럿

### T20. 최종 점검과 평가 입력 동결 — S

현재 공식 규칙과 최종 환경·demo·scorer·export 증거를 확인한다.

- [ ] 입력/결과 hash와 모든 필수 gate가 현재 버전과 일치한다.
- [ ] 기존 Dream 이력·자격증명·카메라/상태/action 규약이 준비됐다.
- [ ] 필수 증거가 빠지면 실제 Dream을 시작하지 않는다.

검증: preflight와 평가 입력 snapshot. 실제 inference 요청은 T21에서만 수행.
의존: T10, T16~T19. 예상 파일: frozen RunInputManifest·preflight report.

### T21. 실제 Dream 1회와 최종 검토 자료 — S

고정한 task에서 한 평가를 실행하고 최신 중복·결과 보고를 작성한다.

- [ ] episode 안의 요청·action·영상·실제 score·종료 이유를 기록한다.
- [ ] 실패/중단 후 자동 재평가하지 않으며 최신 중복 검사를 다시 수행한다.
- [ ] 유효한 최종 ReleaseManifest와 보고서를 승인 대기로 둔다.

검증: 실제 결과의 validity·trace 재채점·영상 대조. 실행 무효면 예외 보고.
의존: T20, T25~T26, 기존 Dream 사용 이력 복구. 예상 파일: actual evaluation record·final report/manifest.

### T22. 승인한 버전의 실제 제출과 접수 확인 — S

사용자의 구체적 승인을 받은 뒤 실제 발행 경로를 처음 수행한다.

- [ ] 최신 중복·승인 digest를 확인하고 허용 메타데이터 변경만 반영한다.
- [ ] 별도 checkout에서 공식 업로드·fork push·단일 PR을 수행한다.
- [ ] PR 생성과 접수를 구분해 실제 bot/board 응답을 기록한다.

검증: 승인 자료 hash·최종 diff·media ID·commit SHA·PR·접수 연결 대조.
의존: T21, 사용자 제출 승인, 제출 인증. 예상 파일: publication journal·receipt record.

### T23. 보존·복구와 파일럿 평가 — S

새 작업 공간에서 실행 기록을 복구하고 자동화의 병목·개입을 정리한다.

- [ ] DB·원본/영상·환경 lock·평가/승인/제출 이력을 검증해 복구한다.
- [ ] 복구가 새 Dream 평가나 PR 생성을 유발하지 않는다.
- [ ] 단계별 시간·수리·사람 개입·미해결 한계와 후속 개선을 보고한다.

검증: 실제 VM 삭제 없이 별도 복구 공간에서 hash와 외부 요청 상태 대조.
의존: T22. 예상 파일: recovery report·pilot retrospective.

### Checkpoint G

- [ ] 승인 후 접수까지 실제 한 경로를 확인하고 복구했다. 채택 여부·일일 처리량은 별도 검증이다.

## 확정한 CLI·skill·start/stop·fork 보존의 추가 작업

번호는 기존 작업의 참조를 유지한다. 실행 위치는 각 의존관계를 따른다.

### T24. 로컬 CLI와 Brev 생명주기 — M

T04 뒤/T06 전에 로컬 control plane과 시작·상태·재개·승인·제출의 command 계약을 연결한다.

- [ ] 지정 instance만 start하고 status/approve/submit은 꺼진 VM에서도 동작한다.
- [ ] snapshot·보고서 반출 후 stop을 확인하며 stop 실패/상태 불명을 구분한다.
- [ ] 재개·로컬 단절·이미 running·다른 작업 점유를 처리한다.

검증: fake lifecycle 먼저, 실제 새 전용 instance start/stop·mount 재현. 다른 실험은 제외.
의존: T01, T04. 예상 파일: CLI·lifecycle adapter·control state·tests.

### T25. fork 작업 브랜치 checkpoint — M

T16 뒤/실제 최종 평가 전에 결과의 Git 보존과 검토 전 stop 순서를 연결한다.

- [ ] task마다 새 작업 브랜치를 만들고 수정은 같은 브랜치에 기록한다.
- [ ] 허용 결과 commit·origin push·remote SHA를 확인하고 큰 파일은 업로드 전에 보고한다.
- [ ] 보존용 branch와 승인 후 깨끗한 제출 checkout의 manifest 대응을 검증한다.

검증: local bare repo 실패 주입 후 허용한 fork에서 실제 작은 checkpoint 보존 확인.
의존: T16, T24, fork 발행 인증. 예상 파일: checkpoint publisher·manifest·tests·보존 보고.

### T26. 전용 운영 skill — S

CLI와 승인/제출 모의 검증 뒤 로컬 skill을 얇은 진입점으로 연결한다.

- [ ] 시작·상태·수정/재개·승인·제출 요청을 정확한 CLI 호출로 변환한다.
- [ ] 보고 digest·실제 상태·보존/stop 실패를 사용자에게 전달한다.
- [ ] skill 자체가 승인·재시도·새 평가·Git 결과를 임의로 생성하지 않는다.

검증: 실제 발행 없는 대표 요청·가짜 승인·꺼진 VM에서 조회 시나리오와 skill 형식 검사.
의존: T18~T19, T24~T25. 예상 파일: 로컬 운영 SKILL.md·필요한 command reference.
