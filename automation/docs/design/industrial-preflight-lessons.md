# 산업 자료와 실행기 사전 점검에서 확인한 사항

2026-11-14 첫 산업 태스크 제작 실행에서 확인했다. 아래의 자료 수집·CLI 오류는
재현하고 수정했으며, 댐퍼의 실제 물리·YAM 성공·Dream 결과를 뜻하지 않는다.
태스크 검증 결과는 해당 실행의 보고서와 원시 측정 자료를 확인한다.

## 원문과 형상

- 검색 도구가 읽은 PDF와 실제로 확보한 원문은 구분한다. 제조사 URL이 HTTP 200을
  반환해도 다운로드 차단 HTML일 수 있다. 인용은 수집에 성공한 원문의 ID·해시·위치에 연결한다.
- 원래 제조사가 작성한 문서를 유통사 사이트에서 확보했다면 작성자, 호스팅 위치,
  판본을 구분한다. 배포 허가를 추정하지 않고 원문·도면·CAD를 제출물에 복사하지 않는다.
  사실인 치수와 작동 원리를 참고해 독립적으로 만든 에셋은 출처와 가정을 기록한다.
- 표의 텍스트 추출만으로 기하를 결정하지 않는다. ECO OEM .35M 도면에서는
  `D=4.0 mm`가 피스톤 로드 지름이고 `G=11.2 mm`가 조정 노브 지름이다.
  `H=14.5 mm`는 축 방향 치수이며, `A=100.6 mm`는 비버튼형 전체 길이다.
  표 머리글·기호·실제 그림을 함께 확인해야 한다.
- 선택 요약표와 상세 기술표의 수치가 다른 경우가 있다. 한 판본의 상세 모델 행을
  기준으로 행정·질량·복귀력·에너지·속도·조정 범위를 일관되게 고정한다.
  HDA 계열의 조정 기구를 ECO OEM 형상에 붙이는 식으로 다른 계열을 섞지 않는다.
- 제조사가 최소 작동 에너지나 속도별 설정 한도를 제시하면 최대 정격 이하라는 이유만으로
  통과시키지 않는다. 하중의 운동에너지·구동 일·스프링 복귀 일까지 검토한다.

자료: [Enidine 제조사 카탈로그의 공개 사본](https://www.omni-tec.it/wp-content/uploads/2017/11/enidine-catalog_en.pdf),
인쇄면 21·35. 이 치수의 확인은 실제 장치의 마찰·잠금 토크·감쇠 곡선 측정을 대신하지 않는다.

## 단계별 통과와 재개

- 후보의 `feasibility_status=ready`는 핵심 근거가 있어 부품 물리 제작·검증을 **시도할 수 있음**을
  뜻한다. 아직 수행하지 않은 후속 검증을 성공으로 표현하지 않는다.
  핵심 기구를 모르는 `research-needed`·`blocked` 후보는 자동 대안으로 승격하지 않는다.
- 비교기에서 원문을 읽지 못한 `read_failure`와, 원문을 읽었지만 실제 의미가 겹치는
  `uncertain`은 다르다. 전자는 제한적으로 다시 읽고 누락으로 남기며 중복 판정으로 저장하지 않는다.
- 인용은 표를 임의로 평탄화하거나 유니코드·수치를 바꾸어 만들지 않는다.
  원문에 있는 짧은 문구와 정확한 위치를 사용한다.
- 단계만 지정해 재시작해도 기존 수정 요청을 보존한다. `--from-stage` 사용으로
  커넥터 제외 등 현재 제작 조건이 사라지지 않도록 회귀 테스트를 추가했다.
- 공개 PDF도 AES 지원이나 CFF 폰트 해석 기능이 필요할 수 있다.
  수집 환경은 `pypdf[crypto]`와 `fonttools`를 잠금 파일에 포함한다.

## 공용 실행기와 측정

- 고정한 YAM 실행기의 기본 주기는 10 Hz지만, 최종 평가 실행기는 `control_hz=20`을
  명시한다. 데모·채점·모델 평가의 실제 메타데이터와 유지 시간·스텝 제한을 맞춘다.
- 0.5 m/s에서 12 mm를 이동하는 시간은 24 ms로, 20 Hz 관측 간격인 50 ms보다 짧다.
  충격·관통·최대 힘·손상은 물리 하위 스텝에서 누적하고 원시 측정값을 남겨야 한다.
  이 계산은 실제 접촉 시뮬레이션의 검증 결과가 아니다.
- 고정한 `yam_arena/task_file.py`는 물체 선언과 `success/score/distance/demo`를 읽는다.
  임의의 `configure_scene` 같은 훅을 지원한다고 가정하지 않는다. 관절·힘 적용이 필요하면
  공용 소스 파일을 고치지 않는 작업 전용 연결 코드를 명시적으로 설계하고, 같은 `task.py`
  진입점에서 데모와 Dream이 동일한 기구를 쓰는지 깨끗한 디렉터리에서 검증한다.
- 상태를 가진 채점기는 `success()`와 `score()` 호출로 같은 스텝을 두 번 세지 않는다.
  손상은 누적하고, 과거 최고 점수로 이후 손상을 덮지 않는다. 초기화는 별도이며
  에피소드 중 물체 순간이동·숨은 고정·직접 상태 조작으로 성공을 만들지 않는다.
- 연결 코드의 중복 등록, 리셋 후 잔류 손상, 원문 판독·일반 로직 테스트와 실제 GPU
  물리 검증을 구분하는 항목은 최종 제작 검증에서도 확인한다.

## Focused novelty evidence review

A comparison can be revisited when a source review exposes omitted mechanism evidence.
`input/novelty-review-notes.json` maps exact catalog IDs to source-review leads, never
forced verdicts. Only the affected comparison cache key includes those notes; original
reviews remain preserved. The reviewer must independently read the originals and may
still return overlap/uncertain. All coverage, objective/skill/outcome criteria and final
novelty gates remain mandatory. Regression coverage confirms notes cannot force a pass
and unrelated exact comparisons remain reusable. This is review infrastructure, not
evidence any proposed task passed novelty or physical validation.

## Original benchmark definitions and stage-aware review

The first full catalog sweep completed2,210 comparisons but the novelty critic correctly
found that several benchmark references were only overviews/registration modules. Those
pages do not establish coverage of actual task definitions. `rcauto.benchmarks` now collects
commit-pinned original definition files, validates Git blob bytes, rejects truncated trees,
missing selector groups and changed/missing local sources, and retains a per-suite manifest
with the exact selectors and registry/navigation inventory. The review must still verify
that those selectors cover the registered benchmark; an inventory is not a semantic verdict.

The live source acquisition verified1,759 files/pages: LIBERO135, RoboCasa381, SimplerEnv11
(including the implementation submodule pinned by its parent), BEHAVIOR1,121 (current1K
plus original100 definitions), DataCenterBench1 published task page and RoboDojo110. These
are source-file counts, not unique task counts or simulator results. Originals remain
private references; public evidence records versions and coverage metadata. Collection
runs locally before GPU startup and refreshes again for final novelty review. Missing
originals now fail before an expensive catalog sweep.

The critic is explicitly given the same archival uncertainty rules as the catalog gate:
closed/unmerged, explicitly other-team, non-board-required unknown history is retained as
unknown; all known overlaps and all required-scope uncertainty continue to block. At
ideation, numeric assumptions are formalized at the following contract stage. Physics,
demo, scorer and evaluation evidence are required by their later gates, not presumed to
exist before a candidate can be selected. This change does not assert that the damper task
has passed novelty, physics, YAM demos or Dream.

A subsequent catalog refresh produced conflicting assessments for the open tourniquet
windlass PR1026. A source-review request records both judgments and asks whether the
candidate actually shares the windlass rotational backdrive constraint; an axial piston
return spring must not be silently treated as a rotary windlass load. This is a request
for a grounded comparison, not a pass or a scope exemption. Source-review notes and
plain-text originals can be queued outside the managed worker download so they survive
resume. The queue rejects authority files such as request.json and nonregular files.
