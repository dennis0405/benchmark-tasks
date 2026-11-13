# 기술 설계 초안: 환경·에셋 제작과 물리 검증

상태: 검토 초안, 구현 전. 작성일: 2026-11-13 (Asia/Seoul).
상위 계약은 [제작 의도](../intent/robocurve-automation.md), [운영 설계](mvp-workflow.md),
[기술 구성](technical-architecture.md)을 따른다. 도구 설치·모델 제작·GPU 실험은
수행하지 않았다. 기존 과자봉지 task나 진행 중인 서버 작업을 변경하지 않는다.

## 1. 작업 경계와 입력

앞단에서 선정한 CandidateCard, 산업 EvidenceCard, AssetRecord, NoveltyReport를
받는다. 환경 제작 전에 **TaskContract**를 작성한다. task마다 objective, 실제
장비/도구, 치수·재질·질량, 초기 배치, 요구 조작, 성공·부분 성공·실패, 측정
방법, 허용 근사, 로봇·카메라·run limit를 지정한다. 숫자의 출처·가정·단위도 기록한다.

TaskContract는 자동 제작 과정에서 먼저 동결할 계약이다. 사용자가 매번 이
문서를 승인해야 한다는 뜻은 아니다. 예외 보고 또는 최종 제출 승인과 구별한다.
구현으로 알 수 없던 조건이 드러나면 계약 변경 이유와 영향 검토를 남긴다.
목표의 의미가 바뀌면 산업 근거와 신규성 검사도 다시 수행한다.

## 2. 역할과 도구

| 역할 ID | 책임·허용 변경 | 연결할 도구 | 산출물 |
|---|---|---|---|
| task-specifier | 후보를 측정 가능한 TaskContract로 구체화 | 근거·에셋 카탈로그, 문서 조회 | 계약과 가정·관측 계획 |
| asset-builder | 재사용/수정/신규 제작 경로 선택, 장비 형상 구현 | CAD 변환 어댑터, mesh 처리, USD authoring | 에셋 패키지·수정 이력 |
| environment-builder | task 작업 공간에서 장면·물리·reset·로봇·관측 구성 | Isaac Sim/Isaac Lab-Arena, YAM 어댑터 | 환경 코드·설정 |
| physics-test-designer | 과제에 필요한 물리 시험과 대조군 제안 | TaskContract·형상·관측 계획 조회 | 시험 계획과 판정 기준 |
| physics-reviewer | 측정된 결과의 의미·오류·한계 검토 | 수치 보고·접촉/형상 증거·영상 | 준비 판정 의견 |
| visual-reviewer | 원문 장비와 형상·시점·가시성이 맞는지 검토 | 도면·사진·setup 및 충돌 형상 렌더 | 형상·영상 불일치 목록 |

역할은 앞단과 동일한 Codex 실행 어댑터를 우선 제안한다. 실행 모델·버전을
기록하며, 역할 분리만으로 독립 검증을 보장하지 않는다. 준비 판정은 에이전트의
동의가 아니라 필수 시험 결과의 완성·유효성도 함께 확인하는 조정기가 확정한다.

일반 프로그램인 **asset-inspector, sim-runner, trace-recorder, gate-evaluator**가
변환 호출, 파일 해시, GPU 작업, 수치 기록, 검사 실행과 상태 전이를 담당한다.
에이전트는 성공 숫자를 직접 작성하는 대신 검사 결과의 참조를 반환한다.

## 3. 에셋 제작 경로

| 입력/상황 | 경로 제안 | 필수 확인 |
|---|---|---|
| 사용할 수 있는 USD/URDF | 상대 참조 정리와 필요한 설정 수정 | 단위·관절축·limit·질량·충돌·하위 참조 |
| CAD STEP 등 | CAD 변환 어댑터 → mesh/USD | 변환 전후 치수·좌표·핵심 형상·부품 분리 |
| 시각용 mesh | mesh 검사·수정 → 물리 재질·관절·충돌 설정 추가 | visual과 collision의 차이 및 task 영향 |
| 맞는 원본 에셋 없음 | 근거 도면·치수로 parametric 형상 제작 | 실제 장비 대응, 출처 있는 치수, 가정 |
| 기존 연성 에셋 | 고정된 원래 solver/runtime에서 필요한 동작 재현 | 선택한 YAM 환경과의 결합·관측까지 검증 |

핵심 도구는 USD Python API/UsdGeom/UsdPhysics, trimesh 등의 형상 처리기와
시뮬레이터 importer를 제안한다. CAD 입력용 FreeCAD/OpenCASCADE, 복잡한
시각 형상용 Blender는 변환 어댑터 후보이며, 정확한 버전과 지원 형식·headless
실행은 구현 전 호환성 시험에서 결정한다. 모든 CAD가 자동 변환된다고 가정하지 않는다.

geometry·시각 재질·collision·관절/물성·하위 참조·출처/라이선스를 하나의 에셋
패키지로 보존한다. 원본과 수정/변환본은 별도 해시로 연결한다. task에는 필요한
패키지를 포함해 다른 서버에서 실행 가능하게 한다. 재사용 원본을 task 수정으로
덮어쓰지 않는다. 현재 특정 서버 디렉터리에만 있는 파일을 제출 의존성으로 남기지 않는다.

충돌 형상을 단순화할 때에는 끼워 맞춤·손잡이·틈·잠금·체결 면을 보존한다.
convex hull 하나가 구멍을 메우거나 잠금부를 제거할 수 있으므로 실제 collision
렌더와 원래 장비 형상을 대조한다. 모든 작은 시각 요소를 물리적으로 재현할
필요는 없지만 과제가 측정할 능력을 제거하면 안 된다.

## 4. 초기 런타임과 이식성

초기 기준 후보는 **RTX GPU + Isaac Sim 6.1 계열 + Isaac Lab-Arena + 양팔 YAM,
강체 PhysX 경로**다. 기존 기록에는 고정한 Arena/YAM revision이 있지만 실제
Brev 환경의 지원 API·버전·패키지 조합을 확인하기 전 정상 동작을 단정하지 않는다.
새로운 연성 solver와의 결합을 기준 환경 설치에 섞지 않는다.

container image digest, driver·GPU 모델, simulator/Lab/Arena/YAM revision과
로컬 patch, 라이브러리 lock, asset hash, 실행 인자를 EnvironmentLock으로 남긴다.
Docker 이미지 태그만 같다는 이유로 같은 환경이라고 판단하지 않는다.

기본 실행은 렌더가 가능한 headless GPU 작업이다. live desktop/streaming은
필수 경로가 아니라 디버깅 옵션이다. 로컬 Brev CLI·VS Code로 실행을 요청하되
원격 조정기가 job ID·프로세스·로그·종료 상태를 관리한다. 한 제작 작업의 GPU
실험을 직렬화하고 다른 작업을 임의로 종료하지 않는다.

### 내부 TaskAdapter 제안

후보 task는 공용 `task.py` 형태의 어댑터로 연결하는 안을 우선한다. 이는
공식 대회가 강제하는 API가 아니라 **프로젝트 내부 규약**이다. 기존
`arena_scene.py` 기반 task는 별도 호환 어댑터를 통해 검증한다.

| 기능 | 내부 책임 |
|---|---|
| 환경 구성/reset | 선택한 embodiment·device·seed·고정 설정으로 환경 생성·초기화 |
| 로봇 step | action shape·범위·유효성 검사 후 한 control step 실행 |
| 정책 관측 | 모델에 전달할 RGB와 실제 joint/gripper state |
| 검사 관측 | root/body pose·관절·접촉·속도·형상·위반 등 task별 계측 |
| 데모 controller | 현재 관측으로 action 생성, 실행 중 객체 pose 직접 수정 금지 |
| scorer | 검사 관측의 연속 갱신과 점수·유효성·이벤트 출력 |
| recording/export | 원시 기록·영상·버전·재현 가능한 standalone runtime 묶음 |

make_environment·read_object_states·make_demo_controller·make_scorer 등
기존 공용 규약의 함수명은 실제 runtime과 대조한 뒤 고정한다. source file 이름만
맞추는 것으로 호환성을 판단하지 않는다. task에 필요한 기능을 지원하지 않으면
어댑터를 확장·검증하거나 후보를 보류한다.

## 5. 관측·기록 계약

검사 관측에는 `episode_id`, 증가하는 control/physics step, sim time, sample
time, 단위·좌표계·회전 순서, validity와 원본 데이터 참조를 포함한다.

- AABB 중심과 body/root origin, world frame과 local frame을 구분한다.
- 관절/pose write 후 필요한 forward/update 없이 오래된 상태를 읽지 않는지 시험한다.
- 접촉은 지정 body pair·센서 설정·힘/impulse 단위·갱신 주기를 기록한다.
  센서 미설정·미갱신을 접촉 없음이나 성공의 증거로 사용하지 않는다.
- 순간 파손·접촉 상실이 중요하면 physics substep 이벤트로 누적한다. 저장한
  매 N번째 프레임만으로 연속적인 안전·무관통을 보장했다고 쓰지 않는다.
- 데이터가 유효하지 않으면 `measurement-invalid`로 표시한다. 에이전트가
  빠진 값에 기본 성공값을 넣지 못하게 한다.
- 로봇 동작 action과 task mechanics의 변경, 초기화·fixture 사용을 따로 기록한다.

## 6. 물리 검증 도구와 시험

TaskContract의 숫자에 근거해 시험별 허용 오차·유지 시간·유효 관측 조건을
정한다. 모든 task에 같은 접촉력·관통 깊이를 적용하지 않는다. 실제 측정값이
없는 설정은 수치 가정이며, 시뮬레이션의 힘이 현실의 안전 한도를 검증한 것은 아니다.

| 시험 | 실행·계측 도구 | 기대 산출물 |
|---|---|---|
| 단위·크기·참조 | USD/mesh/XML 정적 검사·해시 | 정적 에셋 검사 보고 |
| 초기 중첩·안정화 | collision 검사, reset·무행동 rollout | 침투/낙하/발산/자발적 성공 여부 |
| 관절·잠금·스프링 | 수치 입력을 주는 구성요소 fixture, 관절/힘 기록 | 구동·limit·복원·잠금 상태 검증 |
| 파지·마찰·해제 | 접촉 센서·물체/로봇 상대 pose·slip 기록 | 접촉 유지·해제 후 상태 |
| 맞물림·삽입 | 실제 collision 형상·깊이/정렬·경계 대조군 | 근접 상태와 실제 체결 구별 |
| 관통·충돌 | 형상 쌍·삼각형/mesh 검사, 지원되는 CCD·고빈도 이벤트 | 검출 결과·샘플 간격·사각지대 |
| 로봇·카메라 | FK/IK·joint limit·기초 제어·RGB 검사 | 도달/추종·관측 호환성 |
| 연성 예외 | 해당 에셋 원래 demo·동일조건 negative control | 필요한 변형/자기 접촉과 결합의 증거 |

trimesh 등의 정적 충돌 검사는 시뮬레이터의 접촉 판정과 다를 수 있다. 표면
근접·접촉·실제 관통을 구분하고 검사 알고리즘의 공면·얇은 면·샘플링 한계를
명시한다. physics-reviewer의 영상 판단만으로 연속 충돌 검사를 대체하지 않는다.

reset 시 배치·관절값을 설정하거나 진단 fixture로 직접 누르는 시험은 가능하다.
그 결과는 **물리 구성요소 검증**이다. full-score 로봇 데모로 승격하지 않는다.

## 7. EnvironmentReadyReport와 실패 전이

보고서에는 task/environment/asset hash, 근거-장비 대응, 필수 시험과 수치,
실패/관측 불명, 허용 근사, 검사 한계, 최종 준비 판정을 담는다. gate-evaluator는
필수 증거가 있고 모두 유효한지 확인한다. 캐시된 다른 에셋 버전의 통과는 무효다.

- 정적 변환 문제: asset-builder에게 반환.
- 실제 장비와 형상 불일치: 에셋 또는 TaskContract 수정, 산업 근거·신규성 영향 검토.
- 물리 불안정/관통/관측 누락: environment-builder에게 반환; 데모 개발은 대기.
- 도달 불가능: 배치/도구/embodiment를 실제 작업과 양립하는 범위에서 재검토.
  현실에 없는 축소·물리 비활성화로 해결하지 않는다.
- 연성 기능이 별도 연구를 요구: MVP에서 보류하고 다음 후보 또는 사용자 판단.
- 환경 준비 후 문제가 드러남: 준비 상태를 취소하고 의존 데모·채점 증거를 낡은 것으로 표시.

반복 횟수는 운영 설계의 미확정 정책을 따른다. 수정 전 원인 가설과 검증 방법을
남기며 동일 증거로 무한 반복하지 않는다.

## 8. 구현 시 검증할 사항

지원 이미지·API와 portable export 경로, CAD 변환 품질, 접촉 데이터의 신뢰성,
관절 초기화·forward 순서, 렌더·제어 호환성, GPU 메모리·시간을 확인해야 한다.
별도 CPU physics 시험은 가능하지만 실제 제출할 Isaac Sim/YAM 환경의 검증을
대신하지 않는다.

### 확인한 출처

- [YAM Arena 고정 revision README](https://github.com/kairosiann/yam-isaac-lab-arena/blob/0059d02a1a91bc6834c1d0846022891df0c78730/README.md)
- [UsdPhysics schemas](https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr/UsdPhysics.html)
- [Isaac Lab contact sensor](https://isaac-sim.github.io/IsaacLab/main/source/concepts/sensors/contact_sensor.html)
- [trimesh collision](https://trimesh.org/trimesh.collision.html)

최신 문서와 고정 런타임의 API가 다를 수 있으므로 구현 시 설치 소스와 대조한다.
