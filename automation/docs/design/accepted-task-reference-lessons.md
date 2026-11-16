# Accepted 사례를 이용한 제작 원칙

2026-11-16 인증된 board에서 TK-HYQRVJ와 TK-TGHDCE의 Accepted 상태를 확인했다. 아래는 원본 구현에서 확인한 사실과 이번 제작에 적용할 판단이다. 채택 사유를 추정한 정책 문서나 새 그리퍼 태스크의 검증 보고서가 아니다.

## 확인한 원본

| 과제 | 고정 원본 | 기록된 결과 |
|---|---|---|
| TK-TGHDCE 호스·바브·클램프 | [PR1892](https://github.com/robocurve/benchmark-tasks/pull/1892), `501851adca10137772bdc2cee39b5ab1a02a446f` | `task.json`: demo1/795steps, Dream0/129steps, DreamSessionExpired |
| TK-HYQRVJ 상자 테이핑 | [PR1893](https://github.com/robocurve/benchmark-tasks/pull/1893), `d12739e5c923856d3be0cc904fd4decfdb99ef15` | `task.json`: demo1/574steps, Dream0/129steps, DreamSessionExpired |

두 과제의 실제 category는 Impact on jobs다. 우리 자동화의 기존 Impact on AI timelines 범위와 구분한다. 이번 그리퍼 후보는 로봇 부품의 설정 작업으로 기존 범위에 포함된다.

## 모델링과 구현

- 호스는 강체 튜브와 슬라이드 조인트로 삽입·유지를 근사한다. 스크루 상태는 안착한 드라이버의 실제 회전에서 계산하며, 드라이버에 반작용 토크는 가하지 않는다. [원본 README](https://github.com/eainx/benchmark-tasks/blob/501851adca10137772bdc2cee39b5ab1a02a446f/tasks/hose-clamp-barb/README.md)
- 테이프는 연속 필름 대신 접착 구간으로 표현한다. 측정한 롤러·손가락의 움직임으로 부착과 눌러 붙이기를 판정하고, 상자 덮개는 강체 패널과 경첩으로 구성한다. [원본 README](https://github.com/eainx/benchmark-tasks/blob/d12739e5c923856d3be0cc904fd4decfdb99ef15/tasks/carton-tape-seal/README.md)
- 두 README 모두 실제 하드웨어 검증을 하지 않았다고 명시한다. 물리 파라미터를 실측값으로 소개하지 않는다.
- 복합 형상, 관절, `make_environment`, 태스크 전용 dynamics, 별도 scorer와 원시 궤적 재채점이 분리돼 있다. `Part`를 기존 Box의 하위 형식으로 만들어 공개 로더와 연결한다. 단순 primitive 선언만으로 복잡한 장치를 포기할 이유는 없다. 코드·에셋 재사용은 해당 라이선스를 먼저 확인하고, 미확인 코드는 구현 패턴을 참고해 독립 작성한다.

## 새 그리퍼 태스크에 적용

1. 제조사 도해와 실제 작업 절차를 바탕으로, 이미 조립된 그리퍼·조정나사·너트·공구를 갖춘 장면부터 만든다. 전체 조립이나 새로운 연성체 연구로 범위를 늘리지 않는다.
2. 핵심 능력이 반대 토크를 유지하며 잠그는 것이라면, 나사를 잡지 않았을 때 조정값이 흐트러지는 물리적 결합을 구현하고 대조군으로 확인한다. 호스 사례의 무반작용 회전 모델을 그대로 복사해서 양손 협응을 주장하지 않는다.
3. 구성 요소·초기 채점·관절·공구 접촉을 확인한 뒤 YAM 데모를 만든다. native14D actions, 원시 상태, 현재 trial에 연결된 영상, 재채점으로 결과를 입증한다. 스크립트의 상태 읽기는 허용하되 물체를 옮기거나 붙여 성공시키지 않는다.
4. Accepted 사례의 전용 runner가 우리 표준 Inspect 최종 평가에 바로 연결된다고 가정하지 않는다. 태스크 전용 scene/dynamics 연결부를 실제로 실행하되 관측·action·guard 의미는 유지한다.
5. Dream 실패와 데모 성공을 구분한다. 위 원본의 Expired 기록은 성공한 모델 평가가 아니다. 새 태스크의 최종 episode는 한 번만 시도하고, 실패·불명 상태와 실제 부분 결과를 남기며 자동 재시도하지 않는다.
6. 후보 선정 시와 최종 제출 직전 모두 전체 제출·팀 이력·기존 벤치마크를 비교한다. 예제 구현 전체를 모든 역할이 반복해 읽게 하지 않고, 중복 역할은 목적·물리 기술·성공 조건, 구현 역할은 필요한 코드 구간을 읽는다.

## 이번에 보완한 수집기

`task.json`이 의도적으로 없는 공개 Draft도 비교에서 제외하지 않는다. 해당 Draft의 원본 README·task.py·scoring.py를 commit별로 수집하고 원문과 해시를 보존한다. 정식 task.json이나 채점 기준을 임의로 생성하지 않으며, 필수 원문이 없는 경우 동기화를 완료로 표시하지 않는다. 회귀 검사를 포함한 전체138 tests와 Ruff가 통과했다.
