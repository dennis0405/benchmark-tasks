# 20261114T031727-dc73f2bb

Status: **blocked**

사용자 요청 원문:
좋아. 이제 인스턴스를 다시 켜고 첫 태스크를 만들어보자.
필요한 CLI 인증부터 확인하고, 로봇 조립·데이터센터 정비 분야에서 산업 근거가 있고 기존 제출과 겹치지 않는 후보를 골라 제출 전 검토 단계까지 진행해줘.

확정된 제작 조건:
- Category: Impact on AI timelines. 로봇 조립 또는 데이터센터 정비 중 실제 산업에서 자동화할 의미가 있는 물리 조작 하나를 선택한다.
- 제조사 설치·정비 매뉴얼, 공식 조립 절차, 공개 설계, 실제 작업 영상·보고서의 원문으로 작업과 장비 형상·치수·절차를 입증한다. 추측이나 가상 산업 용도를 만들어내지 않는다.
- 여러 후보를 비교하고 목적·물리 기술·성공 조건의 신규성과 로봇의 어려운 조작 능력을 함께 검토하여 하나만 제작한다. 조작의 어려움은 극단적일 필요는 없지만 단순 물체 옮기기를 산업 이름만 바꾼 작업은 피한다.
- 모든 분야의 기존 accepted/pending/open 제출 및 우리 팀 기존 제출, 지정된 기존 벤치마크와 비교한다. 기하·색상·배치·시작 자세·난도만 다른 것은 새 태스크가 아니다. 선정 시점 및 완성 후 다시 전수 비교한다. 불확실한 중복을 통과시키지 않는다.
- 가능한 강체 PhysX 작업을 선택한다. 기존의 검증된 연성 에셋이 없다면 새 연성 솔버 연구는 하지 않는다. 실제 산업 장비의 형태와 중요한 물리 제약을 재현한다. 공개 에셋은 라이선스·출처·치수를 기록하고 적합하지 않으면 직접 제작한다.
- objective와 환경·부품 물리를 먼저 완성·진단한 뒤 YAM 전체 데모·채점기를 연결한다. 물체 순간이동/상태 덮어쓰기로 성공을 만들지 않는다. 관련 부품 파손·금지 동작·실패·부분 점수와 성공 유지 시간을 수치로 평가한다.
- 실제 simulator 측정·두 번의 full-score YAM 데모·실패/부분 점수 대조군·영상/원시 trace 및 깨끗한 경로 재현 검사를 수행한다. 미검증 수치와 성공을 만들어 보고하지 않는다.
- 모든 제작·검증을 마친 후에만 Dream 최종 평가 episode 1회. 실패나 불명 결과도 자동 재평가하지 않는다. 누적 $20 ledger를 준수하고 인증이 없으면 제작을 보존한 뒤 평가 대기한다.
- 결과는 이 run의 새로운 origin 브랜치에 보존한다. 원본 매뉴얼 전체/인증 정보는 공개하지 않는다. 50MB/file 또는 100MB checkpoint 초과는 먼저 보고한다.
- 완료 후 GPU를 stop하고 제출 전 검토 보고서·영상·채점 결과를 제공한다. 실제 미디어 업로드·대회 PR·승인은 하지 않는다.
- 기존 feature/close-snack-bag checkout 및 robocurve-snack-film 인스턴스는 별개 작업이므로 건드리지 않는다.


contract-critic: Recomputed terminal scores for the 14 required controls are 0, 0.25, 0.5, 0.75, 0.75, 0, 0.25, 0, 0.25, 0, 0, 0, 0, and 0.75. Controls 5 and 14 cannot reach their declared 1.0: CLEAR requires every robot rigid link in R to remain at least 30 mm from F, which includes the bench. The pinned YAM link set includes each arm's base, and both bases are placed at TABLE_HEIGHT on the tabletop, so their surface separation from the bench is not 30 mm. Restrict clearance to the grippers/moving arm links versus the task work area, consistent with selected.json.; The score can be gamed without demonstrating retained stop function. A released window excludes robot/tool contact only with S and N73, so the robot may hold or directly move the jaws; cycle qualification observes only jaw-angle crossings and never requires cold-lever/piston actuation. After C3, L is not revoked by a wrong final open angle and CLEAR does not require the jaws to remain released, settled, and within ±3 degrees. Require externally released stop endpoints, native lever/piston-linked cycles without direct robot/tool jaw driving, and the final angle tolerance throughout the one-second clearance dwell.; The fixture-displacement hard-failure control is not physically executable as specified: F explicitly contains an anchored fixture frame, cover, and bench, yet the control requires fixture translation over 10 mm. A static anchored body cannot produce that event through normal robot actions, and an after-initialization state write is forbidden. Specify a finite, physically breakable fixture mount and its datum, or replace this control with a physically realizable loss event without weakening damage handling.; Several score-critical terms remain ambiguous: “open settled endpoint,” “released settled endpoint,” “adjustment motion occurs,” and “endpoint range” lack complete numeric predicates. In particular, undefined adjustment motion can revoke L on arbitrary numerical jitter, while g73<=0.5 mm has no lower bound and therefore accepts substantial nut/seat penetration until the separate 2 mm damage threshold. Define stop-contact/release and settling observables, a motion threshold with units, the exact endpoint set used for range, and a tight signed seating interval.; selected.json says angle credit is revoked whenever locking prerequisites fail, but contract.json deliberately awards 0.25 for an unseated locking nut in the angle-only and wrong-seating controls. Resolve this contradiction explicitly while preserving meaningful partial credit.; Timing and freshness are otherwise internally consistent: 4800 actions at 20 Hz equal 240 seconds; 240 Hz uses 12 substeps/action and 480 Hz refinement uses 24; the 60-, 48-, and 240-sample windows equal 0.25, 0.20, and 1.00 seconds; action 4800 completes before current-score evaluation; no post-timeout progress or bonus exists. The restored runtime preflight and pinned sources support the 20 Hz configuration path and pre/post physics callbacks, while correctly reserving contact, force, joint-state, and refinement certification for later native probes.; The prior contact-manifold and energized-motion specification failures are repaired: contact force, penetration, and work are now point-wise with explicit pair aggregation/event reset rules, and energized axial motion uses fresh consecutive substep displacement. Industrial evidence also supports the unpressurized closed-position adjustment, nuts 73 and 74, locking with nut 73, and subsequent actuation test; the cold lever, counterholding law, target, and tolerances are properly disclosed as benchmark assumptions.

- Release: `release/`
- Evidence: `workspace/artifacts/`
- This report is not submission approval.
