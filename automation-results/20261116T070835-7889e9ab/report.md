# 20261116T070835-7889e9ab

Status: **blocked**

사용자 요청 원문과 확정 방향 (변경하지 말 것)

아니면
## Space out touching parcels on a running belt before they reach the scanner
이 제출물 참고해서,
우리는 컨베이어 벨트 위에서 박스가 지나가는 과정에서,
박스가 열려 있는 경우는 불량품으로 판단해서 컨베이어 벨트에서 빼고,
그렇지 않은 건 통과하도록 하는 task는 어떨까?
먼저 기존 제출물 한번 참고해볼래?

좋아. 그러면 열린 박스는 플랩을 피해 몸통을 잡고 내용물을 흘리지 않게 빼내고,
닫힌 박스는 그대로 통과시키는 방향으로 구체화해봐. 기존 태스크와 차별점, 채점 기준, 구현 순서까지 정리해줘.

[$robocurve-workflow](/Users/dennis0405/Workspace/SNU-PI/robocurve/automation/skills/robocurve-workflow/SKILL.md) 이거 참고해서 진행하도록 해.
다만 아이디어 자체는 지금 방향으로 확정해서 작업 진행해보자.

실행 범위
- 이 run은 새 과제다. 기존 snack-bag 및 다른 산업 과제의 run/branch/worker를 변경하거나 종료하지 않는다. 다른 run이 등록 GPU를 소유하면 건드리지 말고 실제 자원 충돌을 보고한다.
- 아이디어는 '열린 상자의 플랩을 피해 몸통을 파지하고 자유 내용물을 보존하며 이동 벨트에서 불량품 트레이로 옮김; 닫힌 정상 상자는 벨트에서 그대로 통과'로 고정한다. 다른 산업 작업군으로 바꾸거나 여러 무관한 후보를 탐색하지 않는다. 이 방향 안에서 형상, 배치, 플랩 상태, 속도 등을 물리적으로 실행 가능하게 조정할 수 있다.
- 최종 목적은 분류 설명 자체보다 상태에 맞는 정밀 조작을 평가하는 것이다. 색깔만 바꾼 pick-and-place와 실제 능력이 같은지 엄격히 확인한다. 산업 설정과 신규성은 별개다. 중복 판정을 억지로 통과시키지 않고 구체적 근거와 해결하지 못한 부분을 보고한다.
- 작업 방식은 공용 task.py / YAM / 실제 pinned runtime. 새 연성 솔버 연구는 하지 않는다. 강체 상자 몸통, 유한 두께의 실제 충돌 플랩과 관절, 자유롭게 움직이는 내용물로 시작한다.

이미 확인한 비교 자료 (재검증 가능한 원문 단서; 신규성 통과를 강제하는 결론이 아님)
1. robocurve/benchmark-tasks PR1214 accepted/merged, Space out touching parcels on a running belt before they reach the scanner.
   https://github.com/robocurve/benchmark-tasks/pull/1214
   https://github.com/robocurve/benchmark-tasks/tree/ed4635d8cdbb80ff3616fb7490ccba921d5dd2ab/tasks/space-touching-parcels
   물리 마찰로 움직이는 벨트, 고정 폐쇄 상자3개, 두 간격을 스캐너에서 확보하는 작업. 벨트 밖 임시 적치/쌓기 금지, 전 구간 모니터. 초기 리뷰가 지적한 일시적 적치 후 복구 exploit과 보완을 참고.
2. PR83 Add conveyor block sorting task는 최종 거절 후 closed/unmerged. 색깔 큐브의 벨트 선택/분류는 기존 RoboDojo와 겹친다는 공식 거절 사유가 있다. 캐시 HTML 상태보다 실제 GitHub 상태/공식 댓글을 우선한다.
   https://github.com/robocurve/benchmark-tasks/pull/83
   https://github.com/robocurve/benchmark-tasks/pull/83#issuecomment-6814162817
3. 원문 비교 필수:
   https://robodojo-benchmark.com/doc/sim-tasks/match-and-pick-from-conveyor/
   https://robodojo-benchmark.com/doc/sim-tasks/pick-from-conveyor-by-image/
   PR910 conveyor-photo-build 및 다른 moving conveyor selection 과제도 비교한다.
- 원래 작업과의 구분 후보는 '움직이는 개방 플랩의 간섭 회피 + 몸통 파지 + 내부 자유 내용물의 유출 방지 + 정상품에 개입하지 않고 라인 통과'의 물리적 결합이다. 플랩과 내용물이 장식에 그치면 차별점으로 주장하지 않는다. 전수 제출/팀 이력/6개 지정 벤치마크 비교를 유지한다.

명세·채점 설계 지침 (아래 수치 제안은 사용자 확정치가 아닌 검증할 초기안)
- 예시 지시문: Remove the open cartons to the reject tray without spilling their contents. Let the closed cartons continue down the belt.
- YAM 파지 폭을 실제 모델에서 확인. 예를 들어 길이120mm x 폭80mm x 높이70mm의 몸통은 시작점일 뿐이며 플랩 간섭, 벽 두께, 자유 파지면, 그리퍼 여유를 실제 충돌 모델로 검증한다.
- 상자는 바닥과4개 벽의 열린 내부를 가진 compound rigid geometry; 열린 플랩은 회전관절/실제 충돌 형상. 닫힌 상태의 유지 장치는 명시한다. 필수 충돌 제거, 가짜 손-상자 weld, 내용물 고정, pose/velocity 덮어쓰기, 순간이동으로 성공을 만들지 않는다. 정상적인 벨트 표면 속도 구현과 상자 속도 강제주입은 구분한다.
- 내용물은 셀 수 있는 자유 강체 몇 개로 시작한다. 실제 내부 공간/출구와 물체 형상에 기반한 유출 판정. 일단 유출한 뒤 다시 주워 넣어도 유출 이력은 지워지지 않는다.
- 정상/불량 라벨은 시작 시 실제 폐쇄 상태에 고정한다. 로봇이 뚜껑을 닫거나 정상품을 열어서 라벨을 바꿀 수 없다. 불량을 닫아 통과시키기는 이 과제의 제거 성공이 아니다.
- 불량 성공: 제한 시간 내 reject tray에 몸통과 내용물이 함께 안정적으로 놓이고 손을 완전히 뗀 상태로3초 유지. 플랩만 집거나 테이블/바닥으로 밀어 떨어뜨리는 것은 성공이 아니다. 몸통 파지는 실제 양측 접촉/지지 위치와 원시 trace로 검증하며 물리적으로 동등하게 유효한 전략은 설명 없이 배제하지 않는다.
- 정상 성공: 원래 벨트 경로로 downstream 판정선을 통과. 잠깐 벨트 밖으로 꺼냈다 복귀하거나 라인을 장시간 막아도 통과하도록 만들지 않는다. 무접촉 통과 대조군의 실제 마찰/속도 기반으로 충분한 시간 여유를 정한다.
- 판정은 매 제어 주기 이력을 누적해 유출/잘못된 정상품 제거/낙하/라인 정지/누락을 잡는다. 초기점수0, 중간 진행과 최종 성공을 구분. 부분점수와 전역 실패 조건이 모순되지 않도록 계약에서 명확히 정한다. 단일 실패가 모든 부분점수를 무조건 지워 변별력을 없애는지 검토한다.
- 개방 플랩 회피가 실제 필요한지, 내용물을 무시한 거친 이동과 정밀 이동의 결과가 다른지, 전부 제거/전부 방치하는 정책이 만점을 못 받는지 대조군으로 확인한다.

구현 순서와 통과 조건
1. 산업 포장 검사/불량 배출 근거와 원문 비교, 이 아이디어의 명세·채점·라이선스 고정. 새로운 무관한 후보 탐색으로 돌리지 않는다.
2. 정지한 열린 상자1개: 실제 YAM 몸통 파지 -> 들어 올리기 ->3초 유지. 자유 내용물/플랩 실제 충돌 유지. 이 단계가 실패하면 벨트나 여러 상자를 추가하지 않는다.
3. 정지 상자1개를 reject tray로 옮겨 내려놓고 그리퍼 해제 후3초 유지. 무조작/플랩 파지/잘못된 파지/기울여 유출/낙하 대조군과 채점 검증.
4. 마찰 기반 움직이는 벨트에 열린 상자1개. 추종, 파지, 수직 분리, 안전한 수평 이동과 트레이 안착. 속도를 낮춰도 되지만 실제 제한과 이유를 기록.
5. 닫힌 상자의 무개입 통과와 물리/채점 대조군 검증 후 열린·닫힌 혼합 스트림. 확장보다 완결 가능한 작은 장면을 우선하며 배치/순서/플랩 초기조건을 명세한다.
6. 같은 최종 소스의 두 번 독립 fresh-process full-score YAM 데모, 실패/부분점수 대조군, raw action/body/joint/score trace와 독립 재채점, 장면/최종/데모 영상, 깨끗한 환경 이식 확인. 미검증 결과를 만들어 보고하지 않는다.
7. 모든 제작 게이트 후에만 trusted coordinator의 Dream 최종 episode1회와 누적20USD ledger. 실패/결과불명에도 자동 새 episode 금지. 키를 역할 에이전트에 넘기지 않는다.
8. 별도 run origin branch에 코드·에셋·재현 명령·라이선스·최종 영상·주요 측정을 보존하고 실제 compute 상태를 보고. 50MB/file 또는100MB checkpoint 초과는 업로드 전 설명한다.
9. review report.html, release-manifest digest와 실제 영상으로 사용자 검토. 구체적 검토 버전의 별도 승인 전 공식 미디어 업로드/대회 PR/최종 제출은 하지 않는다.


candidate selection ambiguous

- Release: `release/`
- Evidence: `workspace/artifacts/`
- This report is not submission approval.
