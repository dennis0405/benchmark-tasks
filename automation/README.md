# RoboCurve automation

현재는 기획·설계 단계다. 실행 코드나 배포 환경을 구현하지 않았다.

- [사용자가 확정한 제작 의도](docs/intent/robocurve-automation.md)
- [선택한 방향과 MVP 범위](docs/ideas/industrial-task-pipeline.md)
- [운영 설계 초안: 단계별 산출물·통과 조건·실패 처리](docs/design/mvp-workflow.md)
- [기술 설계 초안: 에이전트·도구·Brev 실행 위치](docs/design/technical-architecture.md)
- [상세 기술 설계: 자료·에셋 수집, 제출 동기화와 중복 검사](docs/design/sources-and-novelty.md)
- [상세 기술 설계: 환경·에셋 제작과 물리 검증](docs/design/environment-and-validation.md)
- [상세 기술 설계: 데모·채점·Dream·승인 후 제출](docs/design/demo-evaluation-and-submission.md)
- [MVP 구현 순서·단계별 검증 계획](tasks/plan.md)
- [MVP 구현 체크리스트: 미착수](tasks/todo.md)
- [Brev 사전 준비: 도구·인증·사용자 결정](docs/design/brev-readiness.md)
- [CLI·전용 skill·Brev start/stop·fork 결과 보존](docs/design/cli-skill-and-lifecycle.md)
- [전용 Brev 환경 준비 점검 기록](docs/design/environment-bootstrap-status.md)

범위는 실제 산업 근거에 기반한 AI timelines 태스크의 후보 다수 검토·단일
태스크 제작·승인 후 제출이다. 강체를 기본으로 하되 검증 가능한 기존 연성
에셋을 허용한다. 제작 의도와 상세 운영 설계의 승인 상태를 구분해서 관리한다.
