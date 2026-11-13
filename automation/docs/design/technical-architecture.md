# 기술 설계 초안: 에이전트·도구·실행 위치

상태: 검토 초안, 구현 전. 작성일: 2026-11-13 (Asia/Seoul).
[제작 의도](../intent/robocurve-automation.md)와 [운영 설계](mvp-workflow.md)를 따른다.
이번 문서는 도구 선택 제안이며 설치·인증·에이전트 실행·배포를 수행하지 않았다.

## 1. 실행 위치

사용자는 로컬 **Brev CLI + VS Code 원격 접속**으로 작업한다. 별도 Notebook
중심 흐름을 도입하지 않는다. 소스 관리 기준 경로는 이 automation 프로젝트다.
Brev 실행 경로·저장소·컨테이너의 실제 배치는 환경 검증 후 정한다.

```text
로컬 Mac: 전용 skill → CLI / Brev start·stop / 실행 색인·승인·보고·발행
    ↓ SSH: 원격 실행 요청·상태 조회·재개·결과 수신
Brev: 실행 조정기 → 수집 도구·동기화 작업·판단 에이전트
                    → 이후 환경 제작·GPU 검증 작업
    ↓
실행 기록 DB + 원본·에셋·보고서 저장소 → 검토할 스냅샷을 로컬로 보존
    ↓ fork 작업 브랜치 origin push → 해당 instance stop → 사용자 검토
```

장기 작업은 VS Code 터미널 세션에만 묶지 않고 원격 실행 조정기가 관리하도록
설계한다. 연결이 끊겨도 외부 요청·프로세스 상태를 확인해 이어간다. Brev 인스턴스
삭제 후 보존되는 저장소라고 가정하지 않는다. 선택한 디스크의 지속성과 결과물
반출 경로를 검증하며, 삭제 전 독립 보존 여부를 확인할 수 있게 한다.

CLI를 실행 기준으로 하고 전용 skill로 감싸는 방향은 사용자 확정이다.
검토 대기 전 보존·stop, 수정 요청 시 start·재개, GPU 없이 승인·제출하는
로컬 control plane의 상세 계약은 [생명주기 설계](cli-skill-and-lifecycle.md)를 따른다.

## 2. 에이전트와 일반 프로그램의 경계

에이전트는 의미·근거·작업 난도·신규성을 판단한다. 수집, 페이지네이션, 파일
해시, DB 갱신, 상태 전이, 승인 확인은 일반 프로그램이 처리한다. 이 설계에서
에이전트는 역할별 입력·출력을 가진 호출이다. 자율적으로 무한 실행되는 여러
봇이나 서로 메시지를 보내는 군집을 먼저 만들지는 않는다.

| 역할 ID | 판단할 일 | 연결 도구 | 결과물 |
|---|---|---|---|
| research-planner | 조사 주제·장비·작업 용어와 다음 검색 계획 | 검색 API, 자료 카탈로그 조회 | 검색 계획·조사 공백 |
| evidence-analyst | 원문이 실제 작업·산업 가치·조작 난도를 뒷받침하는지 판단 | 원문/PDF/이미지·영상의 위치가 지정된 추출물 | 주장별 근거 카드 |
| asset-analyst | 실제 장비와 에셋의 대응·수정 필요성 판단 | 에셋 manifest, 형상 검사·문서·미리보기 | 에셋 적합성 카드 |
| task-normalizer | 기존 제출과 후보를 동일한 작업 구조로 정리 | 원문 instruction/setup/scoring 및 필요 시 이미지 | TaskSignature |
| candidate-designer | 근거에 연결된 여러 후보 생성 | 검증된 근거·에셋·중복 카탈로그 | CandidateCard 묶음 |
| novelty-reviewer | 같은 작업 목적·물리 능력·결과를 이미 평가하는지 비교 | 전수 요약 대조, 혼합 검색, 원문 조회 | 중복/차별성/불확실 보고 |
| novelty-critic | 판정의 빠진 사례·잘못된 차별성 근거 반박 | 같은 버전의 근거와 비교 후보, 원문 | 유지·재검토 사유 |
| task-specifier | 후보의 목표·실패·근사·측정 방법 구체화 | 산업 근거·에셋 카탈로그 | TaskContract |
| asset-builder | 실제 장비에 맞는 에셋 재사용·수정·신규 제작 | CAD/USD·mesh 처리기 | 에셋 패키지·변환 이력 |
| environment-builder | 동결한 명세에 맞는 장면·물리·관측 구성 | Isaac Sim/Arena·YAM 어댑터 | 환경과 설정 변경 |
| physics-test-designer / physics-reviewer | 구성요소 시험 설계·측정 결과 검토 | sim-runner·접촉·형상 검사 | 시험 계획·준비 판정 근거 |
| visual-reviewer | 실제 장비·충돌 형상·모델 시점의 대응 확인 | 원문 도면·사진·렌더 | 불일치·가시성 보고 |
| demo-developer | 환경 준비 판정 후 물리 로봇 데모 개선 | 시뮬레이터·제어·접촉 기록 | 데모와 실패 분석 |
| scoring-author / scoring-reviewer | 계약의 채점 구현과 실제 성공·실패의 대응 검토 | 계측·궤적 재채점·대조군 | scorer·타당성 보고 |
| run-reviewer | 동작·영상·계측·점수의 일치 확인 | 원시 기록·영상·환경 검사 | 실행 검토 보고 |
| submission-reviewer | 결과물·보고서의 일치·미해결 문제 확인 | 동결된 소스·영상·점수·라이선스 | 사용자 검토 보고서 |

이는 논리적인 책임 구분이며 역할 수만큼 상주 프로세스를 만들겠다는 뜻은 아니다.
상세 계약은 [자료·신규성](sources-and-novelty.md),
[환경·물리 검증](environment-and-validation.md),
[데모·채점·평가·제출](demo-evaluation-and-submission.md)로 나눈다.
Dream은 위 판단 역할에 쓰지 않고 최종 로봇 정책 평가에만 사용한다.

## 3. 에이전트 실행 수단 제안

첫 구현 후보는 **Codex CLI의 비대화식 `codex exec`**다. 역할별 지침·입력
스냅샷·결과 JSON Schema를 전달하고 JSON 이벤트와 최종 결과를 수집하는
얇은 실행 어댑터를 둔다. 특정 모델 ID는 고정하지 않고 환경 설정으로 선택하되,
실행마다 모델·CLI 버전·프롬프트 버전·입력 해시를 기록한다.

로컬에서 확인한 CLI는 0.147.0이며 `--json`, `--output-schema`,
`--output-last-message`, `--sandbox` 옵션이 있다. **Brev 설치·로그인·모델 접근·
호출 제한은 아직 검증하지 않았다.** 공식 비대화식 실행 설명은
[Codex 문서](https://learn.chatgpt.com/docs/developer-commands#codex-exec)를 따른다.

근거 분석·신규성 판단 역할은 원본 카탈로그를 직접 변경하지 않고 결과만 반환한다.
실행 조정기가 형식과 참조의 유효성을 확인한 뒤 저장한다. 환경·코드 변경 권한은
후반 제작 역할의 해당 task 작업 디렉터리로 구분한다. 모든 역할에 제출 권한을
주는 방식은 사용하지 않는다.

JSON 형식이 맞는다는 것이 내용이 사실이라는 뜻은 아니다. URL·원문 위치·
revision·관측 근거가 실재하는지 도구로 검증한다. 역할을 나눠도 같은 모델이
같은 오류를 공유할 수 있으므로 critic의 동의를 채택 보증으로 쓰지 않는다.

## 4. 도구 선택 제안

| 기능 | MVP 제안 | 선택 이유·경계 |
|---|---|---|
| 웹 탐색 | Parallel Search API를 우선 검토하는 교체 가능한 search adapter | 검색 계획과 원문 탐색 분리. Brev용 API 인증은 별도 검증 |
| 원문 확보 | HTTP 클라이언트(예: httpx), 필요 시 Parallel Extract | 원본 바이트와 추출문을 구분. 추출문만 원본인 것처럼 저장하지 않음 |
| 브라우저 접근 | Playwright + 전용 인증 상태 | board·JS 페이지에 한정. 현재 데스크톱 브라우저/커넥터를 원격에서 쓸 수 있다고 가정하지 않음 |
| PDF | pypdf, 페이지 렌더/표 위치 보존 도구, 필요 시 OCR | 페이지 단위 근거. 스캔 PDF의 빈 추출을 내용 없음으로 처리하지 않음 |
| 영상 | ffmpeg/ffprobe, 제공된 자막·타임스탬프, 필요 시 프레임 검토 | 텍스트만으로 접촉·체결을 입증하지 않음. 실제 사용 가능한 접근 방법 확인 |
| GitHub | REST API; 운영 진단용 gh | 전수 페이지네이션과 댓글·revision 보존. Search API 결과를 전수 목록으로 사용하지 않음 |
| 에셋 | 원본 다운로드·해시, XML/mesh/USD 검사 어댑터 | trimesh·USD Python 도구 등을 형식에 맞게 선택. CAD 변환은 후속 환경 제작에서 검증 |
| 상태·카탈로그 | SQLite + FTS5, 원본 파일 저장소 | 단일 제작 흐름에 별도 DB 서버를 먼저 요구하지 않음 |
| 의미 검색 | sentence-transformers + BAAI/bge-m3의 dense embedding | 다국어 자료 검색. 모델 revision과 정규화 규칙을 함께 기록 |
| 벡터 검색 | 초기에는 NumPy cosine 검색 | 규모와 실제 지연을 측정한 후 별도 벡터 DB 도입 판단 |
| 계약 검증 | JSON Schema 검증기 | 에이전트 출력과 수집 응답의 필수 필드·참조 유효성 확인 |
| GPU 실행·계측 | Isaac Sim/Arena·YAM 어댑터, sim-runner·trace-recorder | 고정 런타임·실행 ID, 정책 관측과 검사 관측 분리 |
| 물리·채점 검사 | 구성요소 fixture·수치 검사·scorer-replay | 필수 대조군과 원시 기록으로 판정. fixture 성공은 로봇 데모와 구분 |
| Dream 최종 평가 | 전용 dream-evaluator와 고정 정책 어댑터 | task lineage별 1회 예약·요청 이력. 숨은 요청·retry는 설치 SDK에서 확인 |
| 사용자 검토 | 정적 HTML + JSON 보고서, 명시적 approval-handler | 실제 결과와 ReleaseManifest digest에 연결한 승인 |
| 제출·접수 | 공식 upload_media.py, Git·gh/REST, receipt-monitor | 승인 후 별도 checkout에서 실행. 응답 유실 시 외부 상태 확인 |

Python orchestration을 제안한다. 정확한 라이브러리 버전은 구현 시작 전 설치
호환성 시험과 lockfile로 고정한다. 지금 읽은 최신 문서의 버전이 Brev에 이미
설치되었다고 간주하지 않는다. FTS5 지원 여부와 임베딩 CPU 처리 시간도 확인한다.
원문 수집·검색·DB 작업에는 GPU를 필수로 두지 않고, 임베딩 GPU 사용은 물리
작업과 충돌하지 않는 별도 자원 정책으로 다룬다.

현재 대화의 Slack/Notion/웹 MCP 연결이나 로그인된 Chrome 상태가 Brev에 자동
승계되지는 않는다. 이들은 이번 설계 조사 도구다. 운영 파이프라인의 인증과
수집 도구는 독립적으로 설정한다. 웹 검색 제공자를 바꾸어도 아래 데이터 계약은
유지되도록 한다.

## 5. 공통 실행 계약

모든 역할 입력은 `schema_version`, `run_id`, `stage_id`, `input_snapshot`,
`source_refs`, `allowed_actions`를 갖는다. 출력은 `status`, `claims/results`,
`evidence_refs`, `unknowns`, `next_action`을 갖도록 제안한다.

- `status`: `ready / needs-evidence / conflict / failed` 등 단계별 허용 상태만.
- 원문·에셋은 ID와 해시로 참조한다. 에이전트가 임의의 로컬 경로를 만들면 검증 실패.
- 수정된 출력은 이전 결과를 덮어쓰지 않고 새 버전으로 기록한다.
- 복구는 저장된 실행과 실제 외부 요청 상태를 기준으로 한다. 모델 세션의 기억에 의존하지 않는다.
- DB 쓰기는 조정기에서 직렬화한다. 에이전트의 판단 결과와 카탈로그 원문을 분리한다.
- URL·문서·CAD 안의 문장은 분석 대상이며 실행 지시로 취급하지 않는다.
- 인증 파일은 카탈로그·Git·제출 산출물 밖에 둔다. 쿠키를 보고서에 포함하지 않는다.

세부 데이터·도구 계약은 세 문서에 기록한다.

- [산업 자료·공개 에셋·기존 제출·신규성 검사](sources-and-novelty.md)
- [환경·에셋 제작과 물리 검증](environment-and-validation.md)
- [데모·채점·Dream·승인 후 제출](demo-evaluation-and-submission.md)

구현의 의존관계·검증 순서는 [MVP 구현 계획](../../tasks/plan.md),
실제 도구·인증 준비와 사용자 결정은 [Brev 준비](brev-readiness.md)에 기록한다.
상세 작업은 모두 미착수이며 이 문서의 작성만으로 설치·실행을 시작하지 않는다.

## 6. 확인한 공식 자료

- [Codex 비대화식 실행](https://learn.chatgpt.com/docs/developer-commands#codex-exec)
- [Parallel Search](https://docs.parallel.ai/search/search-quickstart),
  [Extract](https://docs.parallel.ai/extract/extract-quickstart)
- [pypdf 텍스트 추출](https://pypdf.readthedocs.io/en/stable/user/extract-text.html)
- [Playwright 인증 상태](https://playwright.dev/python/docs/auth)
- [SQLite FTS5](https://www.sqlite.org/fts5.html)
- [BGE-M3 모델 카드](https://huggingface.co/BAAI/bge-m3),
  [SentenceTransformer API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)

위 자료는 도구의 지원 기능을 확인한 근거다. 본 프로젝트에서의 성능과
호환성이 검증됐다는 뜻은 아니며 모델·검색 제공자 선택은 검토 대상이다.
