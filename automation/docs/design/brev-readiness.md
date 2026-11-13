# Brev 사전 준비: 도구·인증·사용자 결정

상태: 준비 계획 초안. 2026-11-13 (Asia/Seoul).
이 문서는 설치·로그인·인스턴스 생성·GPU/Dream 실행 지시가 아니다.
이 문서의 일반 준비 계획과 후속 실제 점검 결과를 구분한다.
후속 환경 준비의 실제 점검은 별도 readiness 보고에 기록한다. CLI·전용 skill,
검토 대기 전 stop·수정 시 start, fork 작업 브랜치 보존은
[확정한 생명주기 설계](cli-skill-and-lifecycle.md)를 따른다.
구현 순서는 [plan.md](../../tasks/plan.md), 세부 작업은 [todo.md](../../tasks/todo.md)다.

## 1. 실행 배치 제안

```text
Mac
  Brev CLI + VS Code Remote SSH + 보고서 검토/명시적 승인
       ↓ SSH / 결과 파일 반출
Brev의 전용 automation 작업 공간
  CPU coordinator/agents/catalog + 일관된 실행 DB/원본 저장
       ↓ 검증된 subprocess/파일 계약
  고정 GPU simulator container + YAM + task
       ↓ 최종 평가 도구만 Dream 인증 사용
  평가 기록 → 최종 보고 → 승인 → publisher
       ↓ 독립 보존
  로컬/선택한 저장소: DB snapshot·원본·영상·lock·journal
```

CPU 조정기와 simulator Python 환경을 분리한다. 일반 Python의 pip upgrade나
새 PyTorch 설치로 simulator의 의존성을 덮지 않는다. CPU 조정기는 venv/uv,
GPU는 검증한 컨테이너를 우선 검토하며 실제 host/container 구조를 확인한 뒤 정한다.
컨테이너에 접속했다는 이유로 Docker host 권한이 있다고 가정하지 않는다.

계획 문서·소스는 지정된 로컬 automation 경로에서 관리하고, 원격에는 전용
checkout을 둔다. 기존 과자봉지 branch나 공용 서버 작업을 전환/덮어쓰지 않는다.

## 2. 먼저 준비할 도구

| 위치/시점 | 도구 | 확인할 사항 |
|---|---|---|
| Mac, T01 | 기존 Brev CLI, VS Code Remote SSH, Git | 설치 여부부터 확인. CLI 계정/조직·인스턴스 이름·SSH·결과 반출 경로 |
| Brev host/container, T01/T03 | NVIDIA driver, Docker, NVIDIA Container Toolkit | 실제 접근 경계, GPU 장치 노출, 선택한 Sim 버전과의 호환성 |
| Brev CPU, T02~T06 | Git/gh, Python venv/uv, Codex CLI | 비대화식 실행·schema 결과·에러/취소·자격증명 전달 제한 |
| Brev CPU, T04 | SQLite/FTS5, hash·JSON Schema 검증 도구 | 일관된 backup/restore, 중복 실행 소유권, schema와 참조 검사 |
| Brev CPU, T07~T10 | HTTP client, 검색 adapter, pypdf·PDF 렌더, ffmpeg/ffprobe | 원본 바이트·페이지/프레임·실패 상태 보존 |
| Brev CPU, 필요한 경우 | Playwright와 browser deps, OCR | board가 실제로 JS/인증을 요구하는지 먼저 확인. 불필요한 도구는 나중에 설치 |
| Brev CPU, T10 | sentence-transformers/BGE-M3, NumPy | revision 고정·CPU 메모리/지연, 전수 signature 대조와 결합 |
| Brev GPU, T03 이후 | Isaac Sim + Lab/Arena + YAM 고정 조합 | RTX 렌더·robot reset/action·RGB/state·contact 계측과 headless 실행 |
| Brev asset 작업, T12 필요 시 | USD Python API, trimesh, CAD 변환 도구 | STEP/mesh 변환·구멍/틈/잠금부·단위·headless. FreeCAD/Blender 등은 필요할 때 검증 |
| Brev 평가, T17 이후 | 고정 Dream adapter/SDK | 초기화·warmup·retry·세션 재생성의 요청 위치를 fake transport로 확인 |
| Brev 발행, T19 이후 | 검토한 공식 uploader, Git/gh·REST | 별도 제출 checkout, 허용 파일만 변경, 외부 응답 유실 복구 |

각 도구는 버전·설치 출처·lock을 남긴다. 초기에 모든 선택 도구를 설치하지 않는다.
Node.js 등 Codex CLI 배포 방식의 의존성은 선택한 버전의 설치 문서에 맞춘다.

GPU 후보는 L40S 등 **선택한 Isaac Sim 버전에서 RTX 렌더가 지원되는 GPU**다.
CUDA 학습용 GPU라는 사실만으로 적합하다고 판단하지 않는다. GPU 종류·VRAM·
CPU/RAM·디스크와 driver를 실제 배포 후 확인한다. 정확한 용량은 runtime 이미지·
shader/cache·에셋·영상 보존을 포함해 smoke test에서 측정한다.

현재 설계의 Sim 6.1 계열은 과거 YAM 경로에 근거한 후보다. 최신 NVIDIA
요구 사항 페이지가 6.2를 안내한다고 자동 업그레이드하지 않는다. 선택한 릴리스의
문서·실제 YAM revision을 함께 대조해 호환 조합을 고정한다.

## 3. 인증 준비와 실제 사용 시점

| 인증 | 준비할 것 | 처음 필요한 시점·경계 |
|---|---|---|
| Brev | Mac의 로그인 계정/조직과 대상 인스턴스 | T01. 이미 쓸 수 있는 CLI를 재설치/재로그인하지 않고 확인부터 |
| 코딩·판단 에이전트 | Codex 인증 방식과 사용 가능한 모델 | T02. ChatGPT 로그인 또는 API key. 현재 desktop 로그인이 원격에 자동 이전되지 않음 |
| 검색 | 선택한 검색 제공자의 API 인증 | T02/T09. Parallel 우선 제안, 다른 제공자도 동일 adapter 가능. desktop MCP 연결과 별개 |
| GitHub 읽기 | PR·댓글·revision 읽기용 인증 | T02/T07. rate limit·자기 팀/fork 읽기를 실제 점검 |
| board 읽기 | 공개 접근 또는 전용 browser session | T08. 인증이 필요한지 확인한 뒤 설정. 개인 Chrome 전체 프로필 복사 불필요 |
| asset registry | 선택한 에셋/이미지 저장소 접근 | T03/T12. 공개 다운로드면 불필요. NGC 등도 선택한 실제 이미지의 요구에 따름 |
| Dream | 실제 SDK가 요구하는 자격증명과 기존 사용/attempt 이력 | T17은 fake만, 실제 키는 T21 전용 도구에 전달. 인증 테스트 이름의 inference 없음 |
| 미디어 업로드 | RoboCurve 제출자/팀 등록 확인과 공식 upload 인증 | T19 모의 시험 후, T22 실제 제출에만 사용 |
| GitHub 발행 | 정확한 fork origin push와 upstream PR 작성이 가능한 인증 | T22. 읽기 collector와 발행 도구의 사용 범위를 분리해 검증 |

OpenAI 공식 문서는 프로그램 방식 Codex CLI에 API key 인증을 권장한다.
기존 ChatGPT 로그인으로 시작하려면 원격 device-code 로그인 가능 여부와
workspace 허용·사용 한도를 먼저 확인한다. 어느 경우든 계정 접근이 가능한지
실제 최소 호출로 확인하며 자동으로 다른 과금 방식에 전환하지 않는다.
이는 Dream과 다른 인증이며 Dream 누적 한도를 새로 논의하는 항목은 아니다.

키/토큰은 채팅·Git·보고서·실험 원본에 넣지 않는다. 원격의 비밀 저장 경로에서
필요한 도구에만 주입하고 agent subprocess는 허용한 환경변수만 상속한다.
Git push용 SSH와 GitHub API용 인증을 같은 기능으로 간주하지 않는다.
외부 공유·메일/Slack 발송 인증은 이번 MVP에 필요하지 않다.

## 4. 영속 상태와 보존

Brev 공식 문서는 workspace의 stop 후 보존을 안내하면서 **인스턴스 삭제 시
workspace도 삭제된다**고 명시한다. 실제 선택한 mount·볼륨과 별도 보존 경로를
확인한다. `/data` 같은 이름만으로 영속성을 판단하지 않는다.

보존 대상은 소스·lock·에셋·영상뿐 아니라 실행 DB, raw trace, source snapshot,
Dream 사용/attempt 기록, 외부 요청 journal, 사용자 승인과 제출 receipt다.
DB는 SQLite backup 등 일관된 방식으로 snapshot을 만들고 referenced 파일과
manifest/hash를 맞춘다. 원본 DB 파일만 복사하고 WAL을 버리지 않는다.

초기 권장안은 **GitHub에 소스/작은 manifest, Mac의 별도 보존 폴더에 실행
상태·원본·영상**이다. 원격 디스크 단독 보존은 기본안이 아니다. 사용자 선택 시
object storage로 교체할 수 있다. 예상 저장 용량이 커지면 자동 업로드 전에 알린다.
사용자는 후속 지시로 benchmark fork의 새 작업 브랜치에 결과를 origin push하는
방식을 확정했다. 로컬 독립 보존은 실행 DB·큰 원본·승인/요청 journal을 보완한다.

외부 요청 전후의 중요한 journal/attempt 상태도 복구 가능하게 반출한다.
VM이 갑자기 사라져 마지막 기록의 외부 처리 여부가 불명확하면 `unknown`으로
복구하고 실제 상태를 대조한다. 이전 사용 이력을 복구하지 못한 새 VM에서
Dream 사용액을 0으로 초기화하거나 자동 새 평가를 시작하지 않는다.

## 5. 사용자가 정할 사항

### 시작 전에 필요한 결정

| 결정 | 제안·선택지 | 필요한 이유 |
|---|---|---|
| 사용할 Brev 환경 | 현재 사용 가능한 인스턴스 이름을 지정. 공용/다른 실험 환경이면 분리 checkout·container | 진행 중 작업을 건드리지 않고 설치/실행할 범위 결정 |
| 에이전트 인증·모델 | 무인 운영에는 API key 권장. 기존 ChatGPT 로그인 활용도 검증 가능. 모델 ID는 접근 가능한 선택으로 명시 | 원격에서 판단·제작을 실행할 실제 경로 |
| 첫 조사 범위 | 데이터센터의 강체 교체/체결·정비, 또는 로봇 제작의 강체 조립 중 한 영역부터 | 후보 수집을 집중하되 중복 비교는 전 분야 유지 |
| 저장·발행 목적지 | automation 소스를 보관할 repo/branch, 제출 fork/author, 대용량 결과의 별도 보존 위치 확인 | 현재 폴더 위치와 Git 원격/등록은 서로 다른 정보 |

제출자 등록을 다시 하라는 요청은 아니다. 기존 등록·fork·remote를 읽어서 확인하고
없거나 불일치할 때만 해결한다. 검색 제공자는 우선 Parallel로 검증하되 사용 가능한
인증이 없을 때 선택을 요청한다. Dream·제출 인증은 필요한 마지막 단계까지 미룰 수 있다.

### 수치가 필요해질 때 확인할 기본값 제안

- 첫 후보 묶음은 3~5개, 제작은 하나. 수집 시간을 무조건 후보 개수로 제한하지 않는다.
- 최종 demo는 고정 제출 배치에서 깨끗한 reset 3회 성공. 현실 검증/다중 seed 일반화와 다름.
- 단계별 수리 계획 최대 3회, 같은 원인·같은 증거 2회 반복은 예외 보고.
- 보고는 HTML+JSON, 승인은 특정 digest에 대한 명시적 사용자 결정 도구.
- 중복 동기화는 시작/선정/제출 전 필수, 제작 중 가벼운 증분 30분·하루 한 번 재조정 제안.

이는 미확정 운영 값이다. 기술 세부 버전·CAD 도구·물리 오차·run limit는 자료와
task 특성에 따라 구현자가 검증해 정할 항목이며 사용자에게 모든 값을 묻지 않는다.
이미 확정된 Dream 전체 누적 한도·태스크당 평가 1회·승인 후 제출은 재선택하지 않는다.

## 6. 준비 완료 판정

ReadinessReport는 도구·인증·접근·런타임·보존 각 항목을
`verified / missing / blocked / not-needed-yet`로 구분한다.
로그인 안내를 받음, CLI가 설치됨, CUDA가 보임을 실제 end-to-end 실행과 구별한다.
실제 GPU/모델/제출 호출이 수행됐는지는 각각 별도로 기록한다.

## 확인한 공식 자료

- [Brev CLI 로그인·SSH·workspace 보존/삭제](https://docs.nvidia.com/brev/cli/getting-started)
- [Brev 배포 모드·기본 도구·VS Code](https://docs.nvidia.com/brev/concepts/environments)
- [OpenAI Codex 인증·headless 로그인](https://learn.chatgpt.com/docs/auth)
- [Isaac Sim 6.2 요구 사항](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)

공식 자료는 일반 지원 범위다. 현재 계정 권한·인스턴스 버전·driver·에셋 접근·
실제 inference 또는 제출 동작을 이번 계획 작성에서 검증했다는 뜻은 아니다.
