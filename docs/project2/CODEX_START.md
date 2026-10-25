# Codex 시작 프롬프트

아래 본문을 기존 저장소가 열린 Codex 작업에 그대로 전달한다.

---

이 저장소의 Project 1에 이어 CUDA Project 2를 진행한다.

먼저 루트 AGENTS.md와 docs/project2/AGENTS_APPEND.md를 읽고, 추가 블록이 없으면 기존 AGENTS.md 끝에 한 번 추가해 줘. 기존 지침을 교체하지 마.

다음 순서로 읽어 줘:
1. docs/project2/PROJECT_SPEC.md
2. docs/project2/PLAN.md
3. 기존 PROJECT_SPEC.md, PLAN.md, FINAL_HANDOFF.md의 관련 부분
4. src/application_validation.py, src/timing.py, src/benchmark_harness.py, src/data.py, src/benchmark.py의 관련 부분
5. results/environment.json, results/analysis.md, results/audit.md
강의자료 검토와 적용할 최적화 선정은 프로젝트 관리 대화에서 완료했다. 강의 PDF 추가·열람은 요청하지 않으며 자료 부재를 blocker로 취급하지 않는다. 확정된 PROJECT_SPEC의 A–D 구현, correctness·timing 검증, 실험 실행 명령 준비와 결과 분석을 담당한다.

주제는 DistilBERT 추론의 비동기 파이프라인 및 동기화 최적화다. 구현 범위와 측정 계약은 docs/project2/PROJECT_SPEC.md를 따른다. 원래 모델·데이터·FP32·sequence length 64는 유지한다. A/B/C/D를 같은 CPU-visible full logits 계약으로 비교한다.

계획만 설명하고 멈추지 말고, M0 환경/작업 상태 확인 → M1 A/B/C → M2 D 및 bounded smoke까지 순서대로 진행해 줘. Routine 구현 선택은 스스로 처리하고 파일에 기록해. 기존 코드/실험 파일은 보존하고 새 코드와 results/project2/를 사용해.

현재 Python, GPU, package와 cache를 확인한 뒤 재사용해. 과거 환경 metadata를 현재 환경으로 그대로 복사하지 마. 다른 계정/경로로 옮겨진 venv나 cache가 그대로 작동한다고 가정하지 마. GPU 또는 cache가 없으면 가능한 정적 작업은 계속하고 blocker를 구체적으로 보고해.

Codex가 직접 실행할 것은 짧은 환경 확인, 정적 검사, 작은 correctness/timing smoke까지다. 전체 872-sample correctness, 384개 전체 benchmark, 큰 profiling, 설치·대형 다운로드는 기존 AGENTS 규칙대로 내가 서버 터미널에서 수행한다. 그 전에 CLI를 구현/검증하고 실제로 실행 가능한 정확한 명령, 작업 디렉터리, 예상 출력, 다음에 공유할 파일을 알려줘. 아직 없는 CLI를 이미 실행 가능한 것처럼 제시하지 마.

기존 Project 1 pageable-only 검사를 제거해서 pinned memory를 허용하지 마. 별도 Project 2 policy를 구현해. 과거 결과와 새 결과를 섞지 말고 baseline A도 같은 새 계약으로 다시 측정하도록 해.

정확성과 timing을 먼저 통과시키고, 속도 개선이나 overlap은 관측한 뒤에만 주장해. 더 느린 결과도 보존해. 다른 모델/precision/custom kernel/torch.compile/CUDA Graphs 등으로 범위를 자동 확장하지 마.

Milestone마다 docs/project2/PLAN.md의 현재 상태를 갱신하고 Completed / Validation / Issues / Files changed / Next step / Session·protocol 형식으로 간단히 보고해. 내가 이 보고를 프로젝트 관리 대화에 공유할 예정이다.

원격 저장소에는 이번 프롬프트만으로 push/merge하지 말고 로컬 변경과 검증을 진행해. 발표자료는 지금 만들지 마.
