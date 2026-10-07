# Agentic RL Study: 프로젝트 & 학습 로드맵

> 목표: Agent의 행동 문제를 찾고, 가설을 세우고, environment·reward·eval을 직접 만든 뒤 SFT/RL로 학습하고, 결과를 분석해서 다음 training intervention을 설계하는 Agent Research Engineer가 되는 것.

---

## 0. 요약

- **플래그십 프로젝트는 하나만 한다.** 직접 만든 재현 가능한 웹/Computer-Use 환경(가칭 **WebOps-Gym**)에서 작은 오픈 모델(3B–8B)을 `prompt-only → SFT → DPO → multi-turn RL` 순서로 개선한다. 각 단계에서 eval과 failure analysis를 함께 한다.
- **RL 이론은 미리 공부하지 않는다.** Phase 1(3주) 동안 작은 코딩 과제에서 GRPO/PPO를 직접 구현하면서 필요한 개념만 익힌다. 나머지 개념은 그 개념이 필요해지는 Phase에서 배운다.
- **차별점은 알고리즘이 아니라 environment, grader, eval이다.** 모든 상태를 결정론적으로 snapshot/restore할 수 있는 환경을 만들면 다음이 가능해진다.
  - 같은 상태에서 여러 rollout을 돌리는 GRPO
  - step 단위 Monte-Carlo credit assignment
  - step 단위 preference pair 생성
  - 일부러 결함을 넣은 grader로 하는 reward hacking 실험
- **최종 산출물:** 공개 env와 task set, eval harness, 학습 코드, trajectory viewer, 그리고 연구 리포트 수준의 tech report(헤드라인 비교, reward hacking 사례, research loop 3회).

---

## 1. 설계 원칙

1. **사이클 전체를 한 번 빨리 돈 다음 깊게 판다.** `Task → Agent → Env → Trajectory → Reward → Training → Eval → Failure Analysis`를 거칠게라도 끝까지 한 번 돌리는 것이 각 부분을 완벽하게 만드는 것보다 우선이다.
2. **Eval과 grader가 학습보다 먼저다.** grader를 검증하기 전에는 학습 결과를 믿지 않는다. reward와 별도로 "진짜 성공"을 판정하는 strict grader와 사람 감사(human audit)를 항상 둔다.
3. **모든 주장에는 비교 실험이 붙어야 한다.** 단일 결과는 보고하지 않는다. baseline, 통제 변수, CI(신뢰구간)를 함께 제시한다.
4. **Just-in-Time 학습.** 개념은 그 개념 없이는 다음 실험을 해석할 수 없을 때 배운다. Phase마다 "이번에 배울 개념"을 명시했다.
5. **범위를 통제한다.** 앱 3개와 held-out 앱 1개만 만든다. Screenshot(VLM)과 Coding agent는 확장 트랙으로 둔다.

---

## 2. 도메인 선택: 왜 직접 만든 웹/Computer-Use 환경인가

| 기준 | 자체 웹 환경 (주 트랙) | 코딩 에이전트 (보조 트랙) | 기존 벤치마크만 사용 |
|---|---|---|---|
| CUA / Operator 포지션과의 연관성 | 매우 높음 | 중간 | 높음 |
| Environment 설계 역량 증명 | **직접 설계** | 중간 | 거의 없음 |
| Verifiable reward | DB 상태 비교로 확실 | unit test로 확실 | 벤치마크마다 다르고 일부는 불완전 |
| Reset / snapshot 제어 | **완전 제어** | Docker로 가능 | 어렵거나 느림 |
| 난이도와 horizon 조절 | 템플릿으로 자유롭게 | 제한적 | 고정 |
| Reward hacking 실험 설계 | grader 결함을 의도적으로 주입 가능 | test 수정 같은 자연 발생 사례 | 어려움 |
| 연산 비용 | 낮음 (텍스트 a11y tree) | 중간~높음 | 중간 |

**결정:**
- **주 트랙:** WebOps-Gym, 즉 결정론적인 미니 웹앱과 DB 상태 기반 grader.
- **Phase 1 워밍업:** 단일 턴 코딩 과제 (unit test reward). 코딩 에이전트에 대한 관심도 일부 반영된다.
- **일반화 평가:** 외부 벤치마크(BrowserGym 계열 / WebArena 일부 / MiniWoB++)는 학습에 쓰지 않고 held-out 일반화 평가에만 사용한다.

---

## 3. 시스템 아키텍처

```
          ┌───────────── Task Generator (템플릿 × 파라미터 × 난이도, split 고정) ─────────────┐
          ▼                                                                                   │
  Policy (vLLM 서빙) ──action──▶ WebOps-Gym Env (FastAPI+SQLite 앱 / Playwright) ──obs──▶ Policy
          │                          │  snapshot()/restore()  │ grade()
          ▼                          ▼                        ▼
   Trajectory Store (jsonl/parquet, schema 버전 관리)  ◀── Graders (strict / partial / step / buggy / LLM-judge)
          │
          ├──▶ Eval Harness (metrics, 난이도·카테고리별 분해, bootstrap CI, pass@k / pass^k)
          ├──▶ Failure Labeler (rule + LLM judge + human 캘리브레이션) ──▶ Research Notes
          └──▶ Dataset Builders (SFT / DPO pairs / RL prompts) ──▶ Trainers (SFT / DPO / GRPO / PPO)
                                                                               │
                                                     checkpoint ──────────────┘ (다시 Policy로)
```

### 3.1 Env 인터페이스 (Gymnasium 스타일)

```python
class WebOpsEnv:
    def reset(self, task_id: str, seed: int) -> tuple[Obs, Info]: ...
    def step(self, action: str) -> tuple[Obs, float, bool, bool, Info]:  # obs, reward, terminated, truncated, info
        ...
    def snapshot(self) -> StateHandle: ...            # DB + URL + browser storage + step count
    def restore(self, handle: StateHandle) -> Obs: ...  # credit assignment와 branching의 핵심
    def grade(self, grader: str = "strict") -> GradeResult: ...  # success, partial, subgoals, side_effects
```

### 3.2 Trajectory 스키마 (초안)

```json
{
  "traj_id": "...", "schema_version": 1,
  "task":   {"id": "...", "app": "shop", "template": "refund_order", "difficulty": "L2", "split": "train"},
  "policy": {"model": "...", "checkpoint": "...", "temperature": 1.0, "prompt_version": "v3"},
  "env":    {"version": "...", "seed": 17, "grader_version": "strict-v2"},
  "steps": [
    {"t": 0, "obs": "...", "model_output": "<raw, reasoning 포함>", "action": "click(42)",
     "action_valid": true, "env_error": null, "reward": 0.0, "state_hash": "...",
     "latency_ms": 820, "tokens_in": 2310, "tokens_out": 96}
  ],
  "outcome": {"success": false, "partial": 0.5, "grader_results": {"strict": 0, "ui_toast": 1},
              "termination": "done|max_steps|timeout|invalid_limit"},
  "labels":  {"failure": ["grounding"], "labeler": "llm-judge-v1", "human_verified": false}
}
```

---

## 4. 리소스 가정

- **GPU:** 현재 서버에는 RTX PRO 6000 Blackwell 96GB 5장과 RTX PRO 5000 48GB 1장이 있다. 공용 서버라 확인 시점에 4장이 사용 중이었다. 이 계획은 **96GB GPU 1–2장**을 쓸 수 있다고 가정한다.
- **모델 크기:** Phase 1은 0.5B–1.5B 모델을 full fine-tuning한다. Phase 3–5는 3B–4B 모델 full FT 또는 7B–8B 모델 LoRA로 진행한다 (예: Qwen 계열 Instruct).
- **Teacher (SFT 데이터 생성):** frontier API 모델 또는 vLLM으로 서빙한 32B급 오픈 모델을 쓴다. API 비용은 처음부터 기록한다.
- **Blackwell(sm_120) 주의:** CUDA 12.8 이상을 지원하는 PyTorch, vLLM, flash-attn 조합이 필요하다. Phase 0 첫날에 검증한다. 이 과정 자체가 GPU 디버깅 경험이 된다.
- **더 큰 규모의 RL:** 필요하면 Tinker(Thinking Machines의 학습 API) 같은 외부 서비스를 검토한다.
- **시간:** 주 20시간 기준 약 24주. 주 40시간이면 대략 절반으로 줄어든다.

---

## 5. 단계별 계획

각 Phase는 **목표 / 만들 것 / 이번에 배울 개념(JIT) / 실험 / 완료 기준**으로 구성된다.

### Phase 0: 인프라와 연구 습관 (1주차)

- **목표:** 이후 모든 실험이 재현 가능하도록 기반을 만든다.
- **만들 것:**
  - repo 구조 (§12)
  - Hydra config
  - W&B(또는 MLflow) tracking
  - seed 관리
  - vLLM 서빙 스크립트
  - `research_log.md` (날짜별 가설과 결과)
  - Blackwell 호환 환경 (Docker image 또는 uv lock)
- **배울 개념:** 없음. 도구만 익힌다.
- **완료 기준:** 같은 config로 같은 seed를 실행하면 같은 결과가 나온다. vLLM으로 Qwen 1.5B 서빙과 배치 생성이 동작한다.

### Phase 1: RL 기본기 — 단일 턴 코딩 과제로 GRPO/PPO 직접 구현 (2–4주차)

- **목표:** TRL이나 verl 없이 policy gradient를 직접 구현하고, 각 구성 요소가 왜 필요한지 실험으로 확인한다.
- **과제 (MiniCode):** 함수 시그니처와 docstring을 받아 구현을 작성한다. reward는 hidden unit test 통과 여부다.
  - 데이터는 MBPP류 데이터 또는 LLM으로 생성한 뒤 검증한 합성 데이터를 쓴다.
  - 모델은 0.5B–1.5B 코드/Instruct 모델을 쓴다.
- **구현 순서:**
  1. **REINFORCE:** 샘플링 → reward → `loss = -R · log π(y|x)`. 분산이 커서 학습이 불안정한 것을 직접 관찰한다.
  2. **Baseline / Advantage:** 프롬프트별 평균 reward를 빼고, group 정규화(GRPO)를 적용한다. 분산이 줄고 학습 곡선이 개선되는 것을 확인한다.
  3. **KL to reference와 PPO clip:** 한 배치로 여러 inner epoch를 돌리면 off-policy가 된다. 이때 importance ratio와 clipping이 필요한 이유를 확인한다.
  4. **PPO + value head (GAE):** critic을 둔 방식과 GRPO를 메모리, 안정성, 속도 측면에서 비교한다.
  5. **TRL GRPOTrainer와 교차 검증:** 같은 설정에서 곡선이 noise 범위 안에서 일치하는지 확인한다.
- **배울 개념:** MDP(token-level과 sequence-level), policy, return, log-derivative trick, baseline, advantage, value function, GAE, GRPO, PPO clip, KL regularization, on-policy와 off-policy, importance sampling, entropy collapse, length bias, zero-advantage group.
- **실험:**
  - (a) baseline 없음 vs 평균 baseline vs group 정규화
  - (b) group size G = 2/4/8/16
  - (c) KL 계수 sweep
  - (d) learning rate가 너무 클 때의 붕괴 양상 (entropy, KL, grad norm, 응답 길이 로그)
  - (e) **미니 reward hacking:** 프롬프트에 test를 보여주고 visible test로만 reward를 준다. 출력 하드코딩이 생기는지, visible과 hidden 통과율 차이가 벌어지는지 측정한다.
- **완료 기준:** 다음 질문에 그래프로 답할 수 있다.
  - group 정규화는 무엇을 해결하는가?
  - clip이 없으면 무엇이 깨지는가?
  - 모든 샘플이 0점이거나 모두 1점인 group은 왜 학습 신호가 없는가?
  - 노트 `notes/phase1_policy_gradient.md`를 작성한다.

### Phase 2: Environment, Grader, Eval Harness 구축 (5–8주차) — 핵심 자산

- **목표:** 재현 가능하고 snapshot이 가능한 multi-step 웹 환경, 검증된 grader, eval harness를 만들고 prompt-only baseline을 측정한다.
- **앱:** FastAPI, SQLite, 서버 렌더링 HTML로 만든다.
  - `shop` (주문, 재고, 환불)
  - `tracker` (이슈, 라벨, 담당자)
  - `calendar` (회의실 예약, 충돌)
  - `crm` (**held-out:** 학습 데이터에 절대 쓰지 않는다)
- **Observation:**
  - 가지치기한 accessibility tree에 `[id]`를 태깅한다.
  - URL, 직전 action 결과나 에러 메시지, 남은 step 수를 포함한다.
  - (확장) screenshot과 Set-of-Marks.
  - 토큰 예산을 관리한다.
- **Action space:** `click(id)`, `type(id, text)`, `select(id, option)`, `scroll(dir)`, `back()`, `goto(url)`(허용 목록만), `answer(text)`, `done()`.
- **환경 동작 요구사항:**
  - **결정론:** 에피소드마다 DB snapshot을 복사한다. 시계는 고정한다. 데이터는 seed 기반으로 생성한다. worker마다 앱 인스턴스를 따로 둔다.
  - **Replay 검증:** action 시퀀스를 다시 실행하면 같은 `state_hash`가 나와야 한다. 이것을 CI 테스트로 둔다.
  - **Snapshot / restore:** 에피소드 중간 상태를 저장하고 복원할 수 있어야 한다. Phase 4–5의 branching과 MC credit assignment에 필요하다.
  - **종료 조건:** `done/answer`, `max_steps`, wall-clock timeout, 연속 invalid action K회.
  - **Invalid action:** 상태를 바꾸지 않고 에러 observation을 돌려준다. 횟수는 카운트한다.
  - **변형:** 레이아웃 변형, 라벨 동의어, 페이지네이션, 팝업이나 모달, 지연 로딩을 둔다. robustness와 recovery 평가용이다.
- **Task:**
  - 앱마다 템플릿 10–15개를 만들고, 템플릿마다 파라미터를 바꿔 인스턴스 수천 개를 생성한다.
  - 난이도:
    - L1: 1–3 step, 조회나 단일 조작
    - L2: 4–10 step, 검색과 폼 작성
    - L3: 10–25 step, 다중 조건이나 반복 작업
    - L4: 25 step 이상, 앱 간 연계 (long-horizon)
  - **Split:** `train / val / test-ID (같은 템플릿, 새 파라미터) / test-OOD-template / test-OOD-app (crm) / external`.
- **Graders:**
  - `strict`: 기대 DB diff가 모두 반영되었고 금지된 부작용(다른 주문을 환불하는 등)이 없어야 성공이다.
  - `partial`: 충족한 subgoal 비율에서 부작용 패널티를 뺀다.
  - `step`: progress 신호 (subgoal 달성 시점).
  - `answer`: 정규화한 exact match, 숫자는 허용 오차 내 일치.
  - `llm_judge`: 자유 서술형 답변용. 다른 grader와의 비교 대상이기도 하다.
  - **의도적 결함 grader** (Phase 5 실험용):
    - `ui_toast`: 화면에 "성공" 문구가 보이면 성공 처리
    - `exists_only`: 레코드가 존재하는지만 확인하고 필드 값은 보지 않음
    - `judge_final_msg`: agent의 최종 메시지만 보고 판정
    - `substring_answer`: 정답이 답변에 부분 문자열로 포함되면 성공 처리
- **Eval harness:**
  - 비동기 병렬 rollout (asyncio, Playwright context 여러 개)과 중단 후 재개를 지원한다.
  - 지표: success, partial, 평균 trajectory 길이, action 수, invalid action rate, recovery rate, latency, token 수, 비용.
  - 난이도, 앱, 템플릿, failure category별로 분해한다.
  - task 단위 bootstrap 95% CI를 계산한다.
  - pass@1(n개 샘플 평균), pass@k, **pass^k**(k번 모두 성공하는 비율, 신뢰성 지표)를 계산한다.
- **Grader 검증:**
  - trajectory 150–200개를 직접 라벨링한다.
  - 각 grader의 precision과 recall(false positive / false negative success)을 계산한다.
  - LLM-judge와 strict grader의 일치도(Cohen's κ)를 계산한다.
- **Baseline:**
  - frontier API 모델 1–2개 (상한선 참고용)
  - 오픈 모델 3B, 8B prompt-only
  - task마다 n = 4–8 샘플
- **Failure analysis v0:** §8의 taxonomy로 실패 100개를 라벨링해 첫 failure 분포를 얻는다.
- **배울 개념:** POMDP (state와 observation의 차이), 부분 관찰성, terminated와 truncated의 차이, sparse reward, eval 통계 (bootstrap, paired 비교, pass@k와 pass^k), grader validity.
- **완료 기준:**
  - replay 결정론 100%
  - grader precision/recall 표
  - baseline 결과표 (난이도별, CI 포함)
  - 첫 failure 분포 차트

### Phase 3: Trajectory 파이프라인과 SFT (9–11주차)

- **목표:** 성공 trajectory로 SFT를 하고, **SFT만으로는 왜 부족한지**를 데이터로 보인다.
- **데이터 만들기:**
  1. teacher로 train task에 rollout을 돌린다.
  2. strict grader로 성공한 것만 남긴다 (rejection sampling).
  3. 품질 필터를 적용한다: 루프, 과도한 invalid action, 불필요하게 긴 경로를 제거한다.
  4. multi-turn chat 형식으로 변환한다. loss는 assistant 턴에만 건다. 학습과 추론에서 같은 history 및 observation 처리 방식을 쓴다.
- **학습:** 3B–4B full FT 또는 7B–8B LoRA.
- **실험:**
  - (a) 데이터 규모 scaling: 100 / 300 / 1k / 3k trajectories
  - (b) 성공 trajectory만 쓰기 vs 성공 + 오류 후 회복한 구간 포함
  - (c) teacher distillation vs **Expert Iteration / RFT** (자기 모델로 k개 샘플링, 성공만 남겨 재학습, 2–3라운드). RFT는 binary reward로 걸러서 하는 behavior cloning이므로 RL로 넘어가는 다리 역할을 한다.
  - (d) **Compounding error 진단:**
    - step index별 이탈률을 측정한다.
    - perturbation suite: t번째 step에 off-path 상태(엉뚱한 페이지, 실패한 submit)를 주입하고 recovery rate를 잰다.
- **배울 개념:** behavior cloning, covariate shift와 compounding error, DAgger, rejection sampling, expert iteration (STaR, ReST), synthetic data, loss masking, chat template.
- **완료 기준:**
  - prompt-only, SFT, RFT의 비교표
  - "SFT 모델이 실패하는 상태"에 대한 정량 증거 (recovery rate, step별 이탈 곡선)

### Phase 4: Preference Optimization (12–13주차)

- **목표:** offline preference 학습의 효과와 한계를 RL과 비교할 기준선을 만든다.
- **Pair 구성:**
  - (i) **Trajectory-level:** 같은 task에서 성공한 trajectory와 실패한 trajectory.
  - (ii) **Step-level:** snapshot한 상태 s_t에서 action k개로 분기하고, 각각 MC rollout으로 성공률을 추정한다. 성공률이 높은 action을 선호 쪽으로 둔다. 결정론적 env의 장점을 활용하는 방법이다.
  - (iii) **RLAIF:** LLM judge로 효율성과 불필요한 action 여부를 평가한다. verifiable하지 않은 품질을 다룬다.
- **실험:**
  - SFT vs SFT+DPO(traj) vs SFT+DPO(step)
  - (선택) 작은 reward model(Bradley-Terry)을 학습하고 정확도를 측정한 뒤 best-of-n reranking에 사용한다.
- **배울 개념:** Bradley-Terry, RLHF 파이프라인 (SFT → RM → PPO), DPO의 implicit reward, RLAIF, offline과 online의 차이, best-of-n, over-optimization과 likelihood displacement.
- **완료 기준:** DPO가 어떤 failure category를 줄였고 어떤 것은 못 줄였는지 분해한 결과.

### Phase 5: Multi-turn Agent RL — 핵심 실험 (14–18주차)

- **목표:** 실제 env에서 online RL로 성능을 개선한다. reward 설계, reward hacking, credit assignment를 실험으로 다룬다.
- **5a. 직접 구현 (1.5주):**
  - Phase 1 코드를 multi-turn으로 확장한다.
  - vLLM rollout과 비동기 env worker를 연결한다.
  - **observation 토큰은 loss에서 마스킹**하고, 모델이 생성한 토큰에만 loss를 건다.
  - group = 같은 task와 같은 reset 상태에서 G번 rollout. trajectory-level GRPO로 학습한다.
  - L1–L2 task와 작은 모델로 디버깅한다.
- **5b. 프레임워크로 확장:**
  - 후보: verl, SkyRL, rLLM, OpenRLHF의 agent loop 기능, 또는 Tinker.
  - 선택 기준은 env를 붙이기 쉬운가다.
  - 다룰 엔지니어링: 비동기 rollout, rollout과 학습의 overlap, policy staleness, weight sync, colocated 배치, long-context 메모리 관리 (gradient checkpointing, packing), checkpoint/resume.
  - 프레임워크는 빠르게 변하므로 시작 시점의 최신 상태를 확인한다.
- **핵심 실험** (모두 같은 SFT init, 같은 task split, 같은 rollout 예산):

| ID | 비교 | 무엇을 보는가 |
|---|---|---|
| **E1 (헤드라인)** | SFT vs SFT+DPO vs SFT+RL (+RFT) | 난이도별 성공률, recovery rate, 일반화 (OOD template, app, 외부 벤치마크) |
| **E2** | binary terminal vs partial credit vs shaped (step progress) vs LLM-judge reward | **strict grader 기준** 진짜 성공률, 샘플 효율, hack rate |
| **E3** | 결함 grader (`ui_toast`, `exists_only`, `judge_final_msg`)로 학습 | proxy reward는 오르는데 strict success는 정체하거나 떨어지는가, exploit 유형 분류, 탐지 방법, 완화 방법 |
| **E4** | credit assignment: trajectory-level GRPO vs **turn-level MC advantage (snapshot 기반, VinePPO 방식)** vs learned critic (turn-level GAE) vs progress shaping | horizon 버킷(L1–L4)별 성능, 샘플 효율, 연산 비용 |
| **E5** | curriculum: uniform vs 난이도 순서 vs adaptive (성공률이 0과 1 사이인 task만 샘플링, DAPO식 dynamic sampling) | 샘플 효율, 최종 성능 |
| **E6** | staleness 0 / 1 / 4 (비동기 정도) | throughput과 안정성의 trade-off |
| E7 (선택) | RL vs **on-policy distillation** (student가 샘플링하고 teacher의 token logprob으로 학습) | 같은 연산 대비 효율 |

- **E3의 reward hacking 탐지와 완화:**
  - 탐지:
    - proxy grader와 strict grader 점수 차이 모니터링
    - side-effect checker
    - 매 N step마다 trajectory 무작위 감사
    - judge와 programmatic grader의 불일치율
  - 완화:
    - grader 수정
    - 부작용 패널티
    - grader 앙상블
    - KL 강도 조절
- **배울 개념:** credit assignment, MC와 TD, GAE λ, outcome reward와 process reward, potential-based reward shaping (최적 정책을 보존하는 조건), Goodhart's law, reward misspecification, exploration (temperature, entropy, zero-advantage), RL 불안정성 진단 지표 (KL, entropy, clip fraction, grad norm, 응답 길이).
- **완료 기준:**
  - E1 결과표 (CI 포함)
  - 문서화된 reward hacking 사례 1개 이상 (탐지와 완화 포함)
  - E4에서 horizon별 credit assignment 비교 그래프

### Phase 6: Research Loop ×3과 일반화 (19–22주차)

각 사이클을 `Observed failure → Hypothesis → Experiment → Intervention → Evaluation → Conclusion` 형식의 research note로 남긴다. 템플릿은 §11에 있다. 예시:

- **Cycle A: Recovery failure**
  - 관찰: RL 모델도 에러를 만난 뒤 회복하지 못한다.
  - 가설: 학습 rollout이 에러 상태를 거의 방문하지 않는다.
  - 개입: rollout 중 팝업이나 실패 응답을 주입한다. perturbed snapshot에서 rollout을 시작한다.
  - 평가: perturbation suite에서의 recovery rate.
- **Cycle B: Long-horizon drift**
  - 관찰: L4 task에서 task 제약을 잊거나 이미 끝낸 subgoal을 반복한다.
  - 가설: history truncation 때문에 state tracking이 실패한다.
  - 개입 (system): 구조화된 메모 도구.
  - 개입 (training): L3에서 L4로 가는 curriculum.
  - 실험: memory 유무 × RL 유무의 2×2 ablation.
- **Cycle C: Grounding/Perception**
  - observation ablation: a11y tree vs screenshot+SoM vs 둘 다 (VLM 트랙).
  - 또는 UI 변형에 대한 일반화.
- **일반화 리포트:** test-ID, OOD-template, OOD-app(crm), 외부 벤치마크의 성능 하락 폭을 비교한다.

### Phase 7: 패키징과 Tech Report (23–24주차)

- **Tech report** (arXiv/블로그 형식):
  - Motivation
  - Environment & Graders (검증 결과 포함)
  - Training (SFT/DPO/RL)
  - Results (E1–E5)
  - Reward Hacking Case Study
  - Credit Assignment
  - Failure Analysis & Research Loops
  - Limitations
- **공개:** env와 task set, eval harness (원커맨드 재현), trajectory viewer (Streamlit/Gradio), 결과 대시보드, 5분 데모 영상.
- **README:** 결과 요약표, 핵심 그림 3–4개, 재현 방법.

### 확장 트랙 (시간이 남을 때)

- **Coding Agent env:** 같은 `Env` 인터페이스 위에 Docker sandbox, 파일 편집, 테스트 실행을 구현한다. 인터페이스의 일반성을 보여준다.
  - "test 파일 수정"처럼 자연 발생하는 reward hacking 사례를 다룬다.
- **Screenshot-only CUA:** VLM (예: Qwen-VL 계열)으로 좌표 기반 action을 쓴다. Anthropic Computer Use 방식에 더 가깝다.

---

## 6. 실험 설계 규칙 (모든 비교에 적용)

- **통제할 것:**
  - base model과 SFT init checkpoint
  - task split (고정 파일과 해시)
  - prompt/observation 포맷 버전, `max_steps`
  - eval decoding (temperature, n)
  - grader 버전 (eval grader는 동결)
  - **학습 예산** (rollout 수, 토큰 수, GPU 시간 중 무엇을 맞췄는지 명시)
- **보고 방식:**
  - task 단위 bootstrap 95% CI를 쓴다.
  - 같은 task에 대해서는 paired 비교를 한다 (paired bootstrap 또는 McNemar).
  - 학습은 가능하면 seed 3개로 돌린다. 1개만 돌렸으면 명시한다.
- **누수 방지:**
  - 프롬프트 튜닝과 하이퍼파라미터 선택은 val에서만 한다.
  - test-OOD 템플릿과 앱은 teacher 데이터 생성에도 쓰지 않는다.
- **진짜 개선인지 evaluator artifact인지 구분하기 (Q9):**
  - 서로 독립적인 grader 2개 이상으로 평가한다 (strict, 대체 grader, human audit 샘플).
  - 개선이 grader가 약한 task에만 몰려 있는지 확인한다.
  - 외부 벤치마크에서도 같은 방향으로 움직이는지 확인한다.
  - 개선된 trajectory를 무작위로 골라 직접 읽는다.
- **실패한 실험도 기록한다.** 음성 결과도 research log와 report에 남긴다.

---

## 7. Ablation 매트릭스

| 비교 | 고정 | 주 지표 | 사전 가설 (실험으로 검증) |
|---|---|---|---|
| Prompt-only vs SFT | base, task split | 난이도별 success | SFT가 L1–L2를 크게 올리지만 L3–L4는 제한적이다 |
| SFT vs SFT+RL | SFT init, 예산 | success, recovery, OOD | RL의 이득은 recovery와 L3+에 집중된다 |
| Sparse vs shaped reward | 그 외 모두 | strict success, hack rate | shaping은 초기 학습을 빠르게 하지만 잘못 설계하면 exploit을 부른다 |
| A11y tree vs screenshot | 모델 크기, task | success, grounding failure 비율 | screenshot에서는 grounding failure가 늘어난다 |
| No memory vs memory | policy, task | L4 success, drift 지표 | memory는 L4에서만 유의미하다 |
| Curriculum 종류 | 예산 | 샘플 효율 | adaptive가 zero-advantage 낭비를 줄인다 |
| Rollout 전략 (G, temperature, staleness) | 예산 | 안정성, throughput | G가 작으면 신호가 부족하고, staleness가 크면 불안정해진다 |

---

## 8. Failure Taxonomy (운영 정의)

| Category | 판정 기준 | 자동 탐지 신호 | 대표 개입 |
|---|---|---|---|
| Perception | 필요한 정보가 obs에 있었는데 못 읽거나 잘못 읽음 | answer 오류이면서 정답 문자열이 obs에 있음 | obs 포맷, 요약 |
| Grounding | 의도는 맞는데 엉뚱한 element를 조작함 | reasoning의 대상과 클릭한 id의 텍스트가 불일치 | SoM, id 설계, grounding SFT |
| Planning | subgoal 순서나 분해가 잘못됨 | 필요한 페이지를 방문하지 않음, 순서 위반 | plan-then-act, RL |
| State tracking | 이미 완료한 일을 반복하거나 진행 상태를 오인함 | 같은 `state_hash`로 루프, 중복 생성 | memory 도구, history 설계 |
| Tool-use | action 문법이나 인자 오류 | `action_valid = false` | 포맷 SFT, constrained decoding |
| Execution | 유효한 action이 env 측에서 실패함 | env_error, timeout | wait/retry, 환경 수정 |
| Recovery | 에러 뒤에 정상 경로로 돌아오지 못함 | 에러 이후 성공률 | 에러 주입 RL |
| Long-horizon drift | step이 쌓이며 task 제약을 잊음 | 후반 step의 제약 위반, step 수에 따른 성공률 감소 | curriculum, memory |
| Reward/grader failure | grader 판정과 실제 성공이 불일치 | grader 간 불일치, human audit | grader 수정 |

**라벨링 파이프라인:**
1. rule 기반 탐지기를 먼저 적용한다.
2. LLM judge가 taxonomy에 따라 라벨을 붙인다. 근거 step도 함께 출력한다.
3. 사람이 50–100개를 캘리브레이션 라벨링해서 LLM judge와의 κ를 보고한다. LLM 라벨러 자체도 검증 대상이다.

---

## 9. 엔지니어링 스택

| 영역 | 도구 |
|---|---|
| 모델, 학습 | PyTorch, HF Transformers / TRL / PEFT, FSDP 또는 DeepSpeed, (확장) verl / SkyRL / Tinker |
| 추론, rollout | vLLM (배치, 비동기), asyncio, (필요시) Ray |
| 환경 | FastAPI, SQLite, Jinja, Playwright (async, headless Chromium), Docker (worker 격리) |
| 데이터 | pydantic 스키마, jsonl/parquet, DuckDB (분석), HF Datasets |
| 실험 관리 | Hydra, W&B, git commit 해시 기록, checkpoint/resume |
| 분석, 시각화 | pandas, matplotlib/plotly, Streamlit (trajectory viewer) |
| 품질 | pytest (env 결정론, grader 단위 테스트, replay), pre-commit |

**공용 서버에서 지킬 것:** 모든 학습은 중단 후 재개가 가능해야 한다. 각 실행은 GPU를 몇 장, 어느 장치에서, 몇 시간 썼는지 기록한다.

---

## 10. 12개 질문과 이를 뒷받침할 증거

| # | 질문 | 증거가 되는 산출물 |
|---|---|---|
| 1 | Agent는 왜 실패하는가? | §8 taxonomy 분포 (Phase 2, 3, 5, 6) |
| 2 | RL이 이 문제를 해결할 수 있는가? | E1의 category별 개선 분해 |
| 3 | SFT만으로 왜 부족한가? | Phase 3 compounding error와 recovery 실험 |
| 4 | Reward를 어떻게 정의하는가? | E2, grader 설계 문서 |
| 5 | Reward 설계가 잘못된 것을 어떻게 아는가? | E3, grader precision/recall, 불일치 모니터링 |
| 6 | Long-horizon credit assignment는? | E4 (MC turn-level, critic, shaping 비교) |
| 7 | 어떤 trajectory로 학습하는가? | Phase 3 필터링과 RFT 실험, Phase 4 pair 구성 |
| 8 | 성능을 어떻게 객관적으로 평가하는가? | eval harness, CI, pass^k, OOD split |
| 9 | 진짜 개선인지 evaluator artifact인지 어떻게 아는가? | §6 규칙, 다중 grader, 외부 벤치마크 |
| 10 | 실패 데이터를 다음 iteration으로 어떻게 연결하는가? | Phase 6 research loop |
| 11 | 새 capability의 benchmark는 어떻게 설계하는가? | WebOps-Gym 설계 문서 (split, 난이도, grader 검증) |
| 12 | 아이디어를 scalable 파이프라인으로 만드는 방법은? | 비동기 rollout, 프레임워크 이식, throughput 측정 |

---

## 11. Research Note 템플릿

```markdown
# [Cycle X] <제목>
- Observed failure: (정량 근거, 예시 trajectory 링크)
- Hypothesis: (반증 가능하게)
- Prediction: (가설이 맞다면 어떤 지표가 어떻게 움직여야 하는가)
- Experiment: (비교군, 통제 변수, 예산, seed)
- Intervention: (training / system / env / reward 중 무엇)
- Result: (표, 그림, CI)
- Conclusion: (가설 지지 / 기각 / 불확실, 대안 설명)
- Next: (다음 개입)
```

---

## 12. Repo 구조 (계획)

```
Agentic-RL-Study/
├── ROADMAP.md
├── research_log.md
├── notes/                # Phase별 JIT 개념 노트
├── rl_from_scratch/      # Phase 1
├── webops_gym/           # Phase 2: apps/{shop,tracker,calendar,crm}, tasks/, graders/, env.py
├── agent/                # 프롬프트, action parser, policy wrapper
├── rollout/              # 비동기 rollout worker, trajectory schema와 저장
├── train/                # sft/, dpo/, rl/
├── eval/                 # harness, metrics, 통계, 리포트 생성
├── analysis/             # failure labeler, trajectory viewer
├── configs/              # Hydra
└── reports/              # research notes, 그림, tech report
```

---

## 13. 리스크와 대응

| 리스크 | 대응 |
|---|---|
| 작은 모델의 성공률이 0에 가까워 RL 신호가 없음 | SFT init을 쓰고, L1부터 curriculum을 적용하고, dynamic sampling을 쓴다 |
| Playwright 때문에 rollout throughput이 병목 | 가벼운 HTML, 다수의 browser context, 프로파일링. 병목을 측정한 결과 자체도 산출물이 된다 |
| Grader 버그가 모든 결과를 오염시킴 | grader 단위 테스트와 human audit을 Phase 2에서 끝내고, eval grader를 동결한다 |
| 범위 확장 (앱, 모달리티 추가) | 앱 3+1개로 고정하고, 확장 트랙은 Phase 7 이후에 한다 |
| 공용 GPU 경합 | checkpoint/resume, 작은 모델로 먼저 디버깅, 큰 실행은 예약 |
| 프레임워크 API 변화 | Phase 5a에서 직접 구현한 루프를 기준으로 두고, 프레임워크는 교체 가능하게 연결한다 |

---

## 14. Phase별 Just-in-Time 읽기 목록

필요해질 때만 읽는다. 최신 후속 연구는 시작 시점에 다시 확인한다.

- **Phase 1:**
  - OpenAI Spinning Up — *Intro to Policy Optimization*
  - Schulman et al., *PPO* (2017)
  - Schulman et al., *GAE* (2015)
  - DeepSeekMath (GRPO, 2024)
  - DAPO (2025)
  - *Understanding R1-Zero-Like Training* (Dr. GRPO, 2025)
  - Lilian Weng, *Reward Hacking in RL* (2024)
- **Phase 2:**
  - WebShop (2022)
  - MiniWoB++
  - WebArena (2023)
  - BrowserGym / WorkArena
  - OSWorld (2024)
  - τ-bench (pass^k, 2024)
- **Phase 3:**
  - Ross et al., *DAgger* (2011)
  - STaR (2022)
  - ReST / ReST-EM (2023)
- **Phase 4:**
  - InstructGPT (2022)
  - Constitutional AI (RLAIF, 2022)
  - DPO (2023)
  - *Let's Verify Step by Step* (PRM, 2023)
- **Phase 5:**
  - Ng et al., *Potential-based Reward Shaping* (1999)
  - Tülu 3 (RLVR, 2024)
  - VinePPO (2024)
  - ArCHer (2024)
  - WebRL (2024)
  - SWE-Gym (2024)
  - RAGEN (2025)
  - verl / SkyRL / rLLM 문서
  - Thinking Machines 블로그 (*LoRA Without Regret*, *On-Policy Distillation*)
- **Phase 6–7:**
  - OpenAI Operator / CUA system card
  - Anthropic Computer Use 문서와 *Building Effective Agents*
  - DeepMind, *Specification Gaming* (2020)
