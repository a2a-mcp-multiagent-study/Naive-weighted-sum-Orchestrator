# Navie weighted sum Orchestrator

## 구현된 에이전트

### Conservative Agent

-   보수적인 투자 조언자
-   미장, 국장에 대한 조언 제공
-   미국은 SPY, 한국은 코스피 지수와 비교해서 안정성 판단 후 조언

### Aggresive Agent

-   공격적 투자 조언자
-   미장, 국장에 대한 조언 제공
-   주가 변동 추이, 뉴스 기사,

## 실행 방법

### 환경 설정

-   환경 설치
    ```bash
        uv venv --python 3.12
        source ./.venv/bin/activate
        uv sync
    ```
-   conservative_agent > .env 파일 설정 (해당 폴더 내 .env.example 참고)
-   aggresive_agent > .env 파일 설정 (해당 폴더 내 .env.example 참고)
-   orchestator > .env 파일 설정 (해당 폴더 내 .env.example 참고)

### A2A 서버로 Agent 띄우기 (Orchestrator 실행하려면 필수)

띄워두면 [a2a sample ui](https://github.com/a2aproject/a2a-samples/tree/main/demo)에서도 연동 가능

```bash
uv run uvicorn run_agents:aggresive_app --host localhost --port 8001
uv run uvicorn run_agents:conservative_app --host localhost --port 8002
```

### 구글 ADK UI에서 에이전트 테스트 해보기

-   [참조 문서](https://google.github.io/adk-docs/get-started/python/#next-build-your-agent)
-   위에서 두 에이전트를 모두 잘 띄워두었다면 Orchestrator도 사용 가능

```bash
adk web --port 8000
```

### 기타 사항

-   langgraph 사용 X
-   google-adk만 사용
