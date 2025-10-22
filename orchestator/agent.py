from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent


from .prompts import GLOBAL_INSTRUCTION, INSTRUCTION


agent1 = RemoteA2aAgent(
    name='conservative_investment_agent',
    description="Provide conservative investment advise",
    agent_card=(
        f"http://localhost:8001/{AGENT_CARD_WELL_KNOWN_PATH}"
    ),
)

agent2 = RemoteA2aAgent(
    name='aggressive_investment_agent',
    description='Provide aggressive investment advise',
    agent_card=(
        f"http://localhost:8002/{AGENT_CARD_WELL_KNOWN_PATH}"
    ),
)


root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="root_agent",
    global_instruction=GLOBAL_INSTRUCTION,
    instruction=INSTRUCTION,
    tools=[AgentTool(agent1), AgentTool(agent2)],
)
