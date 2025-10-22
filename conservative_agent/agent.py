from google.adk.agents import LlmAgent

from .prompt import INSTRUCTION
from .tools import TOOLS

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='conservative_investment_agent',
    description="Provide conservative investment advise",
    instruction=INSTRUCTION,
    tools=TOOLS,
)
