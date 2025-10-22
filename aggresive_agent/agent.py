from google.adk.agents import LlmAgent
from .prompt import INSTRUCTION
from .tools import TOOLS

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='aggressive_investment_agent',
    description='Provide aggressive investment advise',
    instruction=INSTRUCTION,
    tools=TOOLS,
)
