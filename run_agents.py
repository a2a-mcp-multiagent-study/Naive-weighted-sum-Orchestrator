from google.adk.a2a.utils.agent_to_a2a import to_a2a
from conservative_agent.agent import root_agent as conservative_agent
from aggresive_agent.agent import root_agent as aggresive_agent


# Make your agent A2A-compatible
aggresive_app = to_a2a(aggresive_agent, port=8001)
conservative_app = to_a2a(conservative_agent, port=8002)
