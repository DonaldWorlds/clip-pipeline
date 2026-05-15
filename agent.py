from pydantic_ai import Agent
from models import ClipDecision

agent = Agent(
    "openai:gpt-4.1-mini",
     "anthropic:claude-sonnet-4",
    result_type=ClipDecision
    result_type=ClipDecision
)



result = agent.run_sync("""
Analyze this clip request.

Return:
- clip start
- clip end
- layout
- title

Clip context:
Funny gaming rage moment at 52 seconds.
""")

ClipDecision(
    clip_start=52.0,
    clip_end=88.0,
    layout='gameplay_bottom',
    title='He Completely Lost It'
)



STEP 5 — Automatic Validation

If AI returns garbage:

"the clip should maybe start around..."

PydanticAI rejects it automatically.

THIS is huge.

STEP 6 — Add Retry Logic

PydanticAI can retry invalid outputs automatically.

Example:

agent = Agent(
    "anthropic:claude-sonnet-4",
    result_type=ClipDecision,
    retries=3
)