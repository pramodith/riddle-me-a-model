# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format, the answer section must be as concise as possible and all the thinking/reasoning should be within the think tags:
<think>
...
</think>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""
