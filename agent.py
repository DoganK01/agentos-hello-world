"""Hello-world agent for AgentOS.

Reads a single text input via `--input=<value>`, asks an OpenAI model
(via the `upsonic` library) to greet the user, and emits a structured
JSON envelope on the last line of stdout. AgentOS reads `OPENAI_API_KEY`
from the LLM config wired into the sandbox env.

Configure in AgentOS as:
    Command:   python agent.py --input={input:string}
    LLM:       openai/gpt-5.5
"""

import argparse
import asyncio
import json
import sys

from upsonic import Agent


async def run(user_input: str) -> str:
    """Send `user_input` to the model and return the assistant reply."""
    agent = Agent(
        "openai/gpt-5.5",
        system_prompt=(
            "You are a friendly hello-world demo agent. "
            "Greet the user warmly in one short sentence, "
            "echoing whatever they said back to them."
        ),
    )
    response = await agent.do_async(user_input)
    return str(response).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Hello-world AgentOS agent.")
    parser.add_argument("--input", required=True, help="User message to greet.")
    args = parser.parse_args()

    reply = asyncio.run(run(args.input))

    # Human-readable line goes to stdout first.
    print(reply)

    # AgentOS picks the last non-empty stdout line and tries json.loads on it.
    # That becomes the run's `result` field on the AgentRun row.
    print(json.dumps({"input": args.input, "reply": reply}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
