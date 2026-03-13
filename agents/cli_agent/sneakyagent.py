from openai import OpenAI
from pydantic import BaseModel, Field
import subprocess
import requests
import datetime
import json
from typing import Optional

local_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# ── Tools ──────────────────────────────────────────────────────────────────────

def write_file(path_and_content: str) -> str:
    try:
        if "|||" not in path_and_content:
            return "Invalid format. Use: filepath|||content"
        filepath, content = path_and_content.split("|||", 1)
        filepath = filepath.strip()

        dir_part = "\\".join(filepath.replace("/", "\\").split("\\")[:-1])
        if dir_part:
            subprocess.run(f'mkdir "{dir_part}"', shell=True, capture_output=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File created: {filepath} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def generate_code(prompt: str) -> str:
    """
    Calls the local model WITHOUT JSON format constraint
    so it can generate long code freely.
    """
    print(f"🖊️  Generating code for: {prompt[:60]}...")
    try:
        response = local_client.chat.completions.create(
            model="qwen2.5:3b",
            max_tokens=8096,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert programmer. "
                        "Return ONLY raw file content. "
                        "No explanations. No markdown. No backticks. "
                        "Just the pure code that goes directly into the file."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        code = response.choices[0].message.content.strip()

        # strip accidental markdown backticks if model adds them
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:-1])  # remove first and last line

        print(f"✅  Generated {len(code)} chars")
        return code
    except Exception as e:
        return f"Code generation error: {str(e)}"

def run_command(cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip()
        error = result.stderr.strip()
        if result.returncode != 0:
            return f"Command failed: {error or 'no error message'}"
        return output if output else f"Command executed: {cmd}"
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except Exception as e:
        return f"Error: {str(e)}"

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text.strip()}"
        return f"Could not fetch weather for {city}"
    except requests.RequestException as e:
        return f"Network error: {str(e)}"

def get_time(timezone: str = "local") -> str:
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Invalid expression."
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

available_tools = {
    "get_weather": get_weather,
    "get_time": get_time,
    "calculator": calculator,
    "run_command": run_command,
    "write_file": write_file,
    "generate_code": generate_code,
}

# ── System Prompt ──────────────────────────────────────────────────────────────

system_prompt = """
You are a helpful general-purpose AI assistant on Windows PowerShell.
Respond ONLY in valid JSON. No text outside JSON ever.

THREE valid steps: THINK, TOOL, OUTPUT.

## TOOLS

1. get_weather   → input: city name
2. get_time      → input: "local"
3. calculator    → input: math expression
4. run_command   → input: powershell command
5. generate_code → input: description of what code to generate
   IMPORTANT: Use this to generate file contents. Never write code yourself inside JSON.
6. write_file    → input: "filepath|||content"
   Use this to save generated code to disk.

## RULES
- FIRST step must always be THINK.
- NEVER write code inside JSON. Always use generate_code tool instead.
- For apps: generate_code → write_file → repeat for each file → OUTPUT.
- You MAY call TOOL multiple times.
- After all files done, produce OUTPUT.
- NEVER produce a step called OBSERVE.

## EXAMPLE — creating an app

User: "create a todo app in folder todo_ai"

{"step": "THINK", "content": "Need 3 files: index.html, style.css, script.js. Will generate each then save."}
{"step": "TOOL", "tool": "generate_code", "input": "Complete index.html for a todo app. Links to style.css and script.js."}
[result returned]
{"step": "TOOL", "tool": "write_file", "input": "todo_ai/index.html|||<paste result here>"}
[result returned]
{"step": "TOOL", "tool": "generate_code", "input": "Complete style.css for a modern clean todo app."}
[result returned]
{"step": "TOOL", "tool": "write_file", "input": "todo_ai/style.css|||<paste result here>"}
[result returned]
{"step": "TOOL", "tool": "generate_code", "input": "Complete script.js for todo app. Add, delete, mark complete."}
[result returned]
{"step": "TOOL", "tool": "write_file", "input": "todo_ai/script.js|||<paste result here>"}
[result returned]
{"step": "OUTPUT", "content": "Created todo_ai/ with index.html, style.css, script.js."}
"""

# ── Pydantic Schema ────────────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    step: str = Field(..., description="THINK, TOOL, or OUTPUT only")
    content: Optional[str] = Field(None, description="Reasoning (THINK) or final answer (OUTPUT)")
    tool: Optional[str] = Field(None, description="Tool name")
    input: Optional[str] = Field(None, description="Tool input")

# ── Agent Loop ─────────────────────────────────────────────────────────────────

def run_agent(user_query: str):
    message_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    max_iterations = 30
    last_generated_code = ""

    for iteration in range(max_iterations):
        print(f"[iter {iteration + 1}] calling model...")

        try:
            response = local_client.chat.completions.parse(
                model="qwen2.5:3b",
                response_format=AgentResponse,
                messages=message_history,
                max_tokens=512  # orchestration only needs small tokens
            )
        except Exception as e:
            if "more system memory" in str(e):
                print("❌ Not enough RAM. Close some apps and try again.")
                return
            raise

        raw_result = response.choices[0].message.content
        parsed: AgentResponse = response.choices[0].message.parsed

        print(f"[iter {iteration + 1}] raw: {raw_result}")

        if parsed is None:
            print("⚠️  Could not parse response.")
            break

        if parsed.step == "OBSERVE":
            message_history.append({
                "role": "user",
                "content": "OBSERVE is system-only. Use THINK, TOOL, or OUTPUT."
            })
            continue

        message_history.append({"role": "assistant", "content": raw_result})

        # ── THINK ─────────────────────────────────────────────────────────────
        if parsed.step == "THINK":
            print(f"💭  {parsed.content}")
            continue

        # ── TOOL ──────────────────────────────────────────────────────────────
        if parsed.step == "TOOL":
            tool_name = parsed.tool
            tool_input = parsed.input

            if tool_name not in available_tools:
                print(f"⚠️  Unknown tool: {tool_name}")
                message_history.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' does not exist. Available: {list(available_tools.keys())}."
                })
                continue

            if not tool_input:
                print("⚠️  No input provided.")
                break

            # key fix: if model calls write_file but forgot to paste code,
            # automatically use last generated code
            if tool_name == "write_file":
                parts = tool_input.split("|||", 1)
                if len(parts) == 1 or parts[1].strip() == "":
                    print("⚠️  write_file has no content — using last generated code.")
                    tool_input = f"{parts[0].strip()}|||{last_generated_code}"

            result = available_tools[tool_name](tool_input)

            if tool_name == "generate_code":
                last_generated_code = result
                print(f"🤖  generate_code → {len(result)} chars")
                # dont send full code back to orchestrator — just confirm
                observe_content = f"Code generated successfully ({len(result)} chars). Now call write_file to save it."
            else:
                print(f"🔧  {tool_name} → {result}")
                observe_content = result

            message_history.append({
                "role": "user",
                "content": (
                    f"Tool result: {observe_content}\n\n"
                    "Continue. If more files needed: generate_code then write_file. "
                    "If all done: OUTPUT."
                )
            })
            continue

        # ── OUTPUT ────────────────────────────────────────────────────────────
        if parsed.step == "OUTPUT":
            print(f"\n✅  {parsed.content}\n")
            return

        print(f"⚠️  Unknown step '{parsed.step}' — skipping.")

    print("⚠️  Agent did not complete within max iterations.")

# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖  General Agent Ready")
    print("    Fully local — no API keys needed.")
    print("    Type 'exit' to quit.\n")

    while True:
        user_query = input("You → ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit", "bye"):
            print("👋  Goodbye!")
            break
        run_agent(user_query)