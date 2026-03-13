from openai import OpenAI
from pydantic import BaseModel, Field
import requests
import json
import datetime
import subprocess
from typing import Optional, List

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# ── Tools ──────────────────────────────────────────────────────────────────────

def run_command(cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15
        )
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode != 0:
            return f"Command failed (code {result.returncode}): {error or 'no error message'}"
        return output if output else f"Command executed successfully: {cmd}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 15 seconds."
    except Exception as e:
        return f"Error running command: {str(e)}"

def write_file(path_and_content: str) -> str:
    """
    Expects input format:  filepath|||content
    Example: todo_ai/index.html|||<!DOCTYPE html>...
    """
    try:
        if "|||" not in path_and_content:
            return "Invalid format. Use: filepath|||content"
        filepath, content = path_and_content.split("|||", 1)
        filepath = filepath.strip()

        # create parent dirs if needed
        os_path = filepath.replace("/", "\\")
        dir_part = "\\".join(os_path.split("\\")[:-1])
        if dir_part:
            subprocess.run(f'mkdir "{dir_part}"', shell=True, capture_output=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File created successfully: {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"The weather in {city} is {response.text.strip()}"
        return f"Could not fetch weather for {city}"
    except requests.RequestException as e:
        return f"Network error for {city}: {str(e)}"

def get_time(timezone: str = "local") -> str:
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Invalid expression: only basic math allowed."
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
}

# ── System Prompt ──────────────────────────────────────────────────────────────

system_prompt = """
You are a helpful general-purpose AI assistant running on Windows PowerShell.
Respond ONLY in valid JSON. No text outside JSON ever.

You have exactly THREE valid steps: THINK, TOOL, OUTPUT.

## AVAILABLE TOOLS

1. get_weather(city)
   {"step": "TOOL", "tool": "get_weather", "input": "Mumbai"}

2. get_time(timezone)
   {"step": "TOOL", "tool": "get_time", "input": "local"}

3. calculator(expression)
   {"step": "TOOL", "tool": "calculator", "input": "12 * (3 + 7)"}

4. run_command(cmd) — runs a Windows PowerShell command, returns output
   {"step": "TOOL", "tool": "run_command", "input": "mkdir todo_ai"}

5. write_file(filepath|||content) — creates a file with content
   {"step": "TOOL", "tool": "write_file", "input": "todo_ai/index.html|||<!DOCTYPE html>..."}

## STRICT RULES
- FIRST response must ALWAYS be THINK.
- After THINK, call TOOL if needed, otherwise go to OUTPUT.
- You MAY call TOOL multiple times if the task requires it (e.g. creating multiple files).
- After ALL tools are done, produce OUTPUT.
- NEVER output a step called OBSERVE. That is system-only.
- NEVER invent tool results. Use only what the system returns.
- Only THREE valid steps: THINK, TOOL, OUTPUT. Nothing else.
- For file creation tasks: use write_file once per file, then OUTPUT when all files are done.

## EXAMPLES

Example — create files:
User: "create a folder hello and put index.html in it"
{"step": "THINK", "content": "I need to create a folder and then write index.html into it. I will use write_file which auto-creates the folder."}
{"step": "TOOL", "tool": "write_file", "input": "hello/index.html|||<!DOCTYPE html><html><body>Hello</body></html>"}
[system injects result]
{"step": "OUTPUT", "content": "Created hello/index.html successfully."}

Example — run command:
User: "list files in current directory"
{"step": "THINK", "content": "I will run the dir command to list files."}
{"step": "TOOL", "tool": "run_command", "input": "dir"}
[system injects result]
{"step": "OUTPUT", "content": "Here are the files: ..."}
"""

# ── Pydantic Schema ────────────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    step: str = Field(..., description="THINK, TOOL, or OUTPUT only")
    content: Optional[str] = Field(None, description="Reasoning (THINK) or final answer (OUTPUT)")
    tool: Optional[str] = Field(None, description="Tool name (TOOL step only)")
    input: Optional[str] = Field(None, description="Tool input string (TOOL step only)")

# ── Agent Loop ─────────────────────────────────────────────────────────────────

def run_agent(user_query: str):
    message_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    max_iterations = 20  # higher limit for multi-file tasks
    tool_results = []    # track all tool calls this session

    for iteration in range(max_iterations):
        print(f"[iter {iteration + 1}] calling model...")

        response = client.chat.completions.parse(
            model="qwen2.5:7b-instruct-q4_K_M",
            response_format=AgentResponse,
            messages=message_history,
            max_tokens=4096
        )

        raw_result = response.choices[0].message.content
        parsed: AgentResponse = response.choices[0].message.parsed

        print(f"[iter {iteration + 1}] raw: {raw_result}")

        if parsed is None:
            print("⚠️  Could not parse model response.")
            break

        if parsed.step == "OBSERVE":
            print("⚠️  Model tried to output OBSERVE — re-prompting.")
            message_history.append({
                "role": "user",
                "content": "OBSERVE is system-only. Your valid steps are THINK, TOOL, or OUTPUT only."
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
                    "content": f"Tool '{tool_name}' does not exist. Available: {list(available_tools.keys())}. Try again."
                })
                continue

            if not tool_input:
                print("⚠️  No input provided for tool.")
                break

            result = available_tools[tool_name](tool_input)
            print(f"🔧  {tool_name}({tool_input[:60]}...) → {result}")
            tool_results.append({"tool": tool_name, "result": result})

            message_history.append({
                "role": "user",
                "content": (
                    f"Tool result:\n{result}\n\n"
                    "If you have more files to create, call TOOL again. "
                    "If all done, produce OUTPUT."
                )
            })
            continue

        # ── OUTPUT ────────────────────────────────────────────────────────────
        if parsed.step == "OUTPUT":
            print(f"\n✅  {parsed.content}\n")
            return

        print(f"⚠️  Unknown step '{parsed.step}' — skipping.")

    print("⚠️  Agent did not produce OUTPUT within max iterations.")

# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🤖  General Agent Ready")
    print("    Ask me anything — weather, math, time, files, or general questions.")
    print("    Type 'exit' to quit.\n")

    while True:
        user_query = input("You → ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit", "bye"):
            print("👋  Goodbye!")
            break
        run_agent(user_query)