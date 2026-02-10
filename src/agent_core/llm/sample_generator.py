from .client import LLMClient

client = LLMClient()

SYSTEM = """
You generate minimal valid sample data files.

Rules:
- Match the task description.
- Keep the file SMALL.
- Ensure it can be parsed correctly.
- CSV preferred if filename ends with .csv.
"""

def generate_sample(task: str, path: str) -> str:
    prompt = f"""
Task:
{task}

Generate minimal valid sample content for file:
{path}
"""
    return client.chat(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
