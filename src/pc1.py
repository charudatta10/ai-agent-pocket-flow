import os, re, sys, yaml, requests
from ddgs import DDGS
from ollama import chat
from pocketflow import Node, Flow


def call_llm(prompt: str) -> str:
    try:
        response = chat(
            model="llama3.1:8b", messages=[{"role": "user", "content": prompt}]
        )
        return response.get("message", {}).get("content", "")
    except Exception as e:
        print(f"⚠️ LLM call failed: {e}")
        return ""


def extract_decision(response: str) -> dict:
    match = re.search(r"```yaml\s*(.*?)\s*```", response, re.DOTALL)
    yaml_text = match.group(1).strip() if match else response.strip()
    try:
        return yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        return {"raw": yaml_text, "_error": str(exc)}


def search_web_duckduckgo(query):
    results = DDGS().text(query, max_results=5)
    return "\n\n".join(
        [
            f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
            for r in results
        ]
    )


class DecideAction(Node):
    PROMPT = """
### CONTEXT
You are a research assistant that can search the web.
Question: {question}
Previous Research: {context}

### ACTION SPACE
[1] search
  Description: Look up more information on the web
  Parameters:
    - query (str): What to search for

[2] answer
  Description: Answer the question with current knowledge
  Parameters:
    - answer (str): Final answer to the question

## NEXT ACTION
Decide the next action based on the context and available actions.
Return your response in this format:

```yaml
thinking: |
    <your step-by-step reasoning process>
action: search OR answer
reason: <why you chose this action>
answer: <if action is answer>
search_query: <specific search query if action is search>
```
IMPORTANT: Make sure to:
1. Use proper indentation (4 spaces) for all multi-line fields
2. Use the | character for multi-line text fields
3. Keep single-line fields without the | character
"""

    def prep(self, shared):
        return shared["question"], shared.get("context", "No previous search")

    def exec(self, prep_res):
        question, context = prep_res
        print("🤔 Agent deciding what to do next...")
        prompt = self.PROMPT.format(question=question, context=context)
        return extract_decision(call_llm(prompt))

    def post(self, shared, prep_res, exec_res):
        if exec_res.get("action") == "search":
            shared["search_query"] = exec_res.get("search_query", "")
            print(f"🔍 Agent decided to search for: {shared['search_query']}")
        else:
            shared["context"] = exec_res.get("answer", "")
            print("💡 Agent decided to answer the question")
        return exec_res.get("action", "")


class SearchWeb(Node):
    def prep(self, shared):
        return shared["search_query"]

    def exec(self, prep_res):
        print(f"🌐 Searching the web for: {prep_res}")
        results = search_web_duckduckgo(prep_res)
        print(f"🌐 Found: {results}")
        return results

    def post(self, shared, prep_res, exec_res):
        shared["context"] = (
            shared.get("context", "")
            + f"\n\nSEARCH: {shared['search_query']}\nRESULTS: {exec_res}"
        )
        print("📚 Found information, analyzing results...")
        return "decide"


class AnswerQuestion(Node):
    PROMPT = """
### CONTEXT
Based on the following information, answer the question.
Question: {question}
Research: {context}

## YOUR ANSWER:
Provide a comprehensive answer using the research results.
"""

    def prep(self, shared):
        return shared["question"], shared.get("context", "")

    def exec(self, prep_res):
        question, context = prep_res
        print("✍️ Crafting final answer...")
        return call_llm(self.PROMPT.format(question=question, context=context))

    def post(self, shared, prep_res, exec_res):
        shared["answer"] = exec_res
        print("✅ Answer generated successfully")
        return "done"


def create_agent_flow():
    decide, search, answer = DecideAction(), SearchWeb(), AnswerQuestion()
    flow = Flow(start=decide)
    _conn1 = (decide - "search") >> search
    _conn2 = (decide - "answer") >> answer
    _conn3 = (search - "decide") >> decide
    return flow


if __name__ == "__main__":
    question = "Who won the Nobel Prize in Physics 2024?"
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            question = arg[2:]
            break
    agent_flow = create_agent_flow()
    shared = {"question": question}
    print(f"🤔 Processing question: {question}")
    agent_flow.run(shared)
    print("\n🎯 Final Answer:")
    print(shared.get("answer", "No answer found"))
