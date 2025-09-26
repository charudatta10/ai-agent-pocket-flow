import os
import re
import sys
import yaml
import requests
from ddgs import DDGS
from ollama import chat
from pocketflow import Node, Flow

# --------------------------------------------------------------------------- 
def call_llm(prompt: str) -> str:
    """
    Send `prompt` to the Ollama LLM and return the plain text reply.
    """
    try:
        response = chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}])
        return response.get("message", {}).get("content", "")
    except Exception as e:
        print(f"⚠️ LLM call failed: {e}")
        return ""




# --------------------------------------------------------------------------- #
def extract_decision(response: str) -> dict:
    """
    Pull out the first ```yaml … ``` block and parse it with PyYAML.
    If the block is missing, we try to parse the entire response.
    If parsing still fails, we return the raw string in a dict.
    """
    # 1️⃣  Search for a fenced YAML block
    match = re.search(r"```yaml\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        yaml_text = match.group(1).strip()
    else:
        # No fenced block – use the whole response
        yaml_text = response.strip()

    # 2️⃣  Try to parse
    try:
        return yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        # YAML was malformed – return raw text for debugging
        return {"raw": yaml_text, "_error": str(exc)}


# --------------------------------------------------------------------------- #


def search_web_duckduckgo(query):
    results = DDGS().text(query, max_results=5)
    # Convert results to a string
    results_str = "\n\n".join(
        [
            f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
            for r in results
        ]
    )
    return results_str


class DecideAction(Node):
    def prep(self, shared):
        """Prepare the context and question for the decision-making process."""
        # Get the current context (default to "No previous search" if none exists)
        context = shared.get("context", "No previous search")
        # Get the question from the shared store
        question = shared["question"]
        # Return both for the exec step
        return question, context

    def exec(self, prep_res):
        """Call the LLM to decide whether to search or answer."""
        question, context = prep_res

        print(f"🤔 Agent deciding what to do next...")

        # Create a prompt to help the LLM decide what to do next with proper yaml formatting
        prompt = f"""
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

        # Call the LLM to make a decision
        response = call_llm(prompt)

        # Parse the response to get the decision
        decision = extract_decision(response)

        return decision

    def post(self, shared, prep_res, exec_res):
        """Save the decision and determine the next step in the flow."""
        # If LLM decided to search, save the search query
        if exec_res["action"] == "search":
            shared["search_query"] = exec_res["search_query"]
            print(f"🔍 Agent decided to search for: {exec_res['search_query']}")
        else:
            shared["context"] = exec_res[
                "answer"
            ]  # save the context if LLM gives the answer without searching.
            print(f"💡 Agent decided to answer the question")

        # Return the action to determine the next node in the flow
        return exec_res["action"]


class SearchWeb(Node):
    def prep(self, shared):
        """Get the search query from the shared store."""
        return shared["search_query"]

    def exec(self, search_query):
        """Search the web for the given query."""
        # Call the search utility function
        print(f"🌐 Searching the web for: {search_query}")
        results = search_web_duckduckgo(search_query)
        print(f"🌐 Found: {results}")
        return results

    def post(self, shared, prep_res, exec_res):
        """Save the search results and go back to the decision node."""
        # Add the search results to the context in the shared store
        previous = shared.get("context", "")
        shared["context"] = (
            previous
            + "\n\nSEARCH: "
            + shared["search_query"]
            + "\nRESULTS: "
            + exec_res
        )

        print(f"📚 Found information, analyzing results...")

        # Always go back to the decision node after searching
        return "decide"


class AnswerQuestion(Node):
    def prep(self, shared):
        """Get the question and context for answering."""
        return shared["question"], shared.get("context", "")

    def exec(self, inputs):
        """Call the LLM to generate a final answer."""
        question, context = inputs

        print(f"✍️ Crafting final answer...")

        # Create a prompt for the LLM to answer the question
        prompt = f"""
### CONTEXT
Based on the following information, answer the question.
Question: {question}
Research: {context}

## YOUR ANSWER:
Provide a comprehensive answer using the research results.
"""
        # Call the LLM to generate an answer
        answer = call_llm(prompt)
        return answer

    def post(self, shared, prep_res, exec_res):
        """Save the final answer and complete the flow."""
        # Save the answer in the shared store
        shared["answer"] = exec_res

        print(f"✅ Answer generated successfully")

        # We're done - no need to continue the flow
        return "done"


def create_agent_flow():
    """
    Create and connect the nodes to form a complete agent flow.

    The flow works like this:
    1. DecideAction node decides whether to search or answer
    2. If search, go to SearchWeb node
    3. If answer, go to AnswerQuestion node
    4. After SearchWeb completes, go back to DecideAction

    Returns:
        Flow: A complete research agent flow
    """
    # Create instances of each node
    decide = DecideAction()
    search = SearchWeb()
    answer = AnswerQuestion()

    # Connect the nodes
    # If DecideAction returns "search", go to SearchWeb
    decide - "search" >> search

    # If DecideAction returns "answer", go to AnswerQuestion
    decide - "answer" >> answer

    # After SearchWeb completes and returns "decide", go back to DecideAction
    search - "decide" >> decide

    # Create and return the flow, starting with the DecideAction node
    return Flow(start=decide)


def main():
    """Simple function to process a question."""
    # Default question
    default_question = "Who won the Nobel Prize in Physics 2024?"

    # Get question from command line if provided with --
    question = default_question
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            question = arg[2:]
            break

    # Create the agent flow
    agent_flow = create_agent_flow()

    # Process the question
    shared = {"question": question}
    print(f"🤔 Processing question: {question}")
    agent_flow.run(shared)
    print("\n🎯 Final Answer:")
    print(shared.get("answer", "No answer found"))


if __name__ == "__main__":
    print("## Testing call_llm")
    prompt = "In a few words, what is the meaning of life?"
    print(f"## Prompt: {prompt}")
    response = call_llm(prompt)
    print(f"## Response: {response}")

    print("## Testing search_web")
    query = "Who won the Nobel Prize in Physics 2024?"
    print(f"## Query: {query}")
    results = search_web_duckduckgo(query)
    print(f"## Results: {results}")
    main()
