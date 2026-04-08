# test_self_correction_loop.py
# Test the LangGraph self-correction loop with ambiguous queries

import asyncio
from agent.graph_state import LangGraphState
from agent.nodes import retrieve_node, generate_node, critique_node, decision_node, log_node

async def run_self_correction_loop(query: str, max_iterations: int = 2):
    state = LangGraphState(query=query)
    for i in range(max_iterations + 1):
        state.iteration_count = i
        state = await retrieve_node(state)
        state = generate_node(state)
        state = critique_node(state)
        action = decision_node(state, max_iterations=max_iterations)
        print(f"Iteration {i}:\n  Faithfulness: {state.faithfulness_score}\n  Relevance: {state.relevance_score}\n  Action: {action}\n  Answer: {state.answer}\n")
        if action == "return":
            break
    log_node(state)
    return state

if __name__ == "__main__":
    ambiguous_queries = [
        "What is the relationship between the main entities in the documents?",
        "Who is responsible for the compliance issues mentioned?",
        "How does policy A affect regulation B?"
    ]
    for query in ambiguous_queries:
        print(f"\nTesting query: {query}")
        asyncio.run(run_self_correction_loop(query))
