---
name: agent-development-patterns
description: Best practices for developing AI agents including architecture, triggering conditions, tool integration, and agent orchestration. Use when building autonomous agents, multi-agent systems, or agent-based applications.
trigger_keywords:
  - agent development
  - ai agents
  - autonomous agents
  - agent architecture
  - agent orchestration
  - multi-agent system
  - tool calling
  - agent patterns
---

# Agent Development Patterns

Master patterns for building autonomous AI agents that can reason, take actions, and coordinate with other agents.

## When to Use This Skill

- Building autonomous AI agents with tool access
- Designing multi-agent coordination systems
- Implementing agent memory and state management
- Creating agent orchestration workflows
- Integrating agents with external APIs and services
- Building production-grade agent systems
- Debugging agent behavior and decision-making

## Core Concepts

- **Agent Loop (Observe-Think-Act-Learn)**: The fundamental cycle where agents perceive their environment, reason about it, take actions, and update their knowledge. Production agents implement this with explicit state management and error recovery at each stage.

- **Tool Abstraction**: Agents interact with external systems through well-defined tool interfaces. Tool descriptions must be precise as LLMs use them for selection. Include input schemas, output types, error modes, and usage examples.

- **Memory Architecture**: Separate short-term (conversation buffer), working (task context), and long-term (vector store) memory. Use semantic retrieval for long-term recall and sliding windows for conversation context.

- **ReAct Pattern**: The standard reasoning pattern that alternates between Thought (reasoning), Action (tool call), and Observation (result). This explicit structure improves reliability and enables debugging of agent decisions.

- **Orchestration vs. Autonomy Trade-off**: More autonomous agents are flexible but harder to control. Production systems often use constrained autonomy with explicit guardrails, approval gates, and fallback behaviors.

## Core Agent Architecture

### Basic Agent Loop
```
1. Observe: Receive input and context
2. Think: Reason about the situation
3. Act: Choose and execute actions
4. Learn: Update knowledge from outcomes
```

### Agent Components

**1. Perception**: How agent receives input
- User messages
- System events
- Tool outputs
- Sensor data

**2. Reasoning**: How agent makes decisions
- LLM-based reasoning (ReAct, CoT)
- Rule-based logic
- Planning algorithms
- Memory retrieval

**3. Action**: What agent can do
- Tool invocation
- API calls
- Database operations
- Message generation

**4. Memory**: How agent maintains state
- Short-term (conversation)
- Long-term (facts, patterns)
- Episodic (past experiences)
- Semantic (knowledge base)

## Agent Design Patterns

### Pattern 1: ReAct (Reasoning + Acting)

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def calculate(expression: str) -> float:
    """Perform mathematical calculations."""
    return eval(expression)

# Create ReAct agent
agent = initialize_agent(
    tools=[search_web, calculate],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent will alternate between reasoning and acting:
# Thought: I need to search for information
# Action: search_web
# Action Input: "current weather"
# Observation: [search results]
# Thought: Now I have the information
# Final Answer: [response]
```

### Pattern 2: Tool Selection Agent

```python
class ToolSelectionAgent:
    """Agent that selects appropriate tools based on task."""

    def __init__(self, tools, llm):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm

    def select_tool(self, task: str) -> str:
        """Select best tool for the task."""
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        prompt = f"""Given the task: {task}

Available tools:
{tool_descriptions}

Which tool should be used? Respond with just the tool name."""

        return self.llm.predict(prompt).strip()

    def execute(self, task: str, **kwargs):
        """Execute task with selected tool."""
        tool_name = self.select_tool(task)
        tool = self.tools.get(tool_name)

        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        return tool.run(**kwargs)
```

### Pattern 3: Multi-Agent Coordination

```python
class AgentOrchestrator:
    """Coordinates multiple specialized agents."""

    def __init__(self):
        self.agents = {
            "researcher": ResearchAgent(),
            "analyst": AnalysisAgent(),
            "writer": WritingAgent()
        }

    async def execute_workflow(self, task: str):
        """Execute multi-agent workflow."""

        # Stage 1: Research
        research_results = await self.agents["researcher"].execute(task)

        # Stage 2: Analysis
        analysis = await self.agents["analyst"].execute(
            research_results
        )

        # Stage 3: Writing
        final_output = await self.agents["writer"].execute(
            analysis
        )

        return final_output

    def parallel_execution(self, tasks: list):
        """Execute multiple agents in parallel."""
        import asyncio

        async def run_all():
            return await asyncio.gather(*[
                self.agents[task['agent']].execute(task['input'])
                for task in tasks
            ])

        return asyncio.run(run_all())
```

### Pattern 4: Agent with Memory

```python
class MemoryAgent:
    """Agent with persistent memory."""

    def __init__(self, llm, memory_store):
        self.llm = llm
        self.memory = memory_store
        self.conversation_history = []

    def remember(self, key: str, value: any):
        """Store information in long-term memory."""
        self.memory.store(key, value)

    def recall(self, query: str):
        """Retrieve relevant memories."""
        return self.memory.search(query, k=5)

    def execute(self, user_input: str):
        """Execute with memory integration."""

        # Recall relevant context
        relevant_memories = self.recall(user_input)

        # Build prompt with memories
        prompt = f"""Relevant context:
{relevant_memories}

Conversation history:
{self.conversation_history[-5:]}

User: {user_input}

Response:"""

        # Generate response
        response = self.llm.predict(prompt)

        # Update conversation history
        self.conversation_history.append({
            "user": user_input,
            "assistant": response
        })

        # Store important information
        if self.is_important(response):
            self.remember(user_input, response)

        return response
```

### Pattern 5: Self-Improving Agent

```python
class SelfImprovingAgent:
    """Agent that learns from experience."""

    def __init__(self, llm):
        self.llm = llm
        self.performance_history = []

    def execute_with_feedback(self, task: str, evaluator):
        """Execute task and learn from feedback."""

        # Initial attempt
        result = self.attempt_task(task)

        # Get feedback
        feedback = evaluator(result)

        # If not satisfactory, improve
        attempts = 1
        while feedback['score'] < 0.8 and attempts < 3:
            # Analyze what went wrong
            improvement_prompt = f"""
Task: {task}
Previous attempt: {result}
Feedback: {feedback['comments']}
Score: {feedback['score']}

How can this be improved?
"""

            improvements = self.llm.predict(improvement_prompt)

            # Retry with improvements
            result = self.attempt_task(task, guidance=improvements)
            feedback = evaluator(result)
            attempts += 1

        # Store performance
        self.performance_history.append({
            "task": task,
            "attempts": attempts,
            "final_score": feedback['score']
        })

        return result

    def analyze_performance(self):
        """Analyze performance trends."""
        avg_attempts = sum(h['attempts'] for h in self.performance_history) / len(self.performance_history)
        avg_score = sum(h['final_score'] for h in self.performance_history) / len(self.performance_history)

        return {
            "avg_attempts": avg_attempts,
            "avg_score": avg_score,
            "improving": self.is_improving_over_time()
        }
```

## Tool Integration Best Practices

### Tool Definition
```python
from langchain.tools import tool
from typing import Optional

@tool
def search_database(
    query: str,
    table: str,
    limit: int = 10
) -> str:
    """
    Search database for records matching query.

    Args:
        query: Natural language search query
        table: Database table to search
        limit: Maximum number of results (default: 10)

    Returns:
        JSON string of matching records
    """
    # Implementation
    results = db.search(query, table, limit)
    return json.dumps(results)

# Good tool design:
# - Clear, descriptive name
# - Comprehensive docstring
# - Type hints for all parameters
# - Default values where appropriate
# - Returns structured data
```

### Tool Error Handling
```python
@tool
def safe_api_call(endpoint: str, params: dict) -> str:
    """Make API call with error handling."""
    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.Timeout:
        return {"error": "API request timed out"}
    except requests.HTTPError as e:
        return {"error": f"HTTP error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
```

## Agent Orchestration Patterns

### Sequential Orchestration
```python
def sequential_agent_workflow(task: str):
    """Execute agents in sequence."""

    # Step 1: Planning agent
    plan = planning_agent.create_plan(task)

    # Step 2: Execution agents (run in sequence)
    results = []
    for step in plan.steps:
        agent = select_agent_for_step(step)
        result = agent.execute(step)
        results.append(result)

    # Step 3: Synthesis agent
    final_output = synthesis_agent.combine(results)

    return final_output
```

### Parallel Orchestration
```python
async def parallel_agent_workflow(task: str):
    """Execute independent agents in parallel."""

    # Decompose into parallel subtasks
    subtasks = decompose_task(task)

    # Execute in parallel
    agent_tasks = [
        select_agent(subtask).execute(subtask)
        for subtask in subtasks
    ]

    results = await asyncio.gather(*agent_tasks)

    # Combine results
    return combine_results(results)
```

### Hierarchical Orchestration
```python
class HierarchicalAgentSystem:
    """Multi-level agent hierarchy."""

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.managers = [ManagerAgent(i) for i in range(3)]
        self.workers = [WorkerAgent(i) for i in range(10)]

    def execute(self, task: str):
        """Execute with hierarchical delegation."""

        # Supervisor creates high-level plan
        plan = self.supervisor.plan(task)

        # Managers coordinate worker agents
        manager_results = []
        for subtask in plan.subtasks:
            manager = self.select_manager(subtask)
            workers = self.assign_workers(manager, subtask)
            result = manager.coordinate(workers, subtask)
            manager_results.append(result)

        # Supervisor synthesizes final result
        return self.supervisor.synthesize(manager_results)
```

## Agent Communication Patterns

### Message Passing
```python
class AgentMessage:
    """Structured message between agents."""

    def __init__(self, sender, receiver, content, message_type):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.type = message_type
        self.timestamp = datetime.now()

class MessageBus:
    """Central message bus for agent communication."""

    def __init__(self):
        self.subscribers = {}

    def subscribe(self, agent_id, message_types):
        """Subscribe agent to message types."""
        for msg_type in message_types:
            if msg_type not in self.subscribers:
                self.subscribers[msg_type] = []
            self.subscribers[msg_type].append(agent_id)

    def publish(self, message: AgentMessage):
        """Publish message to subscribers."""
        subscribers = self.subscribers.get(message.type, [])
        for agent_id in subscribers:
            self.deliver_message(agent_id, message)
```

## Testing Agent Systems

### Unit Testing Agents
```python
import pytest
from unittest.mock import Mock, patch

def test_agent_tool_selection():
    """Test agent selects correct tool."""
    mock_llm = Mock()
    mock_llm.predict.return_value = "search_web"

    agent = ToolSelectionAgent(tools, mock_llm)
    tool_name = agent.select_tool("Find information about AI")

    assert tool_name == "search_web"

def test_agent_error_handling():
    """Test agent handles tool errors gracefully."""
    failing_tool = Mock(side_effect=Exception("API error"))

    agent = create_agent(tools=[failing_tool])

    # Should not raise exception
    result = agent.execute("test task")
    assert "error" in result.lower()
```

### Integration Testing
```python
def test_multi_agent_workflow():
    """Test complete multi-agent workflow."""
    orchestrator = AgentOrchestrator()

    result = orchestrator.execute_workflow("Research and summarize AI trends")

    # Verify all agents were called
    assert orchestrator.agents["researcher"].was_called()
    assert orchestrator.agents["analyst"].was_called()
    assert orchestrator.agents["writer"].was_called()

    # Verify output quality
    assert len(result) > 100
    assert "AI" in result
```

## Best Practices

1. **Clear Responsibilities**: Each agent should have a well-defined role
2. **Tool Documentation**: Provide comprehensive tool descriptions
3. **Error Handling**: Implement robust error handling and recovery
4. **Observability**: Log agent decisions and actions
5. **Testing**: Unit test individual agents and integration test workflows
6. **Memory Management**: Prevent memory growth in long conversations
7. **Cost Control**: Monitor and limit LLM API usage
8. **Security**: Validate tool inputs and outputs

## Common Pitfalls

- **Tool Description Quality**: Poor descriptions confuse agent tool selection
- **Infinite Loops**: Agent gets stuck in repetitive reasoning
- **Context Overflow**: Exceeding LLM token limits with large context
- **No Fallbacks**: Agent fails without graceful degradation
- **Over-Engineering**: Starting with complex multi-agent systems
- **Ignoring Costs**: Not monitoring LLM API usage

## Production Considerations

- [ ] Implement request timeout limits
- [ ] Add circuit breakers for external APIs
- [ ] Monitor agent performance metrics
- [ ] Set up error alerting
- [ ] Implement rate limiting
- [ ] Add comprehensive logging
- [ ] Version control agent configurations
- [ ] Test with adversarial inputs
- [ ] Document agent decision logic
- [ ] Establish rollback procedures
