---
name: prompt-engineering-patterns
description: Master advanced prompt engineering techniques to maximize LLM performance, reliability, and controllability in production. Use when optimizing prompts, improving LLM outputs, or designing production prompt templates.
trigger_keywords:
  - prompt engineering
  - few shot learning
  - chain of thought
  - zero shot
  - prompt optimization
  - prompt templates
  - system prompts
  - cot prompting
---

# Prompt Engineering Patterns

Master advanced prompt engineering techniques to maximize LLM performance, reliability, and controllability.

## When to Use This Skill

- Designing complex prompts for production LLM applications
- Optimizing prompt performance and consistency
- Implementing structured reasoning patterns (chain-of-thought, tree-of-thought)
- Building few-shot learning systems with dynamic example selection
- Creating reusable prompt templates with variable interpolation
- Debugging and refining prompts that produce inconsistent outputs
- Implementing system prompts for specialized AI assistants

## Core Concepts

- **Instruction Following Hierarchy**: LLMs process prompts in order. Place system instructions first, then task context, examples, input data, and output format. Later content can override earlier content, so structure accordingly.

- **Few-Shot Learning**: 2-5 examples dramatically improve task performance. Select examples by semantic similarity to the input, ensure diversity of patterns, and place the most relevant example closest to the input.

- **Chain-of-Thought (CoT)**: Explicit reasoning steps improve accuracy on complex tasks. Use "Let's think step by step" for zero-shot CoT, or provide reasoning traces in few-shot examples for even better results.

- **Prompt Robustness**: Production prompts must handle edge cases. Test with adversarial inputs, ambiguous queries, and out-of-distribution examples. Add explicit fallback instructions for uncertainty.

- **Structured Output Control**: Specify exact output formats (JSON schemas, markdown templates) in the prompt. Include format examples and validation instructions to ensure parseable, consistent responses.

## Core Capabilities

### 1. Few-Shot Learning
- Example selection strategies (semantic similarity, diversity sampling)
- Balancing example count with context window constraints
- Constructing effective demonstrations with input-output pairs
- Dynamic example retrieval from knowledge bases
- Handling edge cases through strategic example selection

### 2. Chain-of-Thought Prompting
- Step-by-step reasoning elicitation
- Zero-shot CoT with "Let's think step by step"
- Few-shot CoT with reasoning traces
- Self-consistency techniques (sampling multiple reasoning paths)
- Verification and validation steps

### 3. Prompt Optimization
- Iterative refinement workflows
- A/B testing prompt variations
- Measuring prompt performance metrics (accuracy, consistency, latency)
- Reducing token usage while maintaining quality
- Handling edge cases and failure modes

### 4. Template Systems
- Variable interpolation and formatting
- Conditional prompt sections
- Multi-turn conversation templates
- Role-based prompt composition
- Modular prompt components

### 5. System Prompt Design
- Setting model behavior and constraints
- Defining output formats and structure
- Establishing role and expertise
- Safety guidelines and content policies
- Context setting and background information

## Quick Start

```python
from prompt_optimizer import PromptTemplate, FewShotSelector

# Define a structured prompt template
template = PromptTemplate(
    system="You are an expert SQL developer. Generate efficient, secure SQL queries.",
    instruction="Convert the following natural language query to SQL:\n{query}",
    few_shot_examples=True,
    output_format="SQL code block with explanatory comments"
)

# Configure few-shot learning
selector = FewShotSelector(
    examples_db="sql_examples.jsonl",
    selection_strategy="semantic_similarity",
    max_examples=3
)

# Generate optimized prompt
prompt = template.render(
    query="Find all users who registered in the last 30 days",
    examples=selector.select(query="user registration date filter")
)
```

## Key Patterns

### Progressive Disclosure
Start with simple prompts, add complexity only when needed:

1. **Level 1**: Direct instruction
   - "Summarize this article"

2. **Level 2**: Add constraints
   - "Summarize this article in 3 bullet points, focusing on key findings"

3. **Level 3**: Add reasoning
   - "Read this article, identify the main findings, then summarize in 3 bullet points"

4. **Level 4**: Add examples
   - Include 2-3 example summaries with input-output pairs

### Instruction Hierarchy
```
[System Context] → [Task Instruction] → [Examples] → [Input Data] → [Output Format]
```

### Error Recovery
Build prompts that gracefully handle failures:
- Include fallback instructions
- Request confidence scores
- Ask for alternative interpretations when uncertain
- Specify how to indicate missing information

## Best Practices

1. **Be Specific**: Vague prompts produce inconsistent results
2. **Show, Don't Tell**: Examples are more effective than descriptions
3. **Test Extensively**: Evaluate on diverse, representative inputs
4. **Iterate Rapidly**: Small changes can have large impacts
5. **Monitor Performance**: Track metrics in production
6. **Version Control**: Treat prompts as code with proper versioning
7. **Document Intent**: Explain why prompts are structured as they are

## Common Pitfalls

- **Over-engineering**: Starting with complex prompts before trying simple ones
- **Example pollution**: Using examples that don't match the target task
- **Context overflow**: Exceeding token limits with excessive examples
- **Ambiguous instructions**: Leaving room for multiple interpretations
- **Ignoring edge cases**: Not testing on unusual or boundary inputs

## Integration Patterns

### With RAG Systems
```python
# Combine retrieved context with prompt engineering
prompt = f"""Given the following context:
{retrieved_context}

{few_shot_examples}

Question: {user_question}

Provide a detailed answer based solely on the context above. If the context doesn't contain enough information, explicitly state what's missing."""
```

### With Validation
```python
# Add self-verification step
prompt = f"""{main_task_prompt}

After generating your response, verify it meets these criteria:
1. Answers the question directly
2. Uses only information from provided context
3. Cites specific sources
4. Acknowledges any uncertainty

If verification fails, revise your response."""
```

## Performance Optimization

### Token Efficiency
- Remove redundant words and phrases
- Use abbreviations consistently after first definition
- Consolidate similar instructions
- Move stable content to system prompts

### Latency Reduction
- Minimize prompt length without sacrificing quality
- Use streaming for long-form outputs
- Cache common prompt prefixes
- Batch similar requests when possible

## Success Metrics

Track these KPIs for your prompts:
- **Accuracy**: Correctness of outputs
- **Consistency**: Reproducibility across similar inputs
- **Latency**: Response time (P50, P95, P99)
- **Token Usage**: Average tokens per request
- **Success Rate**: Percentage of valid outputs
- **User Satisfaction**: Ratings and feedback

## Advanced Techniques

### Chain-of-Thought Examples

**Zero-Shot CoT:**
```
Question: What is 15% of 240?

Let's think step by step:
1. Convert 15% to decimal: 0.15
2. Multiply 240 × 0.15
3. 240 × 0.15 = 36

Answer: 36
```

**Few-Shot CoT:**
```
Q: A restaurant bill is $85. If you want to leave a 18% tip, how much is the tip?
A: Let's break this down:
- Bill amount: $85
- Tip percentage: 18% = 0.18
- Tip = $85 × 0.18 = $15.30

Q: If a shirt costs $45 and is on sale for 25% off, what's the sale price?
A: Let's solve step by step:
```

### Self-Consistency
```python
# Generate multiple reasoning paths
responses = []
for i in range(5):
    response = llm.generate(prompt, temperature=0.7)
    responses.append(response)

# Take majority vote on final answer
final_answer = most_common(extract_answers(responses))
```

### Prompt Templating
```python
TASK_TEMPLATE = """
Role: {role}
Task: {task}
Context: {context}

Input: {input}

Requirements:
{requirements}

Output Format: {output_format}
"""

# Use with variables
prompt = TASK_TEMPLATE.format(
    role="Data Analyst",
    task="Analyze sales trends",
    context="Q4 2024 sales data",
    input=sales_data,
    requirements="- Identify top 3 trends\n- Provide actionable insights",
    output_format="Markdown report with visualizations"
)
```

## Next Steps

1. Review the prompt template library for common patterns
2. Experiment with few-shot learning for your specific use case
3. Implement prompt versioning and A/B testing
4. Set up automated evaluation pipelines
5. Document your prompt engineering decisions and learnings
