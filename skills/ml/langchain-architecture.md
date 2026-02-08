---
name: langchain-architecture
description: Design LLM applications using the LangChain framework with agents, memory, and tool integration patterns. Use when building LangChain applications, implementing AI agents, or creating complex LLM workflows.
trigger_keywords:
  - langchain
  - llm agents
  - langchain agents
  - chains
  - memory
  - tools integration
  - react agent
  - agent executor
  - callbacks
---

# LangChain Architecture

Master the LangChain framework for building sophisticated LLM applications with agents, chains, memory, and tool integration.

## When to Use This Skill

- Building autonomous AI agents with tool access
- Implementing complex multi-step LLM workflows
- Managing conversation memory and state
- Integrating LLMs with external data sources and APIs
- Creating modular, reusable LLM application components
- Implementing document processing pipelines
- Building production-grade LLM applications

## Core Concepts

- **Chains as Composable Units**: Chains encapsulate LLM calls with pre/post-processing. Design chains as pure functions with clear inputs/outputs to enable composition, testing, and reuse across applications.

- **Memory Window Management**: LLMs have fixed context windows. Use ConversationSummaryMemory for long sessions, BufferWindowMemory for recent context, or VectorStoreMemory for semantic retrieval of relevant history.

- **LCEL (LangChain Expression Language)**: The declarative syntax for building chains using `|` pipe operators. Enables streaming, parallel execution, and fallbacks with cleaner code than imperative chain construction.

- **Retrieval-Augmented Generation**: Combine document retrieval with generation. The retriever fetches relevant chunks, the chain stuffs them into context, and the LLM generates grounded responses with source citations.

- **Callback-Based Observability**: Implement BaseCallbackHandler for logging, metrics, and debugging. Track token usage, latency, and errors across all chain/agent operations for production monitoring.

## LangChain Components

### 1. Agents
Autonomous systems that use LLMs to decide which actions to take.

**Agent Types:**
- **ReAct**: Reasoning + Acting in interleaved manner
- **OpenAI Functions**: Leverages function calling API
- **Structured Chat**: Handles multi-input tools
- **Conversational**: Optimized for chat interfaces
- **Self-Ask with Search**: Decomposes complex queries

### 2. Chains
Sequences of calls to LLMs or other utilities.

**Chain Types:**
- **LLMChain**: Basic prompt + LLM combination
- **SequentialChain**: Multiple chains in sequence
- **RouterChain**: Routes inputs to specialized chains
- **TransformChain**: Data transformations between steps
- **MapReduceChain**: Parallel processing with aggregation

### 3. Memory
Systems for maintaining context across interactions.

**Memory Types:**
- **ConversationBufferMemory**: Stores all messages
- **ConversationSummaryMemory**: Summarizes older messages
- **ConversationBufferWindowMemory**: Keeps last N messages
- **EntityMemory**: Tracks information about entities
- **VectorStoreMemory**: Semantic similarity retrieval

### 4. Document Processing
Loading, transforming, and storing documents for retrieval.

**Components:**
- **Document Loaders**: Load from various sources
- **Text Splitters**: Chunk documents intelligently
- **Vector Stores**: Store and retrieve embeddings
- **Retrievers**: Fetch relevant documents
- **Indexes**: Organize documents for efficient access

### 5. Callbacks
Hooks for logging, monitoring, and debugging.

**Use Cases:**
- Request/response logging
- Token usage tracking
- Latency monitoring
- Error handling
- Custom metrics collection

## Quick Start

```python
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Initialize LLM
llm = OpenAI(temperature=0)

# Load tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Add memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Run agent
result = agent.run("What's the weather in SF? Then calculate 25 * 4")
```

## Architecture Patterns

### Pattern 1: RAG with LangChain
```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load and process documents
loader = TextLoader('documents.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Query
result = qa_chain({"query": "What is the main topic?"})
```

### Pattern 2: Custom Agent with Tools
```python
from langchain.agents import Tool, AgentExecutor
from langchain.agents.react.base import ReActDocstoreAgent
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """Search internal database for information."""
    # Your database search logic
    return f"Results for: {query}"

@tool
def send_email(recipient: str, content: str) -> str:
    """Send an email to specified recipient."""
    # Email sending logic
    return f"Email sent to {recipient}"

tools = [search_database, send_email]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

### Pattern 3: Multi-Step Chain
```python
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# Step 1: Extract key information
extract_prompt = PromptTemplate(
    input_variables=["text"],
    template="Extract key entities from: {text}\n\nEntities:"
)
extract_chain = LLMChain(llm=llm, prompt=extract_prompt, output_key="entities")

# Step 2: Analyze entities
analyze_prompt = PromptTemplate(
    input_variables=["entities"],
    template="Analyze these entities: {entities}\n\nAnalysis:"
)
analyze_chain = LLMChain(llm=llm, prompt=analyze_prompt, output_key="analysis")

# Step 3: Generate summary
summary_prompt = PromptTemplate(
    input_variables=["entities", "analysis"],
    template="Summarize:\nEntities: {entities}\nAnalysis: {analysis}\n\nSummary:"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# Combine into sequential chain
overall_chain = SequentialChain(
    chains=[extract_chain, analyze_chain, summary_chain],
    input_variables=["text"],
    output_variables=["entities", "analysis", "summary"],
    verbose=True
)
```

## Memory Management Best Practices

### Choosing the Right Memory Type
```python
# For short conversations (< 10 messages)
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# For long conversations (summarize old messages)
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# For sliding window (last N messages)
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=5)

# For entity tracking
from langchain.memory import ConversationEntityMemory
memory = ConversationEntityMemory(llm=llm)

# For semantic retrieval of relevant history
from langchain.memory import VectorStoreRetrieverMemory
memory = VectorStoreRetrieverMemory(retriever=retriever)
```

## Callback System

### Custom Callback Handler
```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM ended with response: {response}")

    def on_llm_error(self, error, **kwargs):
        print(f"LLM error: {error}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started with inputs: {inputs}")

    def on_agent_action(self, action, **kwargs):
        print(f"Agent taking action: {action}")

# Use callback
agent.run("query", callbacks=[CustomCallbackHandler()])
```

## Testing Strategies

```python
import pytest
from unittest.mock import Mock

def test_agent_tool_selection():
    # Mock LLM to return specific tool selection
    mock_llm = Mock()
    mock_llm.predict.return_value = "Action: search_database\nAction Input: test query"

    agent = initialize_agent(tools, mock_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    result = agent.run("test query")

    # Verify correct tool was selected
    assert "search_database" in str(mock_llm.predict.call_args)

def test_memory_persistence():
    memory = ConversationBufferMemory()

    memory.save_context({"input": "Hi"}, {"output": "Hello!"})

    assert "Hi" in memory.load_memory_variables({})['history']
    assert "Hello!" in memory.load_memory_variables({})['history']
```

## Performance Optimization

### 1. Caching
```python
from langchain.cache import InMemoryCache
import langchain

langchain.llm_cache = InMemoryCache()
```

### 2. Batch Processing
```python
# Process multiple documents in parallel
from langchain.document_loaders import DirectoryLoader
from concurrent.futures import ThreadPoolExecutor

loader = DirectoryLoader('./docs')
docs = loader.load()

def process_doc(doc):
    return text_splitter.split_documents([doc])

with ThreadPoolExecutor(max_workers=4) as executor:
    split_docs = list(executor.map(process_doc, docs))
```

### 3. Streaming Responses
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
```

## Common Pitfalls

1. **Memory Overflow**: Not managing conversation history length
2. **Tool Selection Errors**: Poor tool descriptions confuse agents
3. **Context Window Exceeded**: Exceeding LLM token limits
4. **No Error Handling**: Not catching and handling agent failures
5. **Inefficient Retrieval**: Not optimizing vector store queries

## Production Checklist

- [ ] Implement proper error handling
- [ ] Add request/response logging
- [ ] Monitor token usage and costs
- [ ] Set timeout limits for agent execution
- [ ] Implement rate limiting
- [ ] Add input validation
- [ ] Test with edge cases
- [ ] Set up observability (callbacks)
- [ ] Implement fallback strategies
- [ ] Version control prompts and configurations

## Advanced Patterns

### Router Chain
```python
from langchain.chains.router import MultiPromptChain

# Define specialized chains
physics_template = """You are a physics expert. Answer: {input}"""
math_template = """You are a math expert. Answer: {input}"""

prompt_infos = [
    {"name": "physics", "description": "Good for physics questions", "prompt_template": physics_template},
    {"name": "math", "description": "Good for math questions", "prompt_template": math_template},
]

# Create router chain
chain = MultiPromptChain.from_prompts(llm, prompt_infos, verbose=True)

# Routes to appropriate specialized chain
result = chain.run("What is Newton's second law?")
```

### MapReduce Pattern
```python
from langchain.chains import MapReduceDocumentsChain

# Map: Process each document
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce: Combine results
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# MapReduce chain
combine_documents_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name="docs",
    return_intermediate_steps=True
)
```
