# ML/AI Skills

Production-quality skills for building ML/AI systems, covering RAG implementation, prompt engineering, LLM evaluation, ML pipelines, agent development, and LangChain architecture.

## Available Skills

### 1. RAG Implementation (`rag-implementation.md`)
**When to use:** Building retrieval-augmented generation systems

**Key Topics:**
- Vector databases (Pinecone, Weaviate, Chroma, Qdrant, FAISS)
- Embeddings (OpenAI, Sentence Transformers, E5, Instructor, BGE)
- Retrieval strategies (dense, sparse, hybrid, multi-query, HyDE)
- Reranking methods (cross-encoders, Cohere, MMR, LLM-based)
- Document chunking strategies
- Metadata filtering and optimization
- Prompt engineering for RAG
- Evaluation metrics

**Trigger Keywords:** rag, retrieval augmented generation, vector database, semantic search, document qa, embeddings, hybrid search, reranking

---

### 2. Prompt Engineering Patterns (`prompt-engineering-patterns.md`)
**When to use:** Optimizing LLM prompts and outputs

**Key Topics:**
- Few-shot learning and example selection
- Chain-of-thought (CoT) prompting
- Zero-shot and few-shot techniques
- Prompt optimization and A/B testing
- Template systems and variable interpolation
- System prompt design
- Progressive disclosure patterns
- Error recovery strategies
- Token efficiency and latency reduction

**Trigger Keywords:** prompt engineering, few shot learning, chain of thought, zero shot, prompt optimization, prompt templates, system prompts, cot prompting

---

### 3. LLM Evaluation (`llm-evaluation.md`)
**When to use:** Testing and measuring LLM application performance

**Key Topics:**
- Automated metrics (BLEU, ROUGE, BERTScore, perplexity)
- Classification metrics (accuracy, precision, recall, F1, AUC-ROC)
- Retrieval metrics (MRR, NDCG, Precision@K, Recall@K)
- Human evaluation frameworks
- LLM-as-judge patterns (pointwise, pairwise, reference-based)
- A/B testing and statistical analysis
- Regression testing and detection
- Inter-rater agreement
- Custom metrics (groundedness, toxicity, factuality)

**Trigger Keywords:** llm evaluation, model evaluation, bleu score, rouge score, bertscore, human evaluation, a/b testing, benchmark, regression testing, llm as judge

---

### 4. ML Pipeline Workflow (`ml-pipeline-workflow.md`)
**When to use:** Building end-to-end MLOps pipelines

**Key Topics:**
- Pipeline architecture and DAG orchestration
- Data preparation and validation
- Feature engineering pipelines
- Model training orchestration
- Experiment tracking (MLflow, W&B)
- Model validation and comparison
- Deployment automation (canary, blue-green)
- Orchestration tools (Airflow, Dagster, Kubeflow)
- Data versioning and lineage
- Model registry integration
- Continuous training patterns

**Trigger Keywords:** ml pipeline, mlops, model training, model deployment, feature engineering, data pipeline, airflow, kubeflow, model validation, continuous training

---

### 5. LangChain Architecture (`langchain-architecture.md`)
**When to use:** Building LLM applications with LangChain framework

**Key Topics:**
- Agent types (ReAct, OpenAI Functions, Structured Chat)
- Chain patterns (LLMChain, SequentialChain, RouterChain, MapReduce)
- Memory systems (Buffer, Summary, Window, Entity, VectorStore)
- Document processing (loaders, splitters, vector stores, retrievers)
- Callback systems for monitoring
- Custom tool integration
- Agent testing strategies
- Performance optimization (caching, batching, streaming)
- Production patterns and best practices

**Trigger Keywords:** langchain, llm agents, langchain agents, chains, memory, tools integration, react agent, agent executor, callbacks

---

### 6. Agent Development Patterns (`agent-development-patterns.md`)
**When to use:** Building autonomous AI agents and multi-agent systems

**Key Topics:**
- Agent architecture (perception, reasoning, action, memory)
- Design patterns (ReAct, tool selection, multi-agent coordination)
- Agent with memory
- Self-improving agents
- Tool integration best practices
- Agent orchestration (sequential, parallel, hierarchical)
- Agent communication patterns
- Message passing and event systems
- Testing agent systems
- Production considerations

**Trigger Keywords:** agent development, ai agents, autonomous agents, agent architecture, agent orchestration, multi-agent system, tool calling, agent patterns

---

## Usage

These skills are automatically loaded when you use relevant trigger keywords in your conversations. You can also explicitly invoke them by mentioning the skill name.

### Example Usage

```
"I need to implement a RAG system for document Q&A"
→ Triggers: rag-implementation

"How do I optimize my prompts using few-shot learning?"
→ Triggers: prompt-engineering-patterns

"What metrics should I use to evaluate my LLM application?"
→ Triggers: llm-evaluation

"I want to build an ML pipeline with Airflow"
→ Triggers: ml-pipeline-workflow

"Help me create a LangChain agent with memory"
→ Triggers: langchain-architecture

"I need to build a multi-agent coordination system"
→ Triggers: agent-development-patterns
```

## Skill Integration

These skills work together for comprehensive ML/AI development:

1. **RAG + Prompt Engineering**: Optimize RAG prompts for better retrieval and generation
2. **LLM Evaluation + ML Pipeline**: Integrate evaluation into ML pipelines for continuous testing
3. **LangChain + Agent Development**: Build sophisticated agents using LangChain primitives
4. **ML Pipeline + All Skills**: Create end-to-end pipelines incorporating RAG, evaluation, and deployment

## Contributing

When adding new ML/AI skills:
1. Include comprehensive trigger keywords
2. Provide practical code examples
3. Cover both basics and advanced patterns
4. Include best practices and common pitfalls
5. Add testing and production considerations

## Skill Quality Standards

All skills in this directory meet these criteria:
- ✅ Production-ready patterns and code
- ✅ Framework-agnostic where possible
- ✅ Comprehensive coverage of topic
- ✅ Real-world examples and use cases
- ✅ Best practices and anti-patterns
- ✅ Testing and debugging guidance
- ✅ Performance optimization tips

## Future Skills

Potential additions:
- Fine-tuning and model training patterns
- Model compression and optimization
- Reinforcement learning for LLM agents
- Multi-modal AI systems (vision + language)
- Distributed training patterns
- Model interpretability and explainability
