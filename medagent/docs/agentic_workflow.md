# MedAgent MVP Agentic Workflow

## End-to-end flow

1. Input source
- SynthLab wrapper loads multimodal synthetic patients.
- File: `medagent/runtime/harness/synthlab_runner.py`

2. CPB construction
- Patient records are normalized into Canonical Patient Bundle (`CPB`).
- Includes timeline, EHR text, structured problems/meds, imaging refs, genomics variants.

3. Orchestrator
- Executes specialist agents and enforces final synthesis/stop conditions.
- File: `medagent/runtime/agents/orchestrator/engine.py`

4. Specialist generation
- Data steward normalization
- Genotype summary + BioMCP query bundle
- Imaging summary
- MedGemma draft SOAP/differential (current MVP placeholder)

5. Claim extraction
- Draft output is converted to claim objects (`must_verify` aware).
- File: `medagent/runtime/agents/biomcp_verifier/agent.py`

6. RAG
- Local retrieval over documentation corpus (`medagent/docs`, `synthlab/docs` by default).
- File: `medagent/runtime/tools/retrieval/simple_rag.py`

7. Reasoning
- Optional OpenAI model (`gpt-5.2` default) classifies claim support as pass/weak/fail.
- File: `medagent/runtime/tools/openai_client/client.py`
- Fallback: if API/client/network fails, verifier falls back to offline mock mode.

8. Tool calling
- Current MVP: local RAG retrieval + optional OpenAI API call.
- Optional extension: Biomni adapter for richer tool orchestration and MCP usage.
- File: `medagent/runtime/tools/exec_env/biomni_adapter.py`

9. Causal robustness
- Counterfactual sensitivity checks produce sensitivity map.
- File: `medagent/runtime/agents/causal_verifier/agent.py`

10. Final synthesis
- Produces SOAP, ranked problems, ranked non-prescriptive plan options, evidence table, sensitivity map, uncertainty/escalation guidance.
- File: `medagent/runtime/agents/synthesis/agent.py`

## Runtime modes

- Offline-safe mode (default): `MEDAGENT_USE_OPENAI=0`
- OpenAI reasoning mode: `MEDAGENT_USE_OPENAI=1`
- Biomni adapter mode: available for planned integration path
