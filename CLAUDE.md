## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Testing

- Before declaring any feature/optimization "done" or testable, ALWAYS simulate-test it end-to-end with several DIFFERENT prompts (short, long, factual, multi-turn, edge cases), not just one canned prompt. Verify the output is coherent AND that generation stops correctly (EOG/stop tokens), across models (Q4/Q3/Q2/mixed) where relevant. A single passing prompt is not proof; regressions like runaway generation, template mismatch, or wrong stop tokens only show up across varied inputs.
- Run the GPU unit tests (`ctest --test-dir build`) and the real-model run on the GB10 box; report tok/s + a sample generation, not just "it builds".
