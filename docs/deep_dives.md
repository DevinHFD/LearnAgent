# Deep Dives

## Core Engine (Week 1)

-   Unified LLM client abstraction
-   Schema-driven structured outputs (Pydantic + JSON schema)
-   Retry + repair self-healing layer
-   RunManager for full execution traceability
-   Tool execution layer

## Auto Code Loop (Week 2 Day 1)

-   LLM-generated code blocks
-   Automatic execution in venv
-   Error classification
-   Dependency self-repair via pip_install
-   Verifier-based goal checking

## Multi-Tool Agent (Week 2 Day 2)

-   LLM-driven action router
-   Tools: python_exec, pip_install, file_write, shell_exec
-   State-machine agent loop
-   Full artifact & action logging
