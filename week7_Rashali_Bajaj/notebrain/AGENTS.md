# NoteBrain QA — Project Rules for Claude Code

Read docs/PRD.md in full before making any changes. It is the source of truth for
scope, tech stack, and feature boundaries.

Rules:
- Only implement what the current phase prompt explicitly asks for. Do not
  implement future phases early, even if related code would be convenient to add
  now.
- Do not add features, libraries, or files not listed in docs/PRD.md Section 11
  (Tech Stack) or explicitly requested in the current phase prompt.
- Do not modify files from previous phases beyond what the current phase requires,
  unless the phase prompt says so.
- This is a time-boxed course assignment. Prioritize simple, working code over
  abstraction, extensibility, or premature optimization.
- Tech stack is locked: Streamlit, pypdf, sentence-transformers, ChromaDB,
  rank_bm25, Groq API. Do not substitute or add major dependencies without
  flagging it to me first.
- After each phase, the project must run without errors via `streamlit run app.py`.
- At the end of every phase, always finish your response with: (1) a short
  summary of exactly what was implemented/changed in this phase, and (2) a
  clear, numbered list of manual checks I should personally run before we move
  to the next phase prompt. Do not skip this even if it feels repetitive.
