flowchart TD
  User[User Input]
  subgraph App
    A1[Streamlit UI]
    A2[Song Library & Profile]
    A3[Retriever]
    A4[LLM Prompt Builder]
    A5[AI Response]
  end
  subgraph Eval
    E1[Tester / Validator]
    E2[Logs & Guardrails]
  end

  User -->|ask for playlist advice| A1
  A1 --> A2
  A2 --> A3
  A3 --> A4
  A4 -->|prompt + context| A5
  A5 -->|recommendation| A1
  A1 -->|records usage| E2
  E1 -->|checks AI output| E2
  E2 -->|feedback / safety| A1