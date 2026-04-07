BACKWARD_prompt_EVAL = """You are an expert evaluator for relation triplet extraction.

Given the extracted triplets: __predict_triplets__,
and their NLI-based self-evaluation results: __evaluate_result__,

provide concise, structured, and actionable feedback on how to improve the extraction quality, focusing on the following aspects:

1. Correctness
2. Completeness
3. Clarity

Important constraints:
- Assume that the target relation schema is fixed.
- The fixed canonical relation schema is:
  CAUSES, CONVERTED_TO, LOCATED_IN, OCCURS_DURING, INCREASES, DECREASES, DOMINATES
- Do NOT recommend introducing new relation labels.
- Do NOT recommend replacing canonical relation names with alternative predicates.
- When triplets are suboptimal, suggest how they could be better expressed using the existing canonical relations.
- Focus on improving relation choice, argument direction, entity boundaries, and schema-consistent formulation.

Your feedback should explain how the predicted triplets can be improved to enhance the evaluation metrics while preserving the fixed relation schema.
"""

BACKWARD_prompt_PRED = """You are an expert evaluator for Open-domain Relation Triplet Extraction (ORTE).

Given

- the structured system prompt: __sys_prompt__,
- the input sentence: __input_sentence__,
- the extracted triplets produced under this prompt: __extract_response__,
- and the NLI-based feedback on the extraction results: __backward_response__,

provide concise and actionable feedback on how to improve the system prompt, so as to improve the evaluation indicators of relation triplet extraction in terms of:

1. Correctness
2. Completeness
3. Clarity

Important constraints:
- The target relation schema is fixed and must not be expanded.
- The fixed canonical relation schema is:
  CAUSES, CONVERTED_TO, LOCATED_IN, OCCURS_DURING, INCREASES, DECREASES, DOMINATES
- Do NOT suggest introducing new relation labels.
- Do NOT suggest replacing the canonical schema with alternative predicates.
- Do NOT suggest adding examples that use predicates outside the fixed canonical schema.
- Improvements must focus only on how to better use the existing relation schema, improve entity selection, improve argument direction, improve handling of temporal expressions, and improve clarity of extraction.

Focus on structural and instructional improvements to the system prompt that better guide the model’s extraction behavior while preserving the fixed relation schema.
"""

OPTIMIZER_EXAMPLE_TEMPLATE = """
---
Here is a conversation:

<CONVERSATION>
<LM_SYSTEM_PROMPT> __sys_prompt__ </LM_SYSTEM_PROMPT>
<LM_INPUT> __input_sentence__ </LM_INPUT>
<LM_OUTPUT> __extract_response__ </LM_OUTPUT>
</CONVERSATION>

This conversation is potentially part of a larger system. The output is used as the response from the language model.

The target relation schema is fixed and must not be expanded.

Fixed canonical relation schema:
CAUSES, CONVERTED_TO, LOCATED_IN, OCCURS_DURING, INCREASES, DECREASES, DOMINATES

Here is the feedback we got for the structured system prompt used for the Relation Triplet Extraction task:
<FEEDBACK>__feedback__</FEEDBACK>
---
"""

OPTIMIZER_prompt = """You are improving a structured system prompt used for the Open-domain Relation Triplet Extraction (ORTE) task.

The variable to improve is:
<VARIABLE> __sys_prompt__ </VARIABLE>

Given the contextual feedback related to this prompt:
<CONTEXT> {__update_insert_examples__} </CONTEXT>

Your task is to optimize the ORTE system prompt by following the improvement strategies in the context, making it clearer, more effective, and more concise, while preserving the original triplet extraction format.

Constraints:
- Preserve the fixed target relation schema.
- The fixed canonical relation schema is:
  CAUSES, CONVERTED_TO, LOCATED_IN, OCCURS_DURING, INCREASES, DECREASES, DOMINATES
- Do NOT introduce any relation outside this schema.
- Do NOT add example predicates outside the canonical schema.
- Do NOT replace the canonical relation inventory with alternative relation names.
- Do NOT change the output format.
- Do NOT introduce instructions that encourage relation labels outside the fixed schema.
- Only improve instructions related to correctness, completeness, clarity, entity span selection, argument direction, temporal formulation, and schema-consistent extraction.

Output Format:
<IMPROVED_PROMPT>
{The improved structured system prompt only.}
</IMPROVED_PROMPT>
"""