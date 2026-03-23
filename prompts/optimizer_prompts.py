BACKWARD_prompt_EVAL = """You are an expert evaluator for relation triplet extraction.

Given the extracted triplets: __predict_triplets__,
 and their NLI-based self-evaluation results: __evaluate_result__,

provide **concise, structured, and actionable feedback** on **how to improve the extraction quality**, focusing on the following aspects:

1. Correctness
2. Completeness
3. Clarity

...

Your feedback should explain **how the predicted triplets can be improved** to enhance the evaluation metrics."""

BACKWARD_prompt_PRED = """You are an expert evaluator for Open-domain Relation Triplet Extraction (ORTE).

Given

- the structured system prompt: __sys_prompt__,
- the input sentence: __input_sentence__,
- the extracted triplets produced under this prompt: __extract_response__,
- and the NLI-based feedback on the extraction results: __backward_response__,

provide **concise and actionable feedback on how to improve the system prompt**, so as to improve the evaluation indicators of relation triplet extraction in terms of:

1. **Correctness**,
2. **Completeness**, and
3. **Clarity**.

...

Focus on **structural and instructional improvements** to the system prompt that better guide the model’s extraction behavior."""



OPTIMIZER_EXAMPLE_TEMPLATE = """
---
Here is a conversation:

<CONVERSATION>
<LM_SYSTEM_PROMPT> __sys_prompt__ </LM_SYSTEM_PROMPT>
<LM_INPUT> __input_sentence__ </LM_INPUT>
<LM_OUTPUT> __extract_response__ </LM_OUTPUT>
</CONVERSATION>

This conversation is potentially part of a larger system. The output is used as response from the language model

Here is the feedback we got for structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the Relation Triplet Extraction task in the conversation:
<FEEDBACK>__feedback__</FEEDBACK>
---
"""

OPTIMIZER_prompt = """You are improving a structured system prompt used for the Open-domain Relation Triplet Extraction (ORTE) task.

The variable to improve is:
 <VARIABLE> __sys_prompt__ </VARIABLE>

Given the contextual feedback related to this prompt:
 <CONTEXT> {__update_insert_examples__} </CONTEXT>

Your task is to **optimize the ORTE system prompt** by following the improvement strategies in the context, making it **clearer, more effective, and more concise**, while preserving the original triplet extraction format.

Constraints:

...

Output Format:
 <IMPROVED_PROMPT>
 {The improved structured system prompt only.}
 </IMPROVED_PROMPT>
"""