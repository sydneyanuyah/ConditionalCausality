PROMPTS = {
    "zero_shot": """
You are given a ground-truth paragraph and a question statement.

The paragraph is the only source of truth.

Your task is to determine whether the question statement is supported by the paragraph.

Use only the paragraph. Do not use outside knowledge.

Decision labels:
- Supported: The paragraph supports the question statement.
- Not Supported: The question statement contradicts the paragraph.
- Not Enough Evidence: The paragraph does not provide enough evidence to verify the question statement.

Reason type labels:
1. The entities, their relationship, and the condition all hold true in the paragraph.
2. The condition is replaced, so we do not know if the statement holds true.
3. There is no way to ascertain if the relationship holds true outside the stated condition.
4. The statement contradicts the condition in the paragraph.
5. The relationship contradicts the paragraph.

Input:
Ground-Truth Paragraph:
{paragraph}

Question Statement:
{probe_question}

Return only valid JSON:
{{
  "decision": "Supported | Not Supported | Not Enough Evidence",
  "reason_type": 1,
  "explanation": "One short sentence explaining the decision."
}}
""".strip(),

    "one_shot": """
You are given a ground-truth paragraph and a question statement.

The paragraph is the only source of truth.

Your task is to determine whether the question statement is supported by the paragraph.

Use only the paragraph. Do not use outside knowledge.

Decision labels:
- Supported: The paragraph supports the question statement.
- Not Supported: The question statement contradicts the paragraph.
- Not Enough Evidence: The paragraph does not provide enough evidence to verify the question statement.

Reason type labels:
1. The entities, their relationship, and the condition all hold true in the paragraph.
2. The condition is replaced, so we do not know if the statement holds true.
3. There is no way to ascertain if the relationship holds true outside the stated condition.
4. The statement contradicts the condition in the paragraph.
5. The relationship contradicts the paragraph.

Example:

Ground-Truth Paragraph:
The study note says that Belmor resin reduces Nalto spread when the storage room is dark. The paragraph limits the relationship to the dark storage-room condition and does not state that Belmor resin reduces Nalto spread generally.

Question Statement:
Does Belmor resin reduce Nalto spread?

Answer:
{{
  "decision": "Not Enough Evidence",
  "reason_type": 3,
  "explanation": "The paragraph only supports the relationship under the dark storage-room condition, not generally."
}}

Now answer the next item.

Ground-Truth Paragraph:
{paragraph}

Question Statement:
{probe_question}

Return only valid JSON:
{{
  "decision": "Supported | Not Supported | Not Enough Evidence",
  "reason_type": 1,
  "explanation": "One short sentence explaining the decision."
}}
""".strip(),

    "two_shot": """
You are given a ground-truth paragraph and a question statement.

The paragraph is the only source of truth.

Your task is to determine whether the question statement is supported by the paragraph.

Use only the paragraph. Do not use outside knowledge.

Decision labels:
- Supported: The paragraph supports the question statement.
- Not Supported: The question statement contradicts the paragraph.
- Not Enough Evidence: The paragraph does not provide enough evidence to verify the question statement.

Reason type labels:
1. The entities, their relationship, and the condition all hold true in the paragraph.
2. The condition is replaced, so we do not know if the statement holds true.
3. There is no way to ascertain if the relationship holds true outside the stated condition.
4. The statement contradicts the condition in the paragraph.
5. The relationship contradicts the paragraph.

Example 1:

Ground-Truth Paragraph:
The field record states that Mavex oil improves Torlan clarity when the inner pipe is unlocked. The paragraph does not support the relationship when the inner pipe is locked.

Question Statement:
Does Mavex oil improve Torlan clarity when the inner pipe is locked?

Answer:
{{
  "decision": "Not Supported",
  "reason_type": 4,
  "explanation": "The question contradicts the paragraph by using the locked-pipe condition instead of the unlocked-pipe condition."
}}

Example 2:

Ground-Truth Paragraph:
The technical summary says that Rendal foam decreases Jorin heat when the glass shield is raised. The relationship is stated only for the raised glass-shield condition.

Question Statement:
Does Rendal foam increase Jorin heat when the glass shield is raised?

Answer:
{{
  "decision": "Not Supported",
  "reason_type": 5,
  "explanation": "The question contradicts the paragraph by changing the relationship from decreasing heat to increasing heat."
}}

Now answer the next item.

Ground-Truth Paragraph:
{paragraph}

Question Statement:
{probe_question}

Return only valid JSON:
{{
  "decision": "Supported | Not Supported | Not Enough Evidence",
  "reason_type": 1,
  "explanation": "One short sentence explaining the decision."
}}
""".strip(),

    "cot_two_shot": """
You are given a ground-truth paragraph and a question statement.

The paragraph is the only source of truth.

Your task is to determine whether the question statement is supported by the paragraph.

Use only the paragraph. Do not use outside knowledge.

Before deciding, check:
1. Whether the same entities are being discussed.
2. Whether the same relationship is being claimed.
3. Whether the condition is preserved, removed, replaced, or contradicted.

Decision labels:
- Supported: The paragraph supports the question statement.
- Not Supported: The question statement contradicts the paragraph.
- Not Enough Evidence: The paragraph does not provide enough evidence to verify the question statement.

Reason type labels:
1. The entities, their relationship, and the condition all hold true in the paragraph.
2. The condition is replaced, so we do not know if the statement holds true.
3. There is no way to ascertain if the relationship holds true outside the stated condition.
4. The statement contradicts the condition in the paragraph.
5. The relationship contradicts the paragraph.

Example 1:

Ground-Truth Paragraph:
The report states that Varnex dust increases Lomar stability when the cooling gate is sealed. The paragraph keeps this relationship tied to the sealed cooling gate and does not claim that Varnex dust increases Lomar stability under other gate settings.

Question Statement:
Does Varnex dust increase Lomar stability when the cooling gate is sealed?

Reasoning:
The entities are the same. The relationship is the same. The condition is also the same.

Answer:
{{
  "decision": "Supported",
  "reason_type": 1,
  "explanation": "The question preserves the same entities, relationship, and condition stated in the paragraph."
}}

Example 2:

Ground-Truth Paragraph:
The lab note says that Dravon liquid lowers Meris pressure when the filter wall is dry. The paragraph does not discuss Dravon liquid when the filter wall is wet or under any other filter-wall condition.

Question Statement:
Does Dravon liquid lower Meris pressure when the filter wall is wet?

Reasoning:
The entities are the same. The relationship is the same. However, the condition has been replaced from a dry filter wall to a wet filter wall.

Answer:
{{
  "decision": "Not Enough Evidence",
  "reason_type": 2,
  "explanation": "The question replaces the dry filter-wall condition with a wet filter-wall condition not supported by the paragraph."
}}

Now answer the next item.

Ground-Truth Paragraph:
{paragraph}

Question Statement:
{probe_question}

Think through the entity, relationship, and condition checks before answering.

Return only valid JSON:
{{
  "decision": "Supported | Not Supported | Not Enough Evidence",
  "reason_type": 1,
  "explanation": "One short sentence explaining the decision."
}}
""".strip()
}
