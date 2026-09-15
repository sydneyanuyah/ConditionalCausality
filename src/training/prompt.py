PROMPTS = {
    "zero_shot": """You are an information extraction system.

Task:

Given one input row with a field called `text`, produce exactly one additional field called `output`.

Goal:

`output` must be a JSON object containing one key, `predicted_tuples`, whose value is a JSON array of relation tuples supported by the text.

Tuple schema:

{
  "e1": string,
  "e2": string,
  "relation": string,
  "condition": string
}

Instructions:
1. Extract only relations explicitly stated in the text.
2. Preserve stated context, subgroup, population, setting, or other condition in the `condition` field.
3. Do not add facts that are only implied, speculative, or assumed.
4. Do not output duplicate tuples.
5. Keep entity and condition strings as close to the source wording as possible.
6. Normalize the relation into a concise label such as "increase", "decrease", "association", "inhibit", or "activate".
7. If no supported relation is present, return an empty list for `predicted_tuples`.
8. Return only valid JSON. No explanation.

Input:

{
  "text": "<TEXT>"
}

Required output format:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "...",
        "e2": "...",
        "relation": "...",
        "condition": "..."
      }
    ]
  }
}""",

    "cot": """You are an information extraction system.

Before answering, think step by step about the paragraph:

- identify the entities mentioned
- identify all explicitly stated relationships between entities
- determine whether any relationship is limited by a stated condition, subgroup, population, setting, time frame, or context
- discard anything that is implied, speculative, assumed, or duplicated
- then produce the final structured output

Task:

Given one input row with a field called `text`, produce exactly one additional field called `output`.

Goal:

`output` must be a JSON object containing one key, `predicted_tuples`, whose value is a JSON array of relation tuples supported by the text.

Tuple schema:

{
  "e1": string,
  "e2": string,
  "relation": string,
  "condition": string
}

Instructions:
1. Read the full paragraph carefully before extracting.
2. Identify entities as nouns, noun phrases, or concepts involved in actions or relationships.
3. Extract only relations explicitly stated in the text.
4. Preserve any stated context, subgroup, population, setting, time frame, or other condition in the `condition` field.
5. Do not add facts that are only implied, speculative, or assumed.
6. Do not output duplicate tuples.
7. Keep entity and condition strings as close to the source wording as possible.
8. Normalize the relation into a concise label such as "increase", "decrease", "association", "inhibit", or "activate".
9. If no supported relation is present, return an empty list for `predicted_tuples`.
10. Return only valid JSON. No explanation.

Input:

{
  "text": "<TEXT>"
}

Required output format:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "...",
        "e2": "...",
        "relation": "...",
        "condition": "..."
      }
    ]
  }
}""",

    "one_shot": """You are an information extraction system.

Think step by step before answering, but return only the final JSON.

Task:

Given one input row with a field called `text`, produce exactly one additional field called `output`.

Goal:

`output` must be a JSON object containing one key, `predicted_tuples`, whose value is a JSON array of relation tuples supported by the text.

Tuple schema:

{
  "e1": string,
  "e2": string,
  "relation": string,
  "condition": string
}

Instructions:
1. Extract only relations explicitly stated in the text.
2. Preserve stated context, subgroup, population, setting, or other condition in the `condition` field.
3. Do not add facts that are only implied, speculative, or assumed.
4. Do not output duplicate tuples.
5. Keep entity and condition strings as close to the source wording as possible.
6. Normalize the relation into a concise label such as "increase", "decrease", "association", "inhibit", or "activate" when appropriate.
7. If no supported relation is present, return an empty list for `predicted_tuples`.
8. Return only valid JSON. No explanation.

Example:

Input:

{
  "text": "Clinically significant weight loss is achievable and sustainable in this setting. Many prostate cancer and cardiovascular disease biomarkers favorably improved with the intervention, which may reduce the risk of cardiovascular disease and, potentially, recurrence"
}

Output:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "Many prostate cancer",
        "e2": "the intervention",
        "relation": "favorably improved",
        "condition": "may reduce the risk of cardiovascular disease and, potentially, recurrence"
      },
      {
        "e1": "cardiovascular disease biomarkers",
        "e2": "the intervention",
        "relation": "favorably improved",
        "condition": "may reduce the risk of cardiovascular disease and, potentially, recurrence"
      }
    ]
  }
}

Now process this input:

{
  "text": "<TEXT>"
}

Required output format:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "...",
        "e2": "...",
        "relation": "...",
        "condition": "..."
      }
    ]
  }
}""",

    "two_shot": """You are an information extraction system.

Think step by step before answering, but return only the final JSON.

Task:

Given one input row with a field called `text`, produce exactly one additional field called `output`.

Goal:

`output` must be a JSON object containing one key, `predicted_tuples`, whose value is a JSON array of relation tuples supported by the text.

Tuple schema:

{
  "e1": string,
  "e2": string,
  "relation": string,
  "condition": string
}

Instructions:
1. Extract only relations explicitly stated in the text.
2. Preserve stated context, subgroup, population, setting, or other condition in the `condition` field.
3. Do not add facts that are only implied, speculative, or assumed.
4. Do not output duplicate tuples.
5. Keep entity and condition strings as close to the source wording as possible.
6. Normalize the relation into a concise label such as "increase", "decrease", "association", "inhibit", or "activate" when appropriate.
7. If no supported relation is present, return an empty list for `predicted_tuples`.
8. Return only valid JSON. No explanation.

Example 1:

Input:

{
  "text": "Clinically significant weight loss is achievable and sustainable in this setting. Many prostate cancer and cardiovascular disease biomarkers favorably improved with the intervention, which may reduce the risk of cardiovascular disease and, potentially, recurrence"
}

Output:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "Many prostate cancer",
        "e2": "the intervention",
        "relation": "favorably improved",
        "condition": "may reduce the risk of cardiovascular disease and, potentially, recurrence"
      },
      {
        "e1": "cardiovascular disease biomarkers",
        "e2": "the intervention",
        "relation": "favorably improved",
        "condition": "may reduce the risk of cardiovascular disease and, potentially, recurrence"
      }
    ]
  }
}

Example 2:

Input:

{
  "text": "The Figure shows the net percent weight loss of all participants. Our calorie restriction plan, coaching healthy nutrition practices, and exercise led to 5.5% weight loss from baseline to radical prostatectomy and a net loss of 11% initial body weight 6 months after surgery. The intervention group improved diet quality, achieved fat loss, enhanced general and emotional health, and improved prostate cancer biomarkers that are also associated with weight/fat loss (insulin, leptin:adiponectin ratio, cholesterol)"
}

Output:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "Our calorie restriction plan",
        "e2": "5.5% weight loss from baseline to radical prostatectomy",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Coaching healthy nutrition practices",
        "e2": "5.5% weight loss from baseline to radical prostatectomy",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Exercise",
        "e2": "5.5% weight loss from baseline to radical prostatectomy",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Our calorie restriction plan",
        "e2": "a net loss of 11% initial body weight",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Coaching healthy nutrition practices",
        "e2": "a net loss of 11% initial body weight",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Exercise",
        "e2": "a net loss of 11% initial body weight",
        "relation": "led to",
        "condition": "6 months after surgery"
      }
    ]
  }
}

Now process this input:

{
  "text": "<TEXT>"
}

Required output format:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "...",
        "e2": "...",
        "relation": "...",
        "condition": "..."
      }
    ]
  }
}""",

    "two_shot_cot": """You are an information extraction system.

Before answering, think step by step about the paragraph:

- identify the entities mentioned
- identify all explicitly stated relationships between entities
- determine whether any relationship is limited by a stated condition, subgroup, population, setting, time frame, or context
- discard anything that is implied, speculative, assumed, or duplicated
- then produce the final structured output

Task:

Given one input row with a field called `text`, produce exactly one additional field called `output`.

Goal:

`output` must be a JSON object containing one key, `predicted_tuples`, whose value is a JSON array of relation tuples supported by the text.

Tuple schema:

{
  "e1": string,
  "e2": string,
  "relation": string,
  "condition": string
}

Instructions:
1. Extract only relations explicitly stated in the text.
2. Preserve stated context, subgroup, population, setting, or other condition in the `condition` field.
3. Do not add facts that are only implied, speculative, or assumed.
4. Do not output duplicate tuples.
5. Keep entity and condition strings as close to the source wording as possible.
6. Normalize the relation into a concise label such as "increase", "decrease", "association", "inhibit", or "activate" when appropriate.
7. If no supported relation is present, return an empty list for `predicted_tuples`.
8. For the examples below, show the reasoning process.
9. For the final input, return only valid JSON. No explanation.

Example 1:

Input:

{
  "text": "Clinically significant weight loss is achievable and sustainable in this setting. Many prostate cancer and cardiovascular disease biomarkers favorably improved with the intervention, which may reduce the risk of cardiovascular disease and, potentially, recurrence"
}

Reasoning:

1. Entities mentioned:
- clinically significant weight loss
- this setting
- many prostate cancer
- cardiovascular disease biomarkers
- the intervention
- the risk of cardiovascular disease
- recurrence

2. Explicitly stated relationships:
- Many prostate cancer → favorably improved → the intervention
- cardiovascular disease biomarkers → favorably improved → the intervention

3. Conditions or contextual restrictions:
- The phrase "may reduce the risk of cardiovascular disease and, potentially, recurrence" is attached in the annotation as the condition for both extracted tuples.

4. Discarded content:
- "Clinically significant weight loss is achievable and sustainable in this setting" is not used as a tuple here.
- No duplicate tuples are kept.

Final Output:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "Many prostate cancer",
        "e2": "the intervention",
        "relation": "favorably improved",
        "condition": "may reduce the risk of cardiovascular disease and, potentially, recurrence"
      },
      {
        "e1": "cardiovascular disease biomarkers",
        "e2": "the intervention",
        "relation": "favorably improved",
        "condition": "may reduce the risk of cardiovascular disease and, potentially, recurrence"
      }
    ]
  }
}

Example 2:

Input:

{
  "text": "The Figure shows the net percent weight loss of all participants. Our calorie restriction plan, coaching healthy nutrition practices, and exercise led to 5.5% weight loss from baseline to radical prostatectomy and a net loss of 11% initial body weight 6 months after surgery. The intervention group improved diet quality, achieved fat loss, enhanced general and emotional health, and improved prostate cancer biomarkers that are also associated with weight/fat loss (insulin, leptin:adiponectin ratio, cholesterol)"
}

Reasoning:

1. Entities mentioned:
- The Figure
- the net percent weight loss of all participants
- Our calorie restriction plan
- coaching healthy nutrition practices
- exercise
- 5.5% weight loss from baseline to radical prostatectomy
- a net loss of 11% initial body weight
- 6 months after surgery
- The intervention group
- diet quality
- fat loss
- general and emotional health
- prostate cancer biomarkers
- weight/fat loss
- insulin
- leptin:adiponectin ratio
- cholesterol

2. Explicitly stated relationships:
- Our calorie restriction plan → led to → 5.5% weight loss from baseline to radical prostatectomy
- Coaching healthy nutrition practices → led to → 5.5% weight loss from baseline to radical prostatectomy
- Exercise → led to → 5.5% weight loss from baseline to radical prostatectomy
- Our calorie restriction plan → led to → a net loss of 11% initial body weight
- Coaching healthy nutrition practices → led to → a net loss of 11% initial body weight
- Exercise → led to → a net loss of 11% initial body weight

3. Conditions or contextual restrictions:
- The condition attached to these extracted tuples is "6 months after surgery".

4. Discarded content:
- "The Figure shows the net percent weight loss of all participants" is not extracted as a tuple here.
- "The intervention group improved diet quality, achieved fat loss, enhanced general and emotional health, and improved prostate cancer biomarkers..." is not included in these gold tuples.
- No duplicate tuples are kept.

Final Output:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "Our calorie restriction plan",
        "e2": "5.5% weight loss from baseline to radical prostatectomy",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Coaching healthy nutrition practices",
        "e2": "5.5% weight loss from baseline to radical prostatectomy",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Exercise",
        "e2": "5.5% weight loss from baseline to radical prostatectomy",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Our calorie restriction plan",
        "e2": "a net loss of 11% initial body weight",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Coaching healthy nutrition practices",
        "e2": "a net loss of 11% initial body weight",
        "relation": "led to",
        "condition": "6 months after surgery"
      },
      {
        "e1": "Exercise",
        "e2": "a net loss of 11% initial body weight",
        "relation": "led to",
        "condition": "6 months after surgery"
      }
    ]
  }
}

Now process this input:

{
  "text": "<TEXT>"
}

Return only:

{
  "output": {
    "predicted_tuples": [
      {
        "e1": "...",
        "e2": "...",
        "relation": "...",
        "condition": "..."
      }
    ]
  }
}"""
}
