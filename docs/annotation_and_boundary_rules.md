# Annotation and boundary rules

An extraction annotation is a condition-scoped tuple with fields `e1`, `relation`,
`e2`, and `condition`. The condition must remain attached to the relation rather
than being treated as optional background. Multiple tuples may be annotated for a
paragraph.

Probe labels use three classes:

- `Supported`: the paragraph supports the complete probe claim under its stated
  condition.
- `Not Supported`: the paragraph contradicts the relation or condition asserted by
  the probe.
- `Not Enough Evidence`: the paragraph does not license the altered or unconditioned
  claim, even if it could be plausible elsewhere.

All P1–P5 rows for the same `Tuple_ID` must stay in the same split. When a source
document identifier is available, all paragraphs from that document must also stay
together. Blank evidence must not be converted to support, and external knowledge
must not override the paragraph.

