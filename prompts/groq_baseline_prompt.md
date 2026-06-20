# Groq Zero-Shot Baseline Prompt

You are classifying r/nba comments into exactly one label.

Labels:

- `analysis`: structured basketball argument supported by specific reasoning, evidence, statistics, scheme observations, roster/cap context, matchup logic, or historical comparison.
- `hot_take`: bold evaluative claim or prediction about a player, team, coach, or front office with little or no supporting evidence.
- `reaction`: immediate emotion, joke, meme, celebration, complaint, or low-context response to a play, result, quote, or news item.

Decision rules:

- If specific evidence would still support the claim after removing emotional wording, choose `analysis`.
- If the comment makes a broader claim without enough support, choose `hot_take`.
- If it mainly reacts to the moment without a durable argument, choose `reaction`.

Return only one of these labels:

`analysis`, `hot_take`, `reaction`

Comment:

`{comment_text}`
