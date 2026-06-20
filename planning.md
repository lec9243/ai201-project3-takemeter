# TakeMeter Planning: r/nba Take Quality Classifier

## Community and Task

I chose r/nba because the community has a large volume of public, text-heavy discussion and a wide range of discourse quality. In the same thread, users may write tactical analysis, make sweeping claims about players or teams, or post immediate emotional reactions to highlights and playoff results. That makes r/nba a good fit for a discourse-quality classifier because the difference between a reasoned basketball argument and a low-substance take is a distinction regular users already care about.

The classifier will label individual r/nba comments. I am focusing on comments rather than original posts because comments provide more varied discourse styles and enough examples to build a balanced dataset.

## Label Taxonomy

### `analysis`

A comment is `analysis` when it makes a structured basketball argument supported by specific reasoning, evidence, statistics, scheme observations, roster/cap context, matchup logic, or historical comparison.

Clear examples:

- "OKC can live with Minnesota's pull-up twos because Chet is staying home at the rim and the weak-side help is taking away the corner pass."
- "The Lakers' issue is not just shooting; it is that their best lineups ask LeBron to be the primary creator and primary transition defender for too many minutes."

### `hot_take`

A comment is `hot_take` when it makes a bold evaluative claim or prediction about a player, team, coach, or front office with little or no supporting evidence. The claim may be plausible, but the comment asserts more than it argues.

Clear examples:

- "This Wolves team is fake and will get exposed the second they play a real contender."
- "Tatum is never winning as the best player on a title team."

### `reaction`

A comment is `reaction` when it mainly expresses an immediate emotion, joke, meme, celebration, complaint, or low-context response to a play, result, quote, or news item. It does not try to make a durable basketball argument.

Clear examples:

- "That dunk was disgusting."
- "Lmao this thread is going to be unbearable tonight."

## Hard Edge Cases and Decision Rules

The hardest boundary is between `analysis` and `hot_take`. r/nba users often make a strong claim and attach one statistic or matchup reference. The decision rule is: if the evidence would still support the claim after removing the emotional framing, label it `analysis`; if the evidence is vague, cherry-picked, or only decorative, label it `hot_take`.

Example edge case:

- "Jokic is still the best player alive; every Denver half-court possession collapses without his passing."

This could be `hot_take` because "best player alive" is a sweeping claim, but it gives a concrete reason tied to offensive structure. I would label it `analysis` if the surrounding comment explains the passing effect, and `hot_take` if it stops at that sentence.

Another hard boundary is between `hot_take` and `reaction`. The decision rule is: if the comment makes a claim about what is true beyond the immediate moment, label it `hot_take`; if it only reacts to the moment, label it `reaction`.

Example edge case:

- "This team is cooked."

If it appears after one bad play with no broader argument, I label it `reaction`. If it appears as a general claim about playoff chances, I label it `hot_take`.

During annotation I will record difficult cases in the `notes` column and preserve at least three specific examples for the README.

Observed difficult cases from the collected dataset:

| Excerpt | Possible labels | Final label | Decision |
| --- | --- | --- | --- |
| "Dubs issue isn't just 'pressure'. It's his literal skill and athletic limitations..." | `analysis`, `hot_take` | `hot_take` | It uses basketball language, but the evidence is too thin; the comment mainly asserts a harsh player limitation claim. |
| "Yup. Brook is not completely unplayable in playoffs, Pacers are just an all-time offensive team..." | `analysis`, `hot_take` | `analysis` | The wording is strong, but the comment explains a matchup-specific reason, so the evidence survives without the inflammatory framing. |
| "Overrated is a wild take. The PG they traded for was a MVP candidate..." | `hot_take`, `reaction` | `reaction` | It pushes back on another comment in the moment, with emoji-style reaction framing, rather than developing a durable claim. |

## Data Collection Plan

I will collect public r/nba comments using the PullPush Reddit search API, which indexes publicly available Reddit data. I will not use private Discords, deleted comments, comments hidden behind authentication, or any private user data.

Target dataset size: at least 210 comments so the final dataset remains above the required 200 examples after cleaning.

Target label distribution:

- `analysis`: about 70 examples
- `hot_take`: about 70 examples
- `reaction`: about 70 examples

The first pass will collect general r/nba comments. If one class is underrepresented, I will collect additional public comments using basketball-related search terms likely to surface that discourse type, then review the labels rather than assigning them automatically from the query alone.

Filtering rules:

- Exclude deleted or removed comments.
- Exclude comments shorter than five words unless they are clear reactions.
- Exclude comments that are mostly URLs, bot messages, or moderation boilerplate.
- Keep only the comment text, label, and annotation notes in the released CSV.

If a label is below 20% after 200 examples, I will collect more examples for that label before training. If one label exceeds 70%, I will rebalance before moving to model training.

## Evaluation Metrics

I will report overall accuracy because the classifier must assign exactly one label to each comment. Accuracy alone is not enough because a model could score well by overpredicting a common class, so I will also report per-class precision, recall, and F1.

Macro F1 is especially important for this task because the labels represent different discourse functions. A model that handles `reaction` well but misses most `analysis` examples would not be useful, even if its overall accuracy looked acceptable. I will also report a confusion matrix to identify which label boundaries are hardest, especially `analysis` vs. `hot_take`.

## Definition of Success

For a real community tool, I would consider the classifier useful if it reaches at least 0.70 overall accuracy and at least 0.65 macro F1 on the test set, with no class F1 below 0.55. It should also beat the zero-shot Groq baseline on the same test set by at least five percentage points or show a clearer per-class balance.

For this course project, I would accept the model as "good enough" if it reaches at least 0.60 macro F1 and produces interpretable errors that point to real label-boundary issues rather than random guessing. Because the task is subjective and the dataset is small, a perfect score would be suspicious rather than automatically better.

## AI Tool Plan

### Label stress-testing

I will ask an AI tool to generate borderline r/nba comments that could plausibly fit two labels, especially `analysis` vs. `hot_take` and `hot_take` vs. `reaction`. If the generated examples are hard to classify using the current definitions, I will tighten the decision rules before annotating the full dataset.

### Annotation assistance

I may use AI assistance to draft labels for collected comments, but each label must be reviewed against the taxonomy before it is treated as final. Any AI-assisted annotation will be disclosed in the README. I will use the `notes` column to flag comments where the final label required judgment rather than being obvious.

### Failure analysis

After fine-tuning, I will give the wrong predictions to an AI tool and ask it to identify possible error patterns such as sarcasm, short comments, vague evidence, or confusion between bold claims and real analysis. I will verify any pattern myself by rereading the examples before writing the evaluation report.

## Stretch Feature Update

I am adding two stretch features after the required pipeline: an error pattern analysis and a deployed local interface. The error analysis will focus on whether one label pair dominates the mistakes, especially `hot_take` being predicted as `analysis`. The interface will be a simple local web app that accepts a new r/nba comment and displays the fine-tuned model's predicted label and confidence.
