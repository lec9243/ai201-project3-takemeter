# TakeMeter: r/nba Discourse Quality Classifier

TakeMeter is a text classifier for r/nba comments. It sorts comments into three discourse types that regular NBA fans recognize: structured basketball analysis, unsupported hot takes, and immediate reactions.

Current status: planning, dataset setup, Colab fine-tuning, Groq zero-shot baseline, and evaluation write-up are complete.

## Community

I chose r/nba because it has active public discussion, high comment volume, and a clear mix of discourse styles. The same game thread can contain tactical analysis, sweeping player judgments, referee complaints, memes, and emotional reactions. That makes it a good community for studying whether a model can distinguish argument quality rather than just basketball topic.

## Labels

| Label | Definition | Clear examples |
| --- | --- | --- |
| `analysis` | A structured basketball argument supported by specific reasoning, evidence, statistics, scheme observations, roster/cap context, matchup logic, or historical comparison. | "The triangle works against any and all defenses because it demands... spacing." / "Dallas doesn't need more defense. They need an offensive PG..." |
| `hot_take` | A bold evaluative claim or prediction about a player, team, coach, or front office with little or no supporting evidence. | "Playoff Jimmy isn't a thing..." / "Celtics exposed, fraudulent championship confirmed." |
| `reaction` | An immediate emotion, joke, meme, celebration, complaint, or low-context response to a play, result, quote, or news item. | "What is this 5 guard lineup lmao?" / "oh wow, did the game continue after I commented?" |

Decision rules:

- If specific evidence would still support the claim after removing emotional wording, label it `analysis`.
- If the comment makes a broader claim without enough support, label it `hot_take`.
- If it mainly reacts to the moment without a durable argument, label it `reaction`.

## Dataset

The labeled dataset is [data/takemeter_nba_labeled.csv](data/takemeter_nba_labeled.csv). It contains 210 public r/nba comments collected through the PullPush Reddit search API. The selected comments are from May 15, 2025 through May 19, 2025. The CSV includes `text`, `label`, `notes`, `source_permalink`, and `comment_id`.

Raw API cache files are not included in the repo because they were temporary collection artifacts; the released dataset is the cleaned labeled CSV. The collection helper is [scripts/collect_pullpush_dataset.py](scripts/collect_pullpush_dataset.py).

Label distribution:

| Label | Count |
| --- | ---: |
| `analysis` | 70 |
| `hot_take` | 70 |
| `reaction` | 70 |

Annotation process:

1. I defined the labels in [planning.md](planning.md) before collecting the final dataset.
2. I collected public r/nba comments using basketball, reaction, and hot-take search terms.
3. I used a rule-based draft labeler based on the taxonomy, then flagged borderline examples in the `notes` column.
4. I reviewed the labels against the taxonomy before final submission and kept the difficult cases documented below. This matters because query terms and rules can bias the dataset toward obvious keywords.

Three difficult examples:

| Excerpt | Possible labels | Final label | Decision |
| --- | --- | --- | --- |
| "Dubs issue isn't just 'pressure'..." | `analysis`, `hot_take` | `hot_take` | It uses basketball language, but the comment is mostly a blunt player limitation claim without developed support. |
| "Yup. Brook is not completely unplayable..." | `analysis`, `hot_take` | `analysis` | The wording is strong, but the comment explains matchup context and why the Pacers are a bad fit for him. |
| "Overrated is a wild take..." | `hot_take`, `reaction` | `reaction` | It responds to another take in the moment and uses emoji-style reaction framing rather than building an argument. |

## Fine-Tuning Approach

The fine-tuned model is `distilbert-base-uncased`, trained as a three-class sequence classifier in [Copy_of_ai201_project3_takemeter_starter_clean.ipynb](Copy_of_ai201_project3_takemeter_starter_clean.ipynb). The notebook loads the CSV, creates a 70% / 15% / 15% train-validation-test split, trains DistilBERT, evaluates on the locked test set, and exports the result files committed here. A standalone backup script is included at [scripts/run_takemeter_colab.py](scripts/run_takemeter_colab.py).

Key hyperparameters:

| Hyperparameter | Value | Reason |
| --- | ---: | --- |
| Epochs | 3 | Small dataset; more epochs would increase overfitting risk. |
| Learning rate | `2e-5` | Standard conservative learning rate for BERT-style fine-tuning. |
| Batch size | 16 | Fits comfortably on a free Colab T4 for short comments. |
| Max length | 256 tokens | Most r/nba comments are short or medium length; this preserves context while still fitting comfortably on the T4. |

To reproduce the run in Colab, upload [Copy_of_ai201_project3_takemeter_starter_clean.ipynb](Copy_of_ai201_project3_takemeter_starter_clean.ipynb) and [data/takemeter_nba_labeled.csv](data/takemeter_nba_labeled.csv), set the runtime to T4 GPU, add `GROQ_API_KEY` through Colab Secrets for the Groq baseline section, and run the notebook cells in order.

## Baseline

The zero-shot baseline uses Groq's `llama-3.3-70b-versatile` on the same locked test split as the fine-tuned model. The prompt, saved in [prompts/groq_baseline_prompt.md](prompts/groq_baseline_prompt.md), defines the task as classifying public r/nba comments by discourse function, gives the same definitions for `analysis`, `hot_take`, and `reaction` used in [planning.md](planning.md), includes one example per label, and instructs the model to output only one label name: `analysis`, `hot_take`, or `reaction`.

Baseline results were collected in Colab using the same 32-example test split as the fine-tuned model. All 32 Groq responses were parseable.

## Evaluation Report

The Colab notebook wrote:

- `evaluation_results.json`
- `confusion_matrix.png`

Overall results:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Groq zero-shot baseline | 0.719 | 0.710 |
| Fine-tuned DistilBERT | 0.438 | 0.340 |

Per-class metrics:

| Model | Label | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Groq zero-shot | `analysis` | 0.800 | 0.800 | 0.800 |
| Groq zero-shot | `hot_take` | 0.830 | 0.450 | 0.590 |
| Groq zero-shot | `reaction` | 0.620 | 0.910 | 0.740 |
| Fine-tuned DistilBERT | `analysis` | 0.500 | 1.000 | 0.670 |
| Fine-tuned DistilBERT | `hot_take` | 0.360 | 0.360 | 0.360 |
| Fine-tuned DistilBERT | `reaction` | 0.000 | 0.000 | 0.000 |

Fine-tuned confusion matrix:

| True \ Predicted | `analysis` | `hot_take` | `reaction` |
| --- | ---: | ---: | ---: |
| `analysis` | 10 | 0 | 0 |
| `hot_take` | 6 | 4 | 1 |
| `reaction` | 4 | 7 | 0 |

Wrong prediction analysis:

1. True `reaction`, predicted `analysis`: "Every other player: FLOPPER Jokic: wow it was so smart of him..." The model likely focused on the long structure and references to contact, refs, and specific plays. The actual function is sarcastic reaction to discourse around Jokic and officiating, not a stable basketball argument.
2. True `reaction`, predicted `hot_take`: "Idk why they'd blame the refs when Jokic's teammates are right there to crucify lol." This is a quick joke/complaint about blame after a game, but the model treated the negative evaluative language as a claim. The boundary is hard because r/nba reactions often include criticism without developing a durable argument.
3. True `hot_take`, predicted `analysis`: "Even folks who think Jokic should be MVP... don't typically see Shai as a 'fraud' MVP." This comment is more balanced and explanatory than many `hot_take` examples, so the model appears to reward tone and basketball specificity. Under my taxonomy, it still asserts a broad MVP-discourse position without enough supporting evidence to count as `analysis`.

Sample classifications:

| Comment excerpt | Predicted label | Confidence | Comment |
| --- | --- | ---: | --- |
| "The spacing is broken because they have two non-shooters..." | `analysis` | 0.389 | Correct; the comment gives a basketball reason about spacing and weak-side defense, so `analysis` is reasonable. |
| "This team is fake and will get exposed by the first real contender." | `analysis` | 0.358 | Incorrect; this is an unsupported broad claim and should be `hot_take`. |
| "What is this lineup lmao?" | `hot_take` | 0.360 | Incorrect; this is an immediate emotional reaction and should be `reaction`. |
| "Jokic is still the best player alive because every Denver half-court possession runs through his passing reads." | `analysis` | 0.359 | Reasonable but borderline; it gives a specific basketball reason for a broad claim. |
| "Refs are cooking again lol." | `hot_take` | 0.357 | Incorrect; this is a low-context officiating reaction and should be `reaction`. |

## Reflection

The intended concept was discourse function: whether a comment argues, asserts, or reacts. The fine-tuned model did not beat the zero-shot Groq baseline. Groq reached 0.719 accuracy, while DistilBERT reached 0.438 accuracy. This suggests that the fine-tuned model did not learn the intended boundaries robustly from 210 examples.

The clearest failure is `reaction`: recall and F1 were both 0.000. The model never correctly predicted the reaction class on the test set; it split reactions mostly into `analysis` or `hot_take`. The gap between intent and learned behavior is that I wanted the model to judge discourse function, but it mostly learned surface cues: basketball-specific wording and longer structure often became `analysis`, while negative or mocking language often became `hot_take`.

## Stretch Features

### Error Pattern Analysis

The main systematic error is directional: the fine-tuned model overpredicts `analysis` and `hot_take` while failing to recover `reaction`. In the confusion matrix, all 10 true `analysis` comments were predicted correctly, but all 11 true `reaction` comments were misclassified. Seven reactions were predicted as `hot_take`, which suggests that the model treated jokes, complaints, and sarcastic wording as evaluative claims rather than momentary responses.

The second pattern is sarcasm. Comments such as "Every other player: FLOPPER Jokic..." carry emotional or sarcastic framing, but the model struggles to decide whether they are `hot_take`, `reaction`, or `analysis`. This suggests the label boundary needs more training examples where sarcasm is explicitly separated from evidence-based reasoning.

### Deployed Interface

I added a simple local web interface in [scripts/takemeter_interface.py](scripts/takemeter_interface.py). It accepts a new r/nba comment, runs the saved fine-tuned model, and displays the predicted label, confidence, and probability breakdown.

Run it after fine-tuning and saving the model locally:

```bash
python3 scripts/takemeter_interface.py
```

Then open:

```text
http://127.0.0.1:7860
```

The model weights are not committed because they are large generated artifacts. The interface looks for `takemeter_distilbert_model/`, `takemeter-model/`, or a path passed with `--model-dir`.

## Spec Reflection

The planning spec helped by forcing the label boundary rules before annotation. In particular, the evidence rule for `analysis` vs. `hot_take` made the dataset more consistent than a vague good/bad taxonomy.

The implementation diverged from the original ideal because the first dataset pass used AI/rule-assisted draft labels instead of fully manual labels. This made the project faster to set up, but it also created a real risk of keyword bias, so I reviewed and corrected the labels against the taxonomy before treating the CSV as final.

## AI Usage

AI assistance was used in three specific ways:

1. Label design and planning: Codex helped draft the r/nba label taxonomy, edge-case rules, data collection plan, evaluation criteria, and [planning.md](planning.md). The key decision was to use discourse-function labels rather than subjective quality labels like "good" and "bad".
2. Dataset setup: Codex helped create a local PullPush collection workflow and generate draft labels in [data/takemeter_nba_labeled.csv](data/takemeter_nba_labeled.csv). I reviewed the draft labels against the taxonomy and kept difficult cases in the notes/documentation.
3. Training pipeline: Codex helped adapt the Colab workflow for this label map, including the DistilBERT training loop, Groq baseline call, metrics, confusion matrix export, and result JSON export.

## Demo Video Checklist

The 3-5 minute demo should show:

- Running the classifier on 3-5 comments with label and confidence visible. The submitted Colab notebook includes sample-classification cells for this.
- One correct prediction and why it is reasonable.
- One incorrect prediction and what boundary it missed.
- The evaluation table and confusion matrix in this README.
