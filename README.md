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

Raw API cache files are ignored by git because they are temporary collection artifacts. The collection and draft-labeling script is [scripts/collect_pullpush_dataset.py](scripts/collect_pullpush_dataset.py).

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
4. Before final submission, these draft labels should be manually reviewed against the taxonomy. This matters because query terms and rules can bias the dataset toward obvious keywords.

Three difficult examples:

| Excerpt | Possible labels | Final label | Decision |
| --- | --- | --- | --- |
| "Dubs issue isn't just 'pressure'..." | `analysis`, `hot_take` | `hot_take` | It uses basketball language, but the comment is mostly a blunt player limitation claim without developed support. |
| "Yup. Brook is not completely unplayable..." | `analysis`, `hot_take` | `analysis` | The wording is strong, but the comment explains matchup context and why the Pacers are a bad fit for him. |
| "Overrated is a wild take..." | `hot_take`, `reaction` | `reaction` | It responds to another take in the moment and uses emoji-style reaction framing rather than building an argument. |

## Fine-Tuning Approach

The planned fine-tuned model is `distilbert-base-uncased`, trained as a three-class sequence classifier. The completed Colab notebook is [Copy_of_ai201_project3_takemeter_starter_clean.ipynb](Copy_of_ai201_project3_takemeter_starter_clean.ipynb). The standalone Colab-ready script [scripts/run_takemeter_colab.py](scripts/run_takemeter_colab.py) is included as a backup pipeline.

Key hyperparameters:

| Hyperparameter | Value | Reason |
| --- | ---: | --- |
| Epochs | 3 | Small dataset; more epochs would increase overfitting risk. |
| Learning rate | `2e-5` | Standard conservative learning rate for BERT-style fine-tuning. |
| Batch size | 16 | Fits comfortably on a free Colab T4 for short comments. |
| Max length | 192 tokens | Most r/nba comments are short or medium length; this preserves context without wasting memory. |

To run the notebook in Colab:

1. Upload [Copy_of_ai201_project3_takemeter_starter_clean.ipynb](Copy_of_ai201_project3_takemeter_starter_clean.ipynb).
2. Set Runtime -> Change runtime type -> T4 GPU.
3. Upload [data/takemeter_nba_labeled.csv](data/takemeter_nba_labeled.csv) when prompted, or place it at `/content/takemeter_nba_labeled.csv`.
4. Add `GROQ_API_KEY` in Colab Secrets before running Section 5.
5. Run all cells, then download `evaluation_results.json` and `confusion_matrix.png`.

Backup script command:

```bash
pip install -q transformers scikit-learn matplotlib groq pandas
python run_takemeter_colab.py
```

Upload both files to Colab:

- `takemeter_nba_labeled.csv`
- `run_takemeter_colab.py`

Set `GROQ_API_KEY` in Colab Secrets before running the baseline.

## Baseline

The zero-shot baseline uses Groq's `llama-3.3-70b-versatile` on the same locked test split as the fine-tuned model. The prompt is saved in [prompts/groq_baseline_prompt.md](prompts/groq_baseline_prompt.md). The model is instructed to output only one label: `analysis`, `hot_take`, or `reaction`.

Baseline results were collected in Colab using the same 32-example test split as the fine-tuned model. All 32 Groq responses were parseable.

## Evaluation Report

The Colab notebook wrote:

- `evaluation_results.json`
- `confusion_matrix.png`

Overall results:

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Groq zero-shot baseline | 0.719 | 0.710 |
| Fine-tuned DistilBERT | 0.562 | 0.497 |

Per-class metrics:

| Model | Label | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| Groq zero-shot | `analysis` | 0.800 | 0.800 | 0.800 |
| Groq zero-shot | `hot_take` | 0.830 | 0.450 | 0.590 |
| Groq zero-shot | `reaction` | 0.620 | 0.910 | 0.740 |
| Fine-tuned DistilBERT | `analysis` | 0.530 | 0.900 | 0.670 |
| Fine-tuned DistilBERT | `hot_take` | 0.500 | 0.090 | 0.150 |
| Fine-tuned DistilBERT | `reaction` | 0.620 | 0.730 | 0.670 |

Fine-tuned confusion matrix:

| True \ Predicted | `analysis` | `hot_take` | `reaction` |
| --- | ---: | ---: | ---: |
| `analysis` | 9 | 0 | 1 |
| `hot_take` | 6 | 1 | 4 |
| `reaction` | 2 | 1 | 8 |

Wrong prediction analysis:

1. True `hot_take`, predicted `analysis`: "A 76ers fan? You most know a lot about 'Fraud MVP's." This is a short insult with almost no evidence, but the model did not catch the hot-take signal. Because the wording is brief and lacks obvious basketball-stat keywords, it appears the model treated it as a generic claim rather than an unsupported evaluative take.
2. True `reaction`, predicted `analysis`: "Every other player: FLOPPER Jokic: wow it was so smart of him..." The model likely focused on the long structure and references to contact, refs, and specific plays. The actual function is sarcastic reaction to discourse around Jokic and officiating, not a stable basketball argument.
3. True `analysis`, predicted `reaction`: "They had no real 2nd level of their offense. The progression tree was something like..." This is a real structured breakdown, but the model assigned `reaction` with low confidence. The numbered style and long sentence may not have been enough because the small training set did not provide many examples of detailed schematic explanation.

Sample classifications:

| Comment excerpt | Predicted label | Confidence | Comment |
| --- | --- | ---: | --- |
| "A 76ers fan? You most know a lot about 'Fraud MVP's" | `analysis` | 0.350 | Incorrect; this should be `hot_take`, but the model missed the unsupported insult frame. |
| "Every other player: FLOPPER Jokic..." | `analysis` | 0.400 | Incorrect; the model mistook a sarcastic reaction for structured analysis. |
| "I love being a fraud and fake number one seed..." | `reaction` | 0.360 | Incorrect; sarcasm and first-person phrasing pushed it toward reaction even though the claim is a hot take. |
| "Even folks who think Jokic should be MVP..." | `analysis` | 0.430 | Incorrect under my label, but understandable because the comment is more balanced and explanatory than most `hot_take` examples. |
| "They had no real 2nd level of their offense..." | `reaction` | 0.350 | Incorrect; this was a detailed `analysis` comment, showing the model did not reliably learn longer strategic explanations. |

## Reflection

The intended concept was discourse function: whether a comment argues, asserts, or reacts. The fine-tuned model did not beat the zero-shot Groq baseline. Groq reached 0.719 accuracy, while DistilBERT reached 0.562 accuracy. This suggests that the fine-tuned model did not learn the intended boundaries robustly from 210 examples.

The clearest failure is `hot_take` recall: only 0.090. The model overpredicted `analysis` for hot takes that mentioned player names, MVP arguments, or basketball context, even when the comment was really an unsupported claim. It also confused sarcastic reactions with analysis when the reaction was long. The gap between intent and learned behavior is that I wanted the model to judge whether evidence genuinely supports a claim, but it mostly learned surface shape: longer basketball-sounding comments often became `analysis`, while emotional phrasing often became `reaction`.

## Stretch Features

### Error Pattern Analysis

The main systematic error is directional: the fine-tuned model overpredicts `analysis`, especially for `hot_take` examples. In the confusion matrix, 6 of 11 true `hot_take` examples were predicted as `analysis`, and only 1 of 11 true `hot_take` examples was correctly recalled. The failure pattern is not random; comments that mention MVP debates, player names, officiating details, or basketball context often look analytical to DistilBERT even when the comment is mostly unsupported evaluation.

The second pattern is sarcasm. Comments such as "I love being a fraud and fake number one seed..." or "Every other player: FLOPPER Jokic..." carry emotional or sarcastic framing, but the model struggles to decide whether they are `hot_take`, `reaction`, or `analysis`. This suggests the label boundary needs more training examples where sarcasm is explicitly separated from evidence-based reasoning.

### Deployed Interface

I added a local web interface in [scripts/takemeter_interface.py](scripts/takemeter_interface.py). It accepts a new r/nba comment, runs it through a saved fine-tuned model, and displays the predicted label, confidence, and full probability breakdown.

Run it after fine-tuning and saving a model locally:

```bash
python3 scripts/takemeter_interface.py
```

Then open:

```text
http://127.0.0.1:7860
```

The interface looks for `takemeter_distilbert_model/`, `takemeter-model/`, or a path passed with `--model-dir`.

## Spec Reflection

The planning spec helped by forcing the label boundary rules before annotation. In particular, the evidence rule for `analysis` vs. `hot_take` made the dataset more consistent than a vague good/bad taxonomy.

The implementation diverged from the original ideal because the first dataset pass used AI/rule-assisted draft labels instead of fully manual labels. This made the project faster to set up, but it also creates a real risk of keyword bias. The dataset should be manually reviewed before final submission.

## AI Usage

AI assistance was used in three specific ways:

1. Label design and planning: Codex helped draft the r/nba label taxonomy, edge-case rules, data collection plan, evaluation criteria, and [planning.md](planning.md). The key decision was to use discourse-function labels rather than subjective quality labels like "good" and "bad".
2. Dataset setup: Codex created [scripts/collect_pullpush_dataset.py](scripts/collect_pullpush_dataset.py), downloaded public PullPush results, and generated draft labels in [data/takemeter_nba_labeled.csv](data/takemeter_nba_labeled.csv). These labels are disclosed as AI/rule-assisted and should be reviewed before final submission.
3. Training pipeline: Codex wrote [scripts/run_takemeter_colab.py](scripts/run_takemeter_colab.py), including the DistilBERT training loop, Groq baseline call, metrics, confusion matrix export, and result JSON export.

## Demo Video Checklist

The 3-5 minute demo should show:

- Running the classifier on 3-5 comments with label and confidence visible. The easiest way is to use the local interface from [scripts/takemeter_interface.py](scripts/takemeter_interface.py).
- One correct prediction and why it is reasonable.
- One incorrect prediction and what boundary it missed.
- The evaluation table and confusion matrix in this README.
