# Introduction

Modern language models do not normally process text directly as words or characters. Before text is given to the model, it is divided into smaller units called *tokens*. One common way of doing this is Byte Pair Encoding (BPE), which starts with small units and repeatedly joins pairs that occur frequently in the training data.

You may read about BPE in this [blog post](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html#bpe).

Tokenization can look like a simple preprocessing step, but it is also an important modeling choice. The tokenizer determines what the model sees as its basic input units, how long its input sequences are, how large its vocabulary is, and how many model parameters are devoted to representing that vocabulary.

These choices may affect different languages differently. English and Turkish both use the Latin alphabet, but Turkish has considerably more productive word morphology. Chinese presents another situation: individual written characters already carry considerably more information than alphabetic characters, and words are not normally separated by spaces. BPE itself does not know anything about words, morphology, or writing systems. It simply learns frequent sequences from data.

In this assignment, you will investigate these issues experimentally. You will train several tokenizers on multilingual data containing **English, Turkish, and Chinese**, use them to train small Transformer language models from scratch, and study the consequences of the different tokenization choices.

The goal is **not** to obtain the best possible language model. The goal is to design a sensible experiment, make justified choices, and use your results to reason about the relationship between tokenization and model behavior.

You will have considerable freedom in how you organize your code and conduct your experiments. You will be graded on whether you built a working experimental pipeline, made reasonable experimental choices, evaluated the models appropriately, and can explain the results you obtained.

Your implementation must be in Python using PyTorch, must run on `mltgpu` or `mltgpu-2`, and must make use of the GPUs for model training. Your report should explain what you did, why you did it, how to reproduce the main experiments, and what you learned from them. If you use AI tools, don't forget to include the required AI-use log.

The AI log should include information about what you've asked the AI about, why, and how its response was integrated into the project. It does not need to include specific timestamps, exact prompts (they can be paraphrased/grouped together) or transcripts of AI responses. It can have the form of a table, e.g.:

| Prompt(s) | Why? | Incorporated as... |
| :--- | :--- | :--- |
| I asked ChatGPT Edu how to solve this ValueError problem that I kept running into with my code while building the character-level tokenizer. | I couldn't figure out where the error was coming from. | I got a response that pointed me to the issue being in [XYZ]. I fixed that part of the code. |

The assignment is worth **30 points**. The due date is **2026-10-04**.

---

# The Data

You will use text in three languages:
- English
- Turkish
- Mandarin Chinese

The corpus is derived from Wikipedia data from the Leipzig Corpora Collection. The data has already been prepared, so you do not need to download or extract the original corpora.

The files are available on `mltgpu` and `mltgpu-2` in:
```
/srv/data/lt2326-h26/a1/
```

The directory has approximately the following structure:
```
train/
    en.txt
    tr.txt
    zh.txt

valid/
    en.txt
    tr.txt
    zh.txt

test/
    en.txt
    tr.txt
    zh.txt

tokenizer/
    balanced.txt
    imbalanced_10-1-1.txt

metadata.json
README.txt
```

Each language file contains one sentence per line.

The datasets contain approximately equal amounts of English, Turkish, and Chinese text, measured in **Unicode characters**. Exact statistics are available in `metadata.json`.

Use:
- `train/` to train tokenizers and language models;
- `valid/` to inspect training progress and make experimental decisions;
- `test/` only for final evaluation.

The validation and test sets must **not** be used to learn the tokenizer vocabulary or to train the language model.

You may use less than the full training data if computational requirements become excessive, but comparisons between models must use the same underlying data. Describe any subsampling in your report.

---

# Part 1: The Tokenizers (7 points)

Create at least three tokenization conditions:
- A character-level tokenizer;
- A BPE tokenizer with a relatively small vocabulary;
- A BPE tokenizer with a substantially larger vocabulary.

Reasonable BPE vocabulary sizes might be around 2,000 and 10,000 tokens, but you may choose different values. Explain your choice.

The BPE tokenizers must be trained on the **combined English, Turkish, and Chinese training data** and should therefore have one shared multilingual vocabulary.

You may use an existing implementation such as SentencePiece or the Hugging Face `tokenizers` library. You are not required to implement BPE from scratch.

Your character-level tokenizer should also be constructed using only the training data. Consider how your implementation will handle a character that appears in validation or test data but was not observed during training.

## Before You Run the Experiment

Write down a few predictions. For example:
- Which language do you expect to require the most tokens?
- How do you expect the larger BPE vocabulary to change the representation of each language?
- What kinds of units do you expect BPE to learn in English, Turkish, and Chinese?
- Which tokenizer do you expect to work best for language modeling?

Briefly explain your reasoning. Your predictions do not have to be correct.

## Analyze the Tokenizers

Using the **validation data**, report for each tokenizer and language:
- Vocabulary size;
- Total number of tokens needed to encode the validation set;
- Average number of tokens per sentence;
- Average number of Unicode characters per token.

Present the results clearly.

Also examine several example sentences from each language under the different tokenizers. Look for interesting patterns: whole words, word fragments, Turkish suffixes, multi-character Chinese expressions, or segmentations that seem surprising.

The important part is not simply describing what you see. Try to explain **why the statistics of the training data may have produced those units**.

---

# Part 2: The Language Models (11 points)

Train one small **decoder-only Transformer language model** for each tokenization condition.

A reasonable starting architecture is:
- 2 Transformer layers;
- Hidden dimension around 256;
- 4 attention heads;
- Feed-forward dimension around 1024;
- Dropout;
- Context length around 128 or 256 tokens.

You may change these values if necessary.

The important requirement is that the **Transformer architecture remains the same across your main tokenization conditions**.

You should therefore train (at least) three multilingual models:
```
character tokenizer  -> Transformer LM
small BPE tokenizer  -> Transformer LM
large BPE tokenizer  -> Transformer LM
```

Train all models from scratch. Do **not** use pretrained language models.

## Fair Comparisons

A central question in this assignment is what constitutes a fair comparison.

One tokenizer may represent exactly the same corpus using many more tokens than another. This means that:
- Training for the same number of epochs exposes models to the same underlying text, but not necessarily the same amount of computation;
- Training for the same number of tokens may expose models to different amounts of underlying text.

Choose a policy for controlling the amount of training. For example, you might control by:
- Number of optimization updates;
- Amount of original text processed;
- Complete passes through the corpus;
- Approximate computation.

There is no single required answer. What matters is that you choose a reasonable policy, apply it consistently, and explain **both why you chose it and one limitation of that choice**.

## Model Size

Vocabulary size also changes the number of model parameters.

If the embedding dimension is $d$ and vocabulary size is $V$, the input embedding matrix contains $V \times d$ parameters:

$$V \times d$$

Report for each model:
- Total number of parameters;
- Parameters in the input embeddings;
- Parameters in the output vocabulary layer.

You do **not** need to make all models the same size. Instead, treat parameter count as something you will need to consider when interpreting your results.

## Training

Record at least training and validation loss and plot appropriate learning curves.

Your report should briefly explain:
- The architecture you used;
- Your training-control policy;
- Important hyperparameters;
- Problems or failed choices you encountered;
- Any changes you made and why.

Save representative trained models so that your evaluation can be reproduced.

---

# Checkpoint Before Final Submission

Before the final assignment deadline, submit a short experimental checkpoint. The checkpoint is meant to document your expectations and early observations **before you have completed the final analysis**.

Include:
- The BPE vocabulary sizes you chose and why;
- Your initial hypotheses about how the three tokenizers will behave;
- The parameter counts of your three models;
- One English, one Turkish, and one Chinese sentence tokenized by all three tokenizers;
- Preliminary tokenizer statistics for the three languages;
- At least one preliminary training or validation curve;
- One result or observation that you find interesting so far;
- One question that you want your final experiments to help answer.

For example, you might observe that Chinese uses considerably fewer tokens than Turkish under one tokenizer and decide to investigate whether this difference is also reflected in language-model performance.

Your hypotheses do not need to be correct. The purpose of the checkpoint is to record what you expected and what you noticed at an intermediate stage of the experiment.

**Keep this checkpoint: you will return to it explicitly in the final analysis.** You will **not** receive feedback on the checkpoint at this stage; feedback will be provided only after the complete assignment has been submitted. The purpose of this checkpoint is to help us verify that you are making steady progress and developing your work independently.

**Checkpoint deadline: 2026-09-30**

---

# Part 3: Evaluation and Analysis (12 points)

Evaluate each model separately on:
- English test data;
- Turkish test data;
- Chinese test data.

The purpose of this part is to interpret your experiment, not simply to identify the model with the lowest loss.

Your analysis should build on the hypotheses, observations, and questions that you submitted in your checkpoint.

## Comparing Models

Ordinary token-level perplexity is **not directly comparable** between models using different tokenizers. You should therefore use a measure normalized against the underlying text.

One possibility is **bits per character (BPC)**:

$$\mathrm{BPC} = -\frac{1}{N_{\mathrm{characters}}} \sum_t \log_2 p(x_t)$$

You may use another appropriate measure, but explain why it allows comparison across tokenizers.

Report results overall and separately for each language.

Then return to your checkpoint. For at least **two of your original hypotheses or questions**:
1. State what you expected;
2. Report the relevant final results;
3. Explain whether the results support or contradict your expectation;
4. Discuss what you think may explain the result.

At least one of these analyses should connect **tokenization behavior** to **language-model behavior**.

For example, you might investigate whether a language that received shorter token sequences also achieved better BPC, or whether a larger BPE vocabulary helped one language more than another.

Do not try to explain every number in your results. Concentrate on the patterns that are most informative, surprising, or relevant to the questions you identified earlier.

## Who Gets the Vocabulary?

Return to the preliminary tokenizer observations from your checkpoint and investigate how the shared BPE vocabulary is distributed among the three languages.

Develop a method for measuring whether a token is primarily associated with English, Turkish, Chinese, or shared among languages.

For example:

$$p(L \mid t) = \frac{\mathrm{count}(t, L)}{\sum_{L'} \mathrm{count}(t, L')}$$

You may define your own reasonable criterion for deciding when a token is language-specific or shared.

Use this analysis to investigate a question raised by your checkpoint observations. For example:
- Why does one language produce longer token sequences?
- Does one language appear to receive more vocabulary capacity?
- How much do English and Turkish actually share?
- How does increasing vocabulary size change the distribution?

Report the relevant statistics and discuss a few informative examples.

## Relating Tokenization to Model Behavior

Choose **one pattern from your checkpoint** that you think is especially interesting.

Investigate it more carefully using your final results.

Bring together evidence from at least two of the following:
- Token counts;
- Characters per token;
- Vocabulary allocation;
- Parameter counts;
- Training curves;
- Validation or test BPC;
- Individual tokenization examples.

Explain what the combined evidence suggests.

If the final results contradict your initial interpretation, that is perfectly acceptable, and potentially more interesting! Explain what changed your mind.

## Error Analysis

Choose several test sentences that help you investigate one of the patterns identified above.

Show the relevant tokenizations and model scores and discuss what might explain the differences.

Useful examples might involve:
- Uncommon words or names;
- Turkish morphology;
- Compounds;
- Numbers or punctuation;
- Recurring Chinese character sequences.

You are not expected to prove that a particular tokenization *caused* a model result. Clearly distinguish between what your experiment shows and your interpretation of why it happened.

## Conclusion

End your report by returning once more to the main expectations you recorded in your checkpoint.

Briefly summarize:
- One expectation that was supported;
- One expectation that was not supported, or a result that surprised you;
- What you think your experiment shows about using a shared tokenizer for English, Turkish, and Chinese;
- Which tokenization approach you would choose for this multilingual setting and why.

Support your conclusions with evidence from your experiments rather than general claims about BPE or the languages.

---

# Submission

Upload a GitHub URL containing your code and upload your report in Canvas along with your AI log if you used AI tools. We recommend making regular commits to your repository as you work. Your commit history should reflect the development of your work over the course of the assignment and help demonstrate your progress and working process.

Your repository should make it reasonably clear how to:
- Train the tokenizers;
- Train the models;
- Evaluate saved models;
- Reproduce the main tables and figures.

Include trained models if they are reasonably small. If they are too large for GitHub, store **representative** models under `/scratch` and specify their locations in the report. **Do not store every checkpoint.**

Your report may use any reasonable format. It should contain enough information to understand and reproduce your main experiment. We recommend using the [ACL LaTeX template](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj), as this will help you become familiar with the format you will use for your final project report.

A strong report will not simply contain scores. It will explain the choices you made, identify important limitations or confounding factors, compare the results with your initial hypotheses, and use evidence from your own experiments to support your conclusions.

## After Submission

After submission, you may be asked to briefly demonstrate your code and discuss your experiments.

You should be able to:
- Explain the main components of your implementation;
- Show how one of your reported results was produced;
- Explain the reasoning behind the experimental choices described in your report;
- Discuss an example or result from your own experiments.

You are responsible for understanding all submitted code, experimental decisions, and conclusions, including material produced with the assistance of AI tools.