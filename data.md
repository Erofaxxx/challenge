# Wunder Fund RNN Challenge

---

## Data Overview

**Understanding the data is key to success in this challenge. Here’s a detailed breakdown of the dataset format, structure, and key properties.**

> To get the data you need to [sign up](https://wundernn.io/register) and go to the [quick start](https://wundernn.io/docs/quick_start) page.

---

### Data Format

The entire dataset is provided as a single table in a **Parquet file**. Each row represents a single market state at a specific point in time.

The table has **N + 3** columns:

- `seq_ix`: The ID of the sequence (integer; identifies which sequence the row belongs to).
- `step_in_seq`: An integer representing the step number within a sequence (from 0 to 999).
- `need_prediction`: Boolean (`True` or `False`). If `True`, you need to provide a prediction for the next step.
- **N feature columns**: The remaining `N` columns are anonymized numeric features that describe the market state.

---

### The Sequences

The data is organized into many independent sequences:

- **Sequence length:** Each sequence is exactly **1000 steps** long (from `step_in_seq` 0 to 999).
- **Independence:** Each sequence is completely independent of the others. The market history from one sequence does not carry over to the next. When `seq_ix` changes, you are starting fresh.
- **Warm-up period:** The first 100 steps (0–99) of every sequence are a *warm-up* period. You can use this data to build up your model's internal state (e.g., for an LSTM or Transformer), but you will not be scored on any predictions for these steps.
- **Scored predictions:** Your score is based on predictions for steps 101 to 1000 (inclusive). These steps have `need_prediction` set to `True`.

#### Data Ordering

- **Inside a sequence:** Rows are always ordered chronologically. `step_in_seq` 1 always comes after `step_in_seq` 0.
- **Between sequences:** The sequences themselves are shuffled. `seq_ix` 10 is not related to `seq_ix` 11. This property is very useful for creating a reliable validation set.

---

> **TIP: How to create a validation set**  
Because the sequences are independent and shuffled, you can create a robust local validation set by splitting the data by `seq_ix`.  
For example, you can train your model on the first 80% of the sequences and test its performance on the remaining 20%.

---

### Dataset Sizes

- **Training set:** The training data (`train.parquet`) contains approximately 500 sequences.
- **Test set:** The hidden test set used for scoring is roughly the same size as the training set.

---

### Evaluation Metric

We evaluate predictions using the **R²** (coefficient of determination) score. For each feature *i*, the score is calculated as follows:

**Feature Score Formula:**  
The final score is the average of the R² scores across all N features:

**Final Score Formula:**  
A higher R² score is better.

---


---

_Notifications_: alt+T
