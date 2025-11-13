
```markdown

## Quick start

Welcome to the Wunder Challenge! This guide will walk you through the first steps, from setting up your environment to making your first submission.

### 1. Get the starter pack

Use this shell oneliner:

macOS/Linux/Windows:
```
curl -L https://files.wundernn.io/wnn_starterpack.tar.gz | \
tar -xz && \
cd competition_package
```
or download the archive manually:

**STARTERPACK**  
• [wnn_starterpack.zip](https://files.wundernn.io/wnn_starterpack.zip)  
• [wnn_starterpack.tar.gz](https://files.wundernn.io/wnn_starterpack.tar.gz)  
• slow download? try [mirror-1](https://files-alternative.wundernn.io/wnn_starterpack.zip)/[mirror-2](https://storage.yandexcloud.net/files-alternative.wundernn.io/wnn_starterpack.zip)

#### What's inside

Inside the archive you'll find `competition_package` folder:

```
competition_package/
  ├── datasets
  │   └── train.parquet
  ├── examples
  │   └── simple
  │       ├── README.md
  │       └── solution.py
  ├── README.md
  └── utils.py
```
Here's some files worth noting:

• `datasets/train.parquet`: a training dataset to build and test your models locally.  
• `utils.py`: contains helper classes (like `DataPoint`) and the scoring function.  
• `examples/simple/solution.py`: a minimal working example to show the required submission format.

### 2. Set up your environment

We strongly recommend using a virtual environment to keep your project dependencies tidy and avoid conflicts.

macOS/Linux/Windows:
```
python -m venv env
source env/bin/activate
```
Now, any packages you install will be contained in this private environment.

### 3. Install dependencies

To work with the data and run the baseline solution, you'll need a few libraries. Install them using pip:

```
pip install pandas scikit-learn pyarrow numpy tqdm
```

### 4. Run the simple example

The competition package includes a simple example to help you understand the basic data processing loop.

1. Go to the example folder:
```
cd competition_package
cd examples/simple
```
2. Run the script:
```
python solution.py
```
The script will read the sample training data, generate predictions, and print the R² score to your screen. It's a great starting point for understanding the data format and the task.

You may notice a few things about the example solution that are critical. Your further submissions will need to meet these requirements as well:

**NOTE**

• The main file must be `solution.py`.  
• It must contain a class named `PredictionModel` with a `predict` method.  
• You can include other files in your solution too, like model weights, config files, or helper modules.  
• Make sure `solution.py` is at the root level of your submisson.

### 5. Prepare and submit solution

When your solution is ready, you need to package it as a `.zip` archive. The `solution.py` file must be at the root of the archive.

macOS/Linux/Windows:
Navigate to your solution's folder and run this command:
```
zip -r submission.zip .
```
Go to the [submit](https://wundernn.io/submit) page and send your solution for scoring.

> 🎉 ta-daa, you're awesome

### Now build your own solution

The simple example is just a starting point. To compete for the top places, you’ll need to train your own model.

**TIP**

Transformer models or recurrent architectures like LSTM, GRU, and Mamba-2 are well-suited for this kind of sequence modeling task.

**TIP**

**Create a validation set**. To test your model’s performance before submitting, you’ll need a validation set. Because all sequences in the data are independent and shuffled, you can easily create one. Just split the sequences by their `seq_ix`. For example, use 80% of the sequences for training and 20% for validation.



```

Если нужно сохранить как отдельный `.md` файл для github, просто скопируй этот текст и положи файл в репозиторий.

[1](https://wundernn.io/docs/quick_start)
