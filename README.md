# Sentiment Classification
* An overview of this task and the presented models can be found [here](https://drive.google.com/file/d/1bvA7Ryw3u7pPzBs-DJ-Bz-wIflXK9vN_/view?usp=sharing)

* Change directory to 'Sentiment Analysis'

`%cd "/content/drive/MyDrive/Sentiment Analysis"`

* Install requirements

`!pip install tensorflow==1.10`

`!pip install transformers`

* [Download](https://drive.google.com/file/d/1Qow1sCYsMbG-sfwVWVsSyDM1eXwNNQYt/view?usp=sharing) the checkpoint for the SOTA model and place in `"models/sota/"`

* Run the SOTA model in

> test mode

`!python src/sent_analysis.py -model 'sota' -mode 'test'`

> inference mode

 `!python src/sent_analysis.py -model 'sota' -mode 'inference' -test_file 'data/sentiment_dataset_test.csv'`
 
> train mode

`!python src/sent_analysis.py -model 'sota' -mode 'train'`

* To repeat the 'train', 'test' or 'inference' modes on the `baseline model` simply change model choice to `baseline`

`!python src/sent_analysis.py -model 'baseline' -mode 'test'`

* See predictions and evaluations in `output` folder
