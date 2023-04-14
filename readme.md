<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Boston House Price Prediction
 
## __Table Of Content__
- (A) [__Brief__](#brief)
  - [__Project__](#project)
  - [__Data__](#data)
  - [__Demo__](#demo) -> [Live Demo](https://ertugruldemir-housepriceprediction.hf.space)
  - [__Study__](#problemgoal-and-solving-approach) -> [Colab](https://colab.research.google.com/drive/1MCdPhbyFcmEix-C6iueAUYzWA_snriyB)
  - [__Results__](#results)
- (B) [__Detailed__](#Details)
  - [__Abstract__](#abstract)
  - [__Explanation of the study__](#explanation-of-the-study)
    - [__(A) Dependencies__](#a-dependencies)
    - [__(B) Dataset__](#b-dataset)
    - [__(C) Pre-processing__](#c-pre-processing)
    - [__(D) Exploratory Data Analysis__](#d-exploratory-data-analysis)
    - [__(E) Modelling__](#e-modelling)
    - [__(F) Saving the project__](#f-saving-the-project)
    - [__(G) Deployment as web demo app__](#g-deployment-as-web-demo-app)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)

## __Brief__ 

### __Project__ 
- This is a __regression__ project that uses the  [__Boston House Price Data__](http://lib.stat.cmu.edu/datasets/boston) to __predict the House Prices__.
- The __goal__ is build a model that accurately __predicts the House Prices__  based on the features. 
- The performance of the model is evaluated using several __metrics__, including _MaxError_, _MeanAbsoluteError_, _MeanAbsolutePercentageError_, _MSE_, _RMSE_, _MAE_, _R2_, _ExplainedVariance_ and other imbalanced regression metrics.

#### __Overview__
- This project involves building a machine learning model to predict the sales price based on number of 14 features. All the featrues are numerical typed. The dataset contains 506 records. The models selected according to model tuning results, the progress optimized respectively the previous tune results. The project uses Python and several popular libraries such as Pandas, NumPy, Scikit-learn.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://ertugruldemir-housepriceprediction.hf.space" height="30"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1VPAKlg1ne7lSdJTMll7smQI1VoksC37Q"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/HousePricePrediction/blob/main/study.ipynb
"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1MCdPhbyFcmEix-C6iueAUYzWA_snriyB"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    - __predict The House Price__  based on features.
    - __Usage__: Set the feature values through sliding the radio buttons then use the button to predict.
- Embedded [Demo](https://ertugruldemir-housepriceprediction.hf.space) window from HuggingFace Space
    

<iframe
	src="https://ertugruldemir-housepriceprediction.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Data__
- The [__Boston House Price Data__](http://lib.stat.cmu.edu/datasets/boston) from kaggle platform.
- The dataset contains 14 features, all the features are numerical.
- The dataset contains the following features:


<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>



| Attribute | Description |
|-----------|-------------|
| CRIM      | per capita crime rate by town |
| ZN        | proportion of residential land zoned for lots over 25,000 sq.ft. |
| INDUS     | proportion of non-retail business acres per town |
| CHAS      | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) |
| NOX       | nitric oxides concentration (parts per 10 million) |
| RM        | average number of rooms per dwelling |
| AGE       | proportion of owner-occupied units built prior to 1940 |
| DIS       | weighted distances to five Boston employment centres |
| RAD       | index of accessibility to radial highways |
| TAX       | full-value property-tax rate per $10,000 |
| PTRATIO   | pupil-teacher ratio by town |
| B         | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town |
| LSTAT     | % lower status of the population |
| MEDV      | Median value of owner-occupied homes in $1000's |


</td></tr> </table>


<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>

|    | Column   | Non-Null Count | Dtype    |
|---:|:--------:|:--------------:|:--------:|
|  0 | crim     | 506 non-null   | float64  |
|  1 | zn       | 506 non-null   | float64  |
|  2 | indus    | 506 non-null   | float64  |
|  3 | chas     | 506 non-null   | int64    |
|  4 | nox      | 506 non-null   | float64  |
|  5 | rm       | 506 non-null   | float64  |
|  6 | age      | 506 non-null   | float64  |
|  7 | dis      | 506 non-null   | float64  |
|  8 | rad      | 506 non-null   | int64    |
|  9 | tax      | 506 non-null   | float64  |
| 10 | ptratio  | 506 non-null   | float64  |
| 11 | black    | 506 non-null   | float64  |
| 12 | lstat    | 506 non-null   | float64  |
| 13 | medv     | 506 non-null   | float64  |

</td><td>

<div style="flex: 50%; padding-left: 50px;">

|Column    | count | mean     | std      | min      | 25%      | 50%      | 75%      | max      |
|:--------:|:-------:|:----------:|----------:|:----------:|:----------:|:----------:|:----------:|----------:|
|crim    |506.0  |-0.138238 |0.435946  |-0.419782 |-0.410970 |-0.390667 |0.007397  |0.844129  |
|zn      |506.0  |-0.155859 |0.585588  |-0.487722 |-0.487722 |-0.487722 |0.048772  |1.121761  |
|indus   |506.0  |11.136779 |6.860353  |0.460000  |5.190000  |9.690000  |18.100000 |27.740000 |
|chas    |506.0  |0.069170  |0.253994  |0.000000  |0.000000  |0.000000  |0.000000  |1.000000  |
|nox     |506.0  |0.554695  |0.115878  |0.385000  |0.449000  |0.538000  |0.624000  |0.871000  |
|rm      |506.0  |6.284634  |0.702617  |3.561000  |5.885500  |6.208500  |6.623500  |8.780000  |
|age     |506.0  |68.574901 |28.148861 |2.900000  |45.025000 |77.500000 |94.075000 |100.000000|
|dis     |506.0  |3.795043  |2.105710  |1.129600  |2.100175  |3.207450  |5.188425  |12.126500 |
|rad     |506.0  |9.549407  |8.707259  |1.000000  |4.000000  |5.000000  |24.000000 |24.000000 |
|tax     |506.0  |0.000000  |1.000990  |-1.313990 |-0.767576 |-0.464673 |1.530926  |1.798194  |
|ptratio |506.0  |18.455534 |2.164946  |12.600000 |17.400000 |19.050000 |20.200000 |22.000000 |
|black   |506.0  |0.260510  |0.242373  |-0.252087 |0.205072  |0.381187  |0.433651  |0.441052  |
|lstat   |506.0  |12.653063 |7.141062  |1.730000  |6.950000  |11.360000 |16.955000 |37.970000 |
|medv    |506.0  |22.532806 |9.197104  |5.000000  |17.025000 |21.200000 |25.000000 |50.000000 |



</div>

</td></tr> </table>


<div style="text-align: center;">
    <img src="docs/images/target_dist.png" style="max-width: 100%; height: auto;">
</div>


#### Problem, Goal and Solving approach
- This is a __regression__ problem  that uses the a bank dataset [__Boston House Price Data__](http://lib.stat.cmu.edu/datasets/boston)  from [kaggle](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) to __predict the house prices__ based on 14 features.
- The __goal__ is to build a model that accurately __predict the house prices__ based on the features.
- __Solving approach__ is that using the supervised machine learning models (linear, non-linear, ensemly).

#### Study
The project aimed predict the house prices using the features. The study includes following chapters.
- __(A) Dependencies__: Installations and imports of the libraries.
- __(B) Dataset__: Downloading and loading the dataset.
- __(C) Pre-processing__: It includes data type casting, feature engineering, missing value handling, outlier handling.
- __(D) Exploratory Data Analysis__: Univariate, Bivariate, Multivariate anaylsises. Correlation and other relations. 
- __(E) Modelling__: Model tuning via GridSearch on Linear, Non-linear, Ensemble Models.  
- __(F) Saving the project__: Saving the project and demo studies.
- __(G) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

#### results
- The final model is __xgbr regression__ because of the results and less complexity.
<div style="flex: 50%; padding-left: 80px;">

|            | MaxError   | MeanAbsoluteError | MeanAbsolutePercentageError | MSE          | RMSE         | MAE          | R2          | ExplainedVariance |
|----------- |-----------|------------------|-----------------------------|-------------|-------------|-------------|-------------|-------------------|
| xgbr      | 10.5| 1.35         | 9.260301                   | 7.800392| 2.792918| 1.966667| 0.909766   | 0.914817          |


</div>


- Model tuning results are below.

<table>
<tr><th>Linear Model</th></tr>
<tc><td>

|          | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE  | RMSE | MAE       | R2       | ExplainedVariance |
| -------- | --------| -----------------| --------------------------- | ---- | ---- | --------- | -------- | ----------------- |
| lin_reg  | 0.775306| 3.272549          | 19.551373                   | 12.3 | 2.7  | 15.484242| 0.773833| 4.421693          |
| l1_reg   | 12.3    | 2.7               | 15.484242                   | 19.551373 | 4.421693 | 3.272549| 0.773833| 0.775306     |
| l2_reg   | 12.3    | 2.7               | 15.484242                   | 19.551373 | 4.421693 | 3.272549| 0.773833| 0.775306     |
| enet_reg | 12.3    | 2.7               | 15.484242                   | 19.551373 | 4.421693 | 3.272549| 0.773833| 0.775306     |


</td><td> </table>


<table>
<tr><th>Non-Linear Model</th></tr>
<tc><td>

| Model | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE | RMSE | MAE | R2 | ExplainedVariance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn_reg | 24.0 | 2.10 | 14.729846 | 30.463137 | 5.519342 | 3.464706 | 0.647607 | 0.662414 |
| svr_reg | 16.5 | 1.65 | 12.464594 | 13.441569 | 3.666274 | 2.550980 | 0.844510 | 0.851877 |
| dt_params | 13.5 | 2.20 | 13.343018 | 12.810196 | 3.579133 | 2.694118 | 0.851813 | 0.852981 |

</td><td> </table>


<table>
<tr><th>Ensemble Model</th></tr>
<tc><td>

| Model        | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE       | RMSE      | MAE       | R2        | ExplainedVariance |
|--------------|----------|-------------------|------------------------------|-----------|-----------|-----------|-----------|-------------------|
| bag_reg      | 11.5     | 1.50              | 10.273731                    | 8.372941  | 2.893603  | 2.135294  | 0.903143  | 0.906935          |
| rf_reg       | 10.5     | 1.50              | 10.086696                    | 8.096471  | 2.845430  | 2.094118  | 0.906341  | 0.908946          |
| gbr          | 10.0     | 1.50              | 10.155048                    | 9.614118  | 3.100664  | 2.141176  | 0.888785  | 0.894613          |
| xgbr         | 10.5     | 1.35              | 9.260301                     | 7.800392  | 2.792918  | 1.966667  | 0.909766  | 0.914817          |
| lgbm_reg     | 12.0     | 1.65              | 10.433434                    | 10.045490 | 3.169462  | 2.305882  | 0.883795  | 0.889463          |
| catboost_reg | 11.0     | 1.45              | 9.494598                     | 9.008235  | 3.001372  | 2.119608  | 0.895794  | 0.904287          |


</td><td> </table>


## Details

### Abstract
- [Boston House Price Dataset](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) is used to predict the house prices. The dataset has 506 records, 14 features which are numerical typed. The problem is supervised learning task as regression. The goal is predicting  a house price value  correctly through using supervised machine learning algorithms such as non-linear, ensemble and similar model.The study includes creating the environment, getting the data, preprocessing the data, exploring the data, modelling the data, saving the results, deployment as demo app. Training phase of the models implemented through cross validation and Grid Search model tuning approachs. Hyperparameter tuning implemented Greedy Greed Search approach which tunes a hyper param at once a time while iterating the sorted order according the importance of the hyperparams. Models are evaluated with cross validation methods using 5 split. Classification results collected and compared between the models. Selected the basic and more succesful model. Tuned __lgbm regression__ model has __1060.35__ RMSE , __752.01__ MAE, __0.6093__ R2, __0.6093__ Explained Variance, the other metrics are also found the results section. Created a demo at the demo app section and served on huggingface space.  

### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── component_configs.json
│   ├── examples.pkl
│   ├── requirements.txt
│   └── xgbr_model.sav
├── docs
│   └── images
├── env
│   ├── env_installation.md
│   └── requirements.txt
├── LICENSE
├── readme.md
└── study.ipynb
```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/component_configs.json :
    - It includes the web components to generate web page.
  - demo_app/examples.pkl
    - It includes example cases to run the demo.
  - demo_app/xgbr_model.sav:
    - The trained (Model Tuned) model as pickle (python object saving) format.
  - demo_app/requirements.txt
    - It includes the dependencies of the demo_app.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE.txt
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance.    


### Explanation of the Study
#### __(A) Dependencies__:
  -  There is a third-parth installation which is kaggle dataset api, just follow the study codes it will be handled. The libraries which already installed on the environment are enough. You can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
#### __(B) Dataset__: 
  - Downloading the [Boston House Price Dataset](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) via kaggle dataset api from kaggle platform. The dataset has 506  records. There are 14 features which are numerical typed. For more info such as histograms and etc... you can look the '(D) Exploratory Data Analysis' chapter.
#### __(C) Pre-processing__: 
  - The processes are below:
    - Preparing the dtypes such as casting the object type to categorical type.
    - Missing value processes: Finding the missing values but there was no missing values.
    - feature engineering processes: min-max scaling.
    - Outlier analysis processes: uses  both visual and IQR calculation apporachs. According to IQR approach, detected statistically significant outliers are handled using boundary value casting assignment method.

      <div style="text-align: center;">
          <img src="docs/images/feat_outliers.png" style="width: 600px; height: 150px;">
      </div>
 
#### __(D) Exploratory Data Analysis__:
  - Dataset Stats
<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>

|    | Column   | Non-Null Count | Dtype    |
|---:|:--------:|:--------------:|:--------:|
|  0 | crim     | 506 non-null   | float64  |
|  1 | zn       | 506 non-null   | float64  |
|  2 | indus    | 506 non-null   | float64  |
|  3 | chas     | 506 non-null   | int64    |
|  4 | nox      | 506 non-null   | float64  |
|  5 | rm       | 506 non-null   | float64  |
|  6 | age      | 506 non-null   | float64  |
|  7 | dis      | 506 non-null   | float64  |
|  8 | rad      | 506 non-null   | int64    |
|  9 | tax      | 506 non-null   | float64  |
| 10 | ptratio  | 506 non-null   | float64  |
| 11 | black    | 506 non-null   | float64  |
| 12 | lstat    | 506 non-null   | float64  |
| 13 | medv     | 506 non-null   | float64  |

</td><td>

<div style="flex: 50%; padding-left: 50px;">

|Column    | count | mean     | std      | min      | 25%      | 50%      | 75%      | max    |
|--------|-------|----------|----------|----------|----------|----------|----------|----------|
|crim    |506.0  |-0.138238 |0.435946  |-0.419782 |-0.410970 |-0.390667 |0.007397  |0.844129  |
|zn      |506.0  |-0.155859 |0.585588  |-0.487722 |-0.487722 |-0.487722 |0.048772  |1.121761  |
|indus   |506.0  |11.136779 |6.860353  |0.460000  |5.190000  |9.690000  |18.100000 |27.740000 |
|chas    |506.0  |0.069170  |0.253994  |0.000000  |0.000000  |0.000000  |0.000000  |1.000000  |
|nox     |506.0  |0.554695  |0.115878  |0.385000  |0.449000  |0.538000  |0.624000  |0.871000  |
|rm      |506.0  |6.284634  |0.702617  |3.561000  |5.885500  |6.208500  |6.623500  |8.780000  |
|age     |506.0  |68.574901 |28.148861 |2.900000  |45.025000 |77.500000 |94.075000 |100.000000|
|dis     |506.0  |3.795043  |2.105710  |1.129600  |2.100175  |3.207450  |5.188425  |12.126500 |
|rad     |506.0  |9.549407  |8.707259  |1.000000  |4.000000  |5.000000  |24.000000 |24.000000 |
|tax     |506.0  |0.000000  |1.000990  |-1.313990 |-0.767576 |-0.464673 |1.530926  |1.798194  |
|ptratio |506.0  |18.455534 |2.164946  |12.600000 |17.400000 |19.050000 |20.200000 |22.000000 |
|black   |506.0  |0.260510  |0.242373  |-0.252087 |0.205072  |0.381187  |0.433651  |0.441052  |
|lstat   |506.0  |12.653063 |7.141062  |1.730000  |6.950000  |11.360000 |16.955000 |37.970000 |
|medv    |506.0  |22.532806 |9.197104  |5.000000  |17.025000 |21.200000 |25.000000 |50.000000 |

</div>

</td></tr> </table>
  - Variable Analysis
    - Univariate analysis, 
      <div style="text-align: center;">
          <img src="docs/images/feat_dist.png" style="width: 400px; height: 200px;">
          <img src="docs/images/feat_violin.png" style="width: 400px; height: 200px;">
      </div>
    - Bivariate analysis
      <div style="text-align: center;">
          <img src="docs/images/feat_1d_scatter.png" style="width: 400px; height: 300px;">
          <img src="docs/images/1d_hue_scatter.png" style="width: 400px; height: 300px;">
          <img src="docs/images/feat_scatter_with_target.png" style="width: 400px; height: 300px;">
      </div>
    - Multivariate analysis.
      <div style="text-align: center;">
          <img src="docs/images/multi_1.png" style="width: 400px; height: 300px;"> 
      </div>
  - Other relations.
    <div style="display:flex; justify-content: center; align-items:center;">
      <div style="text-align: center;">
      <figure>
      <p>Correlation</p>
      <img src="docs/images/feats_corrs_heatmap.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
       <div style="text-align: center;">
      <figure>
      <p>Correlation between target</p>
      <img src="docs/images/feat_corrs_between_target.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
      <div style="text-align: center;">
      <figure>
      <p>Variance</p>
      <img src="docs/images/variance.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
      <div style="text-align: center;">
      <figure>
      <p>Covariance</p>
      <img src="docs/images/covariance.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
    </div>

#### __(E) Modelling__: 
  - Data Split
    - Splitting the dataset via  sklearn.model_selection.train_test_split (test_size = 0.2).
  - Util Functions
    - Greedy Step Tune
      - It is a custom tuning approach created by me. It tunes just a hyperparameter per step using through GridSerchCV. It assumes the params ordered by importance so it reduces the computation and time consumption.  
    - Model Tuner
      - It is an abstraction of the whole training process. It aims to reduce the code complexity. It includes the corss validation and GridSerachCV approachs to implement training process.
    - Learning Curve Plotter
      - Plots the learning curve of the already trained models to provide insight.
  - Linear Model Tuning Results _without balanciy process_
    - linear, l1, l2, enet regressions
    - Cross Validation Scores
      |          | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE  | RMSE | MAE       | R2       | ExplainedVariance |
      | -------- | --------| -----------------| --------------------------- | ---- | ---- | --------- | -------- | ----------------- |
      | lin_reg  | 0.775306| 3.272549          | 19.551373                   | 12.3 | 2.7  | 15.484242| 0.773833| 4.421693          |
      | l1_reg   | 12.3    | 2.7               | 15.484242                   | 19.551373 | 4.421693 | 3.272549| 0.773833| 0.775306     |
      | l2_reg   | 12.3    | 2.7               | 15.484242                   | 19.551373 | 4.421693 | 3.272549| 0.773833| 0.775306     |
      | enet_reg | 12.3    | 2.7               | 15.484242                   | 19.551373 | 4.421693 | 3.272549| 0.773833| 0.775306     |
    - Feature Importance
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/lin_reg_f_imp.png" style="width: 150px; height: 200px;">
          <img src="docs/images/lin_f_imp.png" style="width: 450px; height: 200px;">
      </div>
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/lin_reg_l_cur.png" style="width: 150px; height: 200px;">
          <img src="docs/images/lin_l_cur.png" style="width: 450px; height: 200px;">
      </div>
  - Non-Linear Models
    - Logistic Regression, Naive Bayes, K-Nearest Neighbors, Support Vector Machines, Decision Tree
    - Cross Validation Scores _without balanciy process_
      | Model | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE | RMSE | MAE | R2 | ExplainedVariance |
      | --- | --- | --- | --- | --- | --- | --- | --- | --- |
      | knn_reg | 24.0 | 2.10 | 14.729846 | 30.463137 | 5.519342 | 3.464706 | 0.647607 | 0.662414 |
      | svr_reg | 16.5 | 1.65 | 12.464594 | 13.441569 | 3.666274 | 2.550980 | 0.844510 | 0.851877 |
      | dt_params | 13.5 | 2.20 | 13.343018 | 12.810196 | 3.579133 | 2.694118 | 0.851813 | 0.852981 |

    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/non_lin_l_cur.png" style="width: 400px; height: 300px;">
      </div>


  - Ensemble Models
    - Random Forest, Gradient Boosting Machines, XGBoost, LightGBoost, CatBoost
    - Cross Validation Scores _without balanciy process_
      | Model        | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE       | RMSE      | MAE       | R2        | ExplainedVariance |
      |--------------|----------|-------------------|------------------------------|-----------|-----------|-----------|-----------|-------------------|
      | bag_reg      | 11.5     | 1.50              | 10.273731                    | 8.372941  | 2.893603  | 2.135294  | 0.903143  | 0.906935          |
      | rf_reg       | 10.5     | 1.50              | 10.086696                    | 8.096471  | 2.845430  | 2.094118  | 0.906341  | 0.908946          |
      | gbr          | 10.0     | 1.50              | 10.155048                    | 9.614118  | 3.100664  | 2.141176  | 0.888785  | 0.894613          |
      | xgbr         | 10.5     | 1.35              | 9.260301                     | 7.800392  | 2.792918  | 1.966667  | 0.909766  | 0.914817          |
      | lgbm_reg     | 12.0     | 1.65              | 10.433434                    | 10.045490 | 3.169462  | 2.305882  | 0.883795  | 0.889463          |
      | catboost_reg | 11.0     | 1.45              | 9.494598                     | 9.008235  | 3.001372  | 2.119608  | 0.895794  | 0.904287          |


    - Feature Importance
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/ensemble_f_imp.png" style="width: 800px; height: 200px;">

      </div>
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/ensemble_l_cur.png" style="width: 800px; height: 400px;">
      </div>

#### __(F) Saving the project__: 
  - Saving the project and demo studies.
    - trained model __xgbr_model.sav__ as pickle format.
#### __(G) Deployment as web demo app__: 
  - Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.
  - Desciption
    - Project goal is predicting the sales price based on four features.
    - Usage: Set the feature values through sliding the radio buttons and dropdown menu then use the button to predict.
  - Demo
    - The demo app in the demo_app folder as an individual project. All the requirements and dependencies are in there. You can run it anywhere if you install the requirements.txt.
    - You can find the live demo as huggingface space in this [demo link](https://ertugruldemir-housepriceprediction.hf.space) as full web page or you can also us the [embedded demo widget](#demo)  in this document.  
    
## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

