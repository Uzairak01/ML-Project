# Comparative Analysis of Machine Learning Models for Customer Churn Prediction

**Author:** Uzair Ahmad Khan, Sayed Farasat Ali chisti. 
**Course:** Machine Learning Project.  
**Date:** 25 November 2026 for submission

## Abstract

Customer attrition is costly in retail banking because it reduces recurring revenue and requires expensive acquisition and onboarding of replacement customers. This study evaluates six classifiers for predicting bank-customer exit from 10,000 records containing demographic, account, product, activity, and transaction-related variables. A Dummy Classifier, Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and XGBoost were evaluated through a leakage-free 70/30 stratified split. Categorical fields were one-hot encoded, continuous fields were standardized, and transformations were fitted on training data only. Five-fold GridSearchCV and cost-sensitive weighting were used for the two tree ensembles. Tuned XGBoost achieved the highest churn recall (75.1%), while tuned Random Forest achieved the strongest accuracy (84.5%), precision (61.1%), and F1-score (0.64). Product count, active-member status, age, and German geography were the leading XGBoost predictors. These results show that customer data can support proactive retention, but model selection should reflect the bank’s relative cost of missed churners and unnecessary customer contact.

**Index Terms—** customer churn, retail banking, Random Forest, XGBoost, class imbalance.

## I. Introduction

Customer retention is a strategic banking problem. A departing customer may take deposits, credit use, card spending, and future cross-selling opportunities to another provider. Since replacement requires marketing, onboarding, and compliance effort, early identification of elevated churn risk can support a targeted retention intervention. Churn prediction does not establish why an individual leaves; it estimates a useful risk signal from observed historical customer characteristics.

The task is challenging because churn signals are often subtle and the target is imbalanced. This dataset contains 7,963 retained customers and 2,037 churners, a churn rate of 20.37%. A classifier that predicts every customer as retained has 79.63% accuracy but finds no churners. Thus, accuracy must be interpreted with precision, recall, F1-score, ROC-AUC, and the confusion matrix. Prior customer-relationship-management research similarly stresses that evaluation should be aligned with retention decisions rather than a single aggregate score [1], [2].

This work asks: Q1, can demographic and transactional measures predict churn? Q2, which model provides the best recall/F1 balance? Q3, which attributes exert the strongest predictive influence? Its objectives are to create a reproducible leakage-free pipeline, benchmark a baseline plus five learning algorithms, and use imbalance-aware tuning to improve minority-class recall.

## II. Related Work

Early churn studies commonly used Logistic Regression and related statistical models because their coefficients are intelligible to managers [3]–[5]. These models are fast and transparent, but additive linear effects can miss interactions between customer activity, balance, products, and geography. KNN and a single decision tree provide useful non-linear baselines, although KNN is sensitive to feature scaling and a single tree is unstable.

Tree ensembles address these limitations. Random Forest averages randomized trees and reduces the variance of an individual tree [6]. Boosting sequentially concentrates on difficult errors [7], [8], and XGBoost provides a regularized, scalable implementation of gradient-boosted trees [9]. Recent churn comparisons in telecom continue to report competitive performance from Random Forest, boosting, and XGBoost, while emphasizing that results vary with data, preprocessing, and metrics [10]–[12].

Class imbalance is especially important for churn. Learners optimized for total error can prefer the dominant retained class [13]. SMOTE synthesizes minority examples [14], but must be applied inside training folds to avoid leakage. This study instead uses class weighting: balanced weights for Random Forest and scale_pos_weight for XGBoost. Explainability is also necessary for accountable retention action. SHAP [15] and model-diagnostic tools such as DALEX [16] allow prediction explanations; this project uses global XGBoost importance as an initial, non-causal attribution. Recent banking and explainable-churn work emphasizes reproducibility, correct categorical encoding, and transparent interpretation [17], [18].

## III. Methodology

### A. Data and preprocessing

The Kaggle Bank Customer Churn dataset has 10,000 instances, ten predictors, and binary target Exited. The health check found zero missing values and zero duplicate rows. RowNumber, CustomerId, and Surname were removed because they are identifiers rather than transferable behavioural predictors. Raw-value EDA was completed before transformation. The included [target distribution](EDA_Target_Distribution.png), [categorical EDA](EDA_Categorical_Features.png), [numerical EDA](EDA_Numerical_Distributions.png), and [correlation matrix](EDA_Correlation_Matrix.png) preserve business-readable units.

The data were separated into predictors X and target y and split with test_size=0.30, random_state=42, and stratify=y. This produced 7,000 training and 3,000 test cases, including 2,389 retained and 611 churned test customers. StandardScaler was applied to CreditScore, Age, Tenure, Balance, NumOfProducts, and EstimatedSalary:

$$z_j=(x_j-\mu_j)/\sigma_j.$$

Geography and Gender were one-hot encoded with drop='first', avoiding the dummy-variable trap. HasCrCard and IsActiveMember passed through unchanged. Crucially, the transformer was fitted only on X_train, then used to transform X_test. This prevents test information from influencing category definitions, means, or standard deviations [19].

### B. Models and tuning

The Dummy Classifier always selects the majority outcome and provides the no-skill reference. Logistic Regression estimates churn probability from linear log-odds. KNN classifies from nearby standardized observations. A Decision Tree learns threshold rules. Random Forest averages many randomized decision trees [6]. XGBoost builds regularized trees sequentially [9].

All six models were benchmarked on the same held-out test set. GridSearchCV used five training folds and F1 selection. Random Forest searched n_estimators {100, 200}, max_depth {6, 12, 18}, and min_samples_split {2, 5}, with class_weight='balanced'. XGBoost searched n_estimators {100, 200}, max_depth {3, 5, 7}, and learning_rate {0.03, 0.10}, with scale_pos_weight equal to the training negative-to-positive ratio. Five-fold cross-validation reduces reliance on a single validation split [20]. The implementation uses pandas [21], scikit-learn [19], and XGBoost.

## IV. Evaluation Metrics

With true positives TP, false positives FP, true negatives TN, and false negatives FN,

$$Accuracy=(TP+TN)/(TP+TN+FP+FN),$$
$$Precision=TP/(TP+FP),\quad Recall=TP/(TP+FN),$$
$$F1=2(Precision\times Recall)/(Precision+Recall).$$

Accuracy measures all correct predictions. Precision measures the credibility of a churn alert. Recall measures the share of actual churners found, and F1 balances precision and recall. ROC-AUC summarizes discrimination over probability thresholds [22]. Recall is prioritized here because FN represents a customer who exits without an opportunity for intervention. Nevertheless, a recall-only model could contact nearly everyone; precision and confusion-matrix counts therefore remain essential.

## V. Results and Discussion

**TABLE I. Initial held-out-test comparison**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (Dummy) | 0.7963 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.8130 | 0.6316 | 0.1964 | 0.2996 | 0.7881 |
| K-Nearest Neighbors | 0.8447 | 0.6845 | 0.4403 | 0.5359 | 0.8069 |
| Decision Tree | 0.7917 | 0.4888 | 0.4992 | 0.4939 | 0.6828 |
| Random Forest (Default) | 0.8683 | 0.7842 | 0.4877 | 0.6014 | 0.8652 |
| XGBoost (Default) | 0.8637 | 0.7371 | 0.5139 | 0.6056 | 0.8487 |

The Dummy result proves why accuracy is insufficient: 0.7963 accuracy accompanies zero recall. Logistic Regression improves ranking but finds only 19.64% of churners. KNN improves recall to 44.03%; the single tree detects about half of churners but has weak AUC. Default Random Forest has the strongest initial accuracy and AUC, whereas default XGBoost has the strongest initial recall and F1. The [comparison chart](Model_Performance_Comparison.png) shows this metric trade-off.

Tuned Random Forest yielded TN=2132, FP=257, FN=207, and TP=404: 84.5% accuracy, 61.1% precision, 66.1% recall, and F1=0.64. Tuned XGBoost yielded TN=1978, FP=411, FN=152, and TP=459: 81.2% accuracy, 52.8% precision, 75.1% recall, and F1=0.62. XGBoost captures 55 additional churners but creates 208 additional false-positive alerts. It is appropriate when a missed churner costs more than low-cost outreach; Random Forest is preferable where retention capacity is constrained.

The [master evaluation figure](Model_Evaluation_Plots.png) contains both confusion matrices, ROC curves, and XGBoost feature importance. NumOfProducts (about 26%), IsActiveMember (about 19%), Age (about 16%), and Geography_Germany (about 10%) dominate the importance ranking. These are predictive associations, not causal effects. A production system should add SHAP analysis, calibration, fairness checks, and an explicit intervention-cost threshold.

## VI. Conclusion and Future Work

Q1 is answered affirmatively: demographic, product, account, and activity data provide meaningful churn discrimination, with initial ROC-AUC above 0.84 for both ensembles and 0.8652 for Random Forest. Q2 has a policy-dependent answer: tuned XGBoost is best for recall at 75.1%, while tuned Random Forest is best for accuracy, precision, and F1. Q3 is answered by the importance plot: product density, active status, age, and geography are the dominant predictive attributes.

The study is limited by a static public tabular dataset and binary exit label. It lacks time-stamped transactions, complaints, digital engagement, and causal intervention outcomes. Future work should use time-aware validation, transaction sequences with LSTM or Transformer models, SHAP dashboards, calibrated probabilities, and threshold optimization against a bank-specific cost matrix. Controlled retention experiments are needed to show that a risk score reduces churn rather than merely predicts it.

## References

[1] J. Hadden, A. Tiwari, R. Roy, and D. Ruta, “Computer assisted customer churn management,” *Computers & Operations Research*, vol. 34, no. 10, pp. 2902–2917, 2007, doi: 10.1016/j.cor.2005.11.007.  
[2] S. A. Neslin *et al.*, “Defection detection,” *Journal of Marketing Research*, vol. 43, no. 2, pp. 204–211, 2006, doi: 10.1509/jmkr.43.2.204.  
[3] K. Coussement and D. Van den Poel, “Churn prediction in subscription services,” *Expert Systems with Applications*, vol. 34, no. 1, pp. 313–327, 2008, doi: 10.1016/j.eswa.2006.09.038.  
[4] B. C. M. Lemmens and C. Croux, “Bagging and boosting classification trees to predict churn,” *Journal of Marketing Research*, vol. 43, no. 2, pp. 276–286, 2006, doi: 10.1509/jmkr.43.2.276.  
[5] W. Verbeke *et al.*, “Building comprehensible customer churn prediction models,” *Expert Systems with Applications*, vol. 38, no. 3, pp. 2354–2364, 2011, doi: 10.1016/j.eswa.2010.08.023.  
[6] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.  
[7] Y. Freund and R. E. Schapire, “A decision-theoretic generalization of on-line learning,” *J. Computer and System Sciences*, vol. 55, no. 1, pp. 119–139, 1997.  
[8] J. H. Friedman, “Greedy function approximation,” *The Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.  
[9] T. Chen and C. Guestrin, “XGBoost,” in *Proc. KDD*, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.  
[10] R. Krishna *et al.*, “Application of machine learning techniques for churn prediction,” *Results in Engineering*, vol. 24, Art. 103165, 2024, doi: 10.1016/j.rineng.2024.103165.  
[11] K. C. Mouli *et al.*, “An analysis on classification models for customer churn prediction,” *Cogent Engineering*, vol. 11, Art. 2378877, 2024, doi: 10.1080/23311916.2024.2378877.  
[12] M. A. S. and P. M. S., “Explaining customer churn prediction in telecom,” *Machine Learning with Applications*, vol. 17, Art. 100567, 2024, doi: 10.1016/j.mlwa.2024.100567.  
[13] H. He and E. A. Garcia, “Learning from imbalanced data,” *IEEE TKDE*, vol. 21, no. 9, pp. 1263–1284, 2009, doi: 10.1109/TKDE.2008.239.  
[14] N. V. Chawla *et al.*, “SMOTE,” *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002, doi: 10.1613/jair.953.  
[15] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” *NeurIPS*, 2017.  
[16] P. Biecek, “DALEX,” *Journal of Machine Learning Research*, vol. 19, no. 84, pp. 1–5, 2018.  
[17] M. A. H. *et al.*, “Bank customer churn prediction using a reproducible and explainable ML framework,” *Proc. IEEE QPAIN*, 2026, doi: 10.1109/QPAIN69676.2026.11545595.  
[18] I. Zerine *et al.*, “Explainable churn prediction in telecom with tabular ML,” *Discover Artificial Intelligence*, vol. 6, Art. 263, 2026, doi: 10.1007/s44163-026-00983-0.  
[19] F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *JMLR*, vol. 12, pp. 2825–2830, 2011.  
[20] R. Kohavi, “A study of cross-validation and bootstrap,” in *Proc. IJCAI*, 1995, pp. 1137–1143.  
[21] W. McKinney, “Data structures for statistical computing in Python,” *Proc. SciPy*, pp. 56–61, 2010.  
[22] T. Fawcett, “An introduction to ROC analysis,” *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006, doi: 10.1016/j.patrec.2005.10.010.
