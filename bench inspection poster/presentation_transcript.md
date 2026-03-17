# Bench Inspection Poster Presentation Transcript

**Project:** Machine Learning Assisted Real-Time Plasma Diagnosis: Domain-Knowledge-Driven Feature Engineering Enable H2O2 Yield Prediction
**Student:** Mingjie Zhu (201850714)
**Supervisor:** Xin Tu | **Assessor:** Xue Yong
**Duration:** ~12 minutes

---

## 1. Opening (~30 seconds)

Good morning/afternoon. My name is Mingjie Zhu, and my project is about using machine learning to predict hydrogen peroxide yield from plasma reactions. The key idea is that instead of relying on purely data-driven methods, I use domain knowledge from plasma chemistry to build better input features for the models. My supervisor is Professor Xin Tu.

---

## 2. Introduction & Why It Matters (~2 minutes)

So let me start with some background.

As the climate change and global warming is a universal trend, and is urgent to be solved, carbon dioxide reduction is a profounded technology for the future.

In the research field of plamsa chemistry, Nanosecond pulsed CO2 bubble discharge is a new way to convert CO2 into useful chemicals like hydrogen and hydrogen peroxide. 

Here is a demonstration of the experiment setting. The chemistry experiment is conducted by research team in XJTU. They also opensourced the data collected, which is what I used in this project.

To understand what is happening inside the experiemnt, a tool called Optical Emission Spectroscopy, or OES is usually used. OES can measure the light emitted by different species in the plasma, and from that, we can learn about the plasma's condition without disturbing the reaction. It is non-intrusive and cost-effective.

The goal of my project is to combine OES data with the reaction's discharge parameters, and use machine learning models to predict how much H2O2 is produced. The dataset I used comes from the work of Gao at XJTU. The dataset includes 701 OES sampling points from this the discharge experiment.

I wanna emphasize that why my research is useful in industy.

This project allows you to predict product yield in real time just from the light spectrum. While in usual, you should stop the process, take samples and do titration.

So this OES based method saves time, reduces cost, and allows for real-time process control. This has applications in chemical manufacturing, environmental engineering, and plasma-based water treatment.

---

## 3. Methodology (~4 minutes)

My project follows 4 iterative phases. Each phase builds on the previous one.

### Phase 1: Baseline Modelling

In Phase 1, I started with the raw OES data. The OES spectrum has 701 wavelength points, so the data is high-dimensional. I used Principal Component Analysis, or PCA, to reduce the dimensions. PCA works by finding directions of maximum variance in the data using the covariance matrix.

I tested seven different models: Ridge regression, Partial Least Squares, Support Vector Regression, XGBoost, Random Forest, Multi-Layer Perceptron, and a 1D Convolutional Neural Network.

I also designed three input configurations:
- Config A uses only OES features after PCA, with 11 components.
- Config B uses only the 4 discharge parameters. voltage, current, frequency, pulse_width or etc.
- Config C combines both OES and discharge parameters together.

The important finding here is that in Phase 1, using discharge parameters alone already gave decent results (linear models have close to 90% of R2), but OES inputs performed not that well. This might because of that either data limitation / OES representability limitation / or the model settings them selves are limited.


### Phase 2: Hyperparameter Tuning

So in phase 2 I tried to change the setting of models, in specific, i tuned the hyperparameters of non-linear models (RF, MLP, CNN) using an automatic tuning framework called Optuna. It is based on baysian optimization / TPE

With Optuna, I ran over 100 trials per model-config pair to find the optimal network structures or some model parameters.

After tuning, R-squared values improved for all models. For example, MLP went from negative 1.13 to 0.33 for Config C, and reached 0.86 for Config B which almost reaches the baseline. CNN with Config C remained the best model that uses OES data. 

But overall, the OES features were still not contributing as much as I expected.

### Phase 3: Domain-Knowledge Feature Engineering

This is where the most astonishing discovery been founded.

Instead of letting PCA decide which OES features to use for model input, I manually selected 13 physically meaningful features based on plasma chemistry knowledge.

For example:
- I picked specific emission lines like the OH line at 309 nm, the  oxygen atomic line at 777 nm, and the hydrogen-beta line at 486 nm. 
These lines represent species that are directly related to H2O2 formation.
- I also calculated band integrals for certain wavelength ranges, like the OH band from 306 to 312 nm and the CO2 band from 398 to 412 nm. Becuase, when you measuring the OES using spectroscopy, the background noise, or any internal or external factor may cause the measurment to drift. i.e. moved horizontally. So if you take integral of a interval/band, it will eliminates these kind of error.
- Apart from that, I created ratio features, like OH over H-alpha, which indicates relative OH availability. And according to many literatures, these ratios are commonly used in plasma diagnostics.

After replacing PCA components with these hand-crafted features, the improvement was dramatic. For the Ridge model with Config C, R-squared jumped from negative 0.17 to 0.80. MLP also improved from 0.33 to 0.80. This proves that domain knowledge makes a huge difference.

### Phase 4: Interpretability and Feature Reduction

In the final phase, I analysed which features are most important that affects the output of model, and tried to simplify the model.

There are different methods to evaluate the features importance for each model. For example, in Ridge model, we use absolute standardized regression coefficients. In simple words, linear regression models are a polynomial formula, for each input varaible x, the importance is the coefficient ahead of it.

I also did bootstrap resampling to get confidence intervals for R-squared, and found that there is no statistically significant difference between Ridge Config C and MLP Config C. This means that a simple linear model performs just as well as the neural network.

I ran a permutation test to make sure the model is truly learning a real pattern. Because, in a small scale data setting, the model it prone to overfitting the data. Overfitting means that the model only remember the data shown to it, find a trivial solution for a given senario. Permutation test is a way to tell if model really learns the relationship between input X and output Y. The result gave a p-value less than 0.00005, which confirms the prediction is genuine.

Finally, I used backward elimination and category ablation to remove less important features one by one, to improve the generalization ability of model. The result is that the best model is a Ridge regression with just 3 OES ratio features plus the 4 discharge parameters -- only 7 features in total.

---

## 4. Results Summary (~2 minutes)

Let me walk you through the key results.

As you can see in the R-squared comparison chart on the poster, there is a clear improvement across phases. In Phase 1, most models with Config C performed poorly because PCA did not capture useful OES information. After tuning in Phase 2, some models improved but the gap between Config B and Config C was still large for most models.

The breakthrough came in Phase 3 with feature engineering. Ridge Config C R-squared went from negative 0.17 all the way to 0.80. This is a massive improvement just from changing how we represent the input data.

The feature importance heatmap shows that CO2 flow rate and its OES band integral are the two most influential predictors across all models. This makes physical sense because CO2 is the main reactant.

Looking at the final ranking table, the best model is Ridge with 3 OES ratios after category ablation, achieving R-squared of 0.920. The second best is also Ridge with 1 OES ratio, at 0.918. These simple models match or even beat complex neural networks like MLP and CNN, which need many more features and much more computation.

---

## 5. Conclusion (~1.5 minutes)

To summarise my project:

First, domain-knowledge-driven feature engineering is the decisive factor for accurate prediction. Simply throwing raw OES data into machine learning models does not work well. You need to understand the physics behind the data.

Second, replacing PCA with physically meaningful features dramatically improved model performance. 

Third, interpretability analysis showed that CO2 flow rate and its OES band integral are the most important predictors, which aligns with the chemistry of the reaction.

Fourth, a simple Ridge regression model with just 7 well-chosen features can match the performance of complex neural networks. This is very valuable for real-time industrial applications because simple models are faster, easier to deploy, and easier to understand.

The permutation test confirmed with very high confidence that the model captures a real relationship between the input features and H2O2 yield.

---

## 6. Future Work (~30 seconds)

If I had more time, I would explore applying this approach to larger datasets from different plasma systems to test how well it generalises. I would also investigate using the model for real-time process control, where the OES signal directly adjusts the discharge parameters to optimise yield.

---

## 7. Prepared Q&A Notes

### "Why is your project useful to society?"
Plasma technology can convert greenhouse gases like CO2 into useful chemicals. My project helps make this process smarter by enabling real-time prediction and monitoring, which brings us closer to practical industrial use of plasma-based green chemistry.

### "What is the state-of-the-art?"
Most existing studies use OES data with standard machine learning but do not carefully consider which spectral features are physically meaningful. My work shows that domain-knowledge-driven feature selection is much more effective than automated methods like PCA.

### "Why did you choose this project?"
I am interested in both machine learning and clean energy technology. This project combines both areas and has real practical value for the environment.

### "What was the most challenging part?"
The hardest part was Phase 3 -- understanding enough plasma chemistry to select the right OES features. I needed to read many papers to understand which emission lines correspond to which chemical species and why they matter for H2O2 production.

### "This is a software/simulation project -- how did you validate it?"
I used leave-one-out cross-validation since the dataset is relatively small with 701 samples. I also used permutation testing to statistically verify that the model's predictions are not due to chance. The p-value was less than 0.00005, which is very strong evidence. Bootstrap resampling gave confidence intervals for the R-squared values.

### "Could industry use this?"
Yes. Companies in plasma processing, water treatment, or chemical manufacturing could use this approach. By monitoring the OES signal in real time and feeding it into the model, they can predict and optimise product yield without stopping production. The model is simple enough to run on standard hardware.

### "What is the originality of your project?"
The main original contribution is showing that domain-knowledge-driven feature engineering outperforms purely data-driven approaches for OES-based prediction in plasma systems. Specifically, using 13 hand-crafted features beat 11 PCA components, and a simple Ridge model with just 7 features matched complex neural networks.

### "How do your results compare to the original specifications?"
The original goal was to use ML to predict H2O2 yield from OES data. Initially this seemed difficult because OES-only models performed poorly. But through the 4-phase approach, I achieved R-squared of 0.92, which exceeded my original expectations. The most significant change from the original plan was the shift from data-driven PCA to domain-knowledge-based feature engineering in Phase 3.

---

*Tip: Remember to point to the relevant figures and tables on the poster as you talk through each section. Speak clearly and not too fast. If you do not understand a question, it is okay to ask them to repeat it.*
