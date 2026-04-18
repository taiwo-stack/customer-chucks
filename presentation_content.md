# 🏗️ Engineering Deep Dive: Customer Churn & CLV Architecture

This version is the most comprehensive technical edition, covering system architecture, modular design, and the specific training regime used to optimize the predictive engines.

---

## Slide 1: System Architecture & Data Flow
**Title**: Modular System Architecture  
**Key Points**:
- **Directory Structure**:
    - `data/`: Raw ingestion vs. Processed Parquet/CSV.
    - `src/`: Core logic (Data Prep, Modeling, CLV Analysis).
    - `models/`: Versioned `.pkl` artifacts (Preprocessor + Estimators).
- **Component Interaction**: Decoupled ingestion and training allows for independent scaling of the data pipeline.
- **Persistence Layer**: `Joblib` serialization ensures consistency between training and real-time inference.

**Speaker Notes (40s)**:  
"Let’s start with the structural engineering. I’ve architected this as a modular Python package. We have a clear separation of concerns: the `data/` directory handles raw ingestion and processed states, while `src/` contains the core functional logic. 

The system relies on a persistence-first approach. By decoupling the preprocessing pipeline from the model estimators and serializing them as independent artifacts in the `models/` folder, I’ve eliminated the risk of training-serving skew. This modularity means I can update the modeling engine without rewriting the data preparation logic, making the entire system highly maintainable and production-ready."

---

## Slide 2: Data Engineering & Feature Synthesis
**Title**: Feature Engineering & Preprocessing Pipeline  
**Key Points**:
- **Preprocessing**: `ColumnTransformer` wrapping `StandardScaler` and `OrdinalEncoder`.
- **Feature Synthesis**:
    - `tenure_bucket`: 0-6m, 6-12m, 12-24m, 24m+ categories.
    - `services_count`: Vector summation of active customer services.
    - `internet_no_tech_support`: Binary interaction term for identifying high-friction cohorts.

**Speaker Notes (45s)**:  
"My data engineering layer focuses on signal extraction. I used a Scikit-learn `ColumnTransformer` to apply differentiated scaling to numeric and categorical features. 

Beyond standard cleaning, I synthesized features to capture business-specific churn drivers. I discretized tenure into strategic buckets to handle non-linear risk, and created a `services_count` feature as a proxy for customer 'stickiness'. Most importantly, I engineered an interaction term, `internet_no_tech_support`, specifically to isolate the high-risk cohort of Fiber Optic users who lack onboarding support—a key driver I identified during exploratory data analysis."

---

## Slide 3: Training Regime & Model Selection
**Title**: The Training & Optimization Regime  
**Key Points**:
- **Data Split**: 60/20/20 (Train/Val/Test) with Stratified Sampling.
- **Model Portfolio**:
    - `LogisticRegression`: $L_2$ penalty, `max_iter=1000`.
    - `RandomForest`: `max_depth=10`, `min_samples_leaf=4`.
    - `XGBoost`: `learning_rate=0.1`, `max_depth=5`.
- **Class Imbalance**: Dynamically calculated `scale_pos_weight` for XGBoost and `balanced` weights for LR/RF.

**Speaker Notes (50s)**:  
"For the training regime, I employed a Stratified 60/20/20 split to maintain class ratios across all sets. My model selection was driven by the trade-off between interpretability and non-linear capturing. 

I tuned hyperparameters to prevent overfitting: for Random Forest, I capped `max_depth` at 10 and set a minimum leaf size of 4. For XGBoost, I used a conservative learning rate of 0.1. Crucially, I handled the 1:3 churn imbalance at the algorithmic level. While Logistic Regression used a global 'balanced' weight, I dynamically calculated the `scale_pos_weight` for XGBoost based on the training set's majority-to-minority ratio, ensuring the booster focused its loss minimization on the positive churn class."

---

## Slide 4: Strategic Threshold Optimization
**Title**: Decision Boundary & Threshold Tuning  
**Key Points**:
- **Optimization Loop**: Automated search across 31 decision thresholds ($\tau \in [0.2, 0.5]$).
- **KPI**: Prioritizing **Recall $\ge$ 60%** for business intervention.
- **Evaluation**: Final benchmarking on the held-out Test set.

**Speaker Notes (45s)**:  
"One of the most critical parts of the training process is my **Threshold Optimization Loop**. I don't settle for the standard 0.5 decision boundary. Instead, the training script iterates through 31 thresholds on the Precision-Recall curve. 

It automatically selects the highest threshold that still clears our business requirement of **60% Recall**. This ensures we catch at least 3 out of every 4 potential churners. By moving the boundary to approximately 0.3 or 0.4 depending on the model, we deliberately trade off some precision to maximize our 'Save' engine's coverage of high-risk customers."

---

## Slide 5: Model Selection & Results
**Title**: Benchmarking & Performance Metrics  
**Key Points**:
- **Winner**: Logistic Regression achieved **80% Recall** on the test set.
- **XGBoost/RF**: ~74% Recall with better AUC-ROC consistency.
- **Metric**: High AUC-ROC (0.84) validates the pipeline's overall discriminative strength.

**Speaker Notes (40s)**:  
"The final results on our held-out test set confirm the pipeline's efficacy. While all models achieved high **AUC-ROC scores of 0.84**, the benchmarked Logistic Regression model actually performed the best for our specific 'Save' objective, hitting **80% Recall**. 

This highlights an important engineering takeaway: while gradient boosting is powerful, for datasets with clear linear drivers like contracts and tenure, a well-tuned Logistic Regression model can be a more sensitive and reliable instrument for risk detection."

---

## Slide 6: Explanability & Interpretability
**Title**: Interpretability Methodology  
**Key Points**:
- **Linear Interpretability**: Standardized $\beta$ coefficients for direct impact ranking.
- **Non-Linear Explanations**: **SHAP** (SHapley Additive exPlanations) for XGBoost and RF.
- **Insights**: Month-to-month contracts identified as the primary global churn driver.

**Speaker Notes (40s)**:  
"We also prioritized explainability. For the linear models, I used standardized coefficients to rank feature impact. For the tree ensembles, I integrated **SHAP values**. 

This allowed us to move beyond simple 'importance' scores and see the *direction* of impact. We confirmed that Month-to-Month contracts have the strongest positive correlation with churn. This interpretability isn't just for us; it’s what allows the stakeholders to trust the model’s predictions when they're deciding which customers to target with retention offers."

---

## Slide 7: CLV Strategy & Segmentation
**Title**: Business Logic: Churn to CLV  
**Key Points**:
- **CLV Formula**: `MonthlyCharges * ExpectedTenure (24m)`.
- **Segmentation**: `pd.qcut` quartiles (Low, Med, High, Premium).
- **Strategic Impact**: 5x higher revenue risk in the 'Premium' segment.

**Speaker Notes (45s)**:  
"Finally, we bridge the gap to business strategy by converting churn probability into **Customer Lifetime Value**. I implemented a CLV calculation based on a 24-month horizon and used quantile-based segmentation to group our base. 

My analysis revealed a critical business insight: our 'Premium' customers often carry the highest risk profile. Because their loss is five times more damaging than a 'Low' segment user, our dashboard prioritizes these individuals for immediate intervention. This turns the technical model into a high-ROI business tool."

---

## Slide 8: The Dashboard & Production Insights
**Title**: Real-time Inference & Executive UI  
**Key Points**:
- **Dashboard**: Streamlit-based interface for non-technical users.
- **Simulation**: Real-time 'What-If' analysis via serialized inference pipelines.
- **Transparency**: ROC/Confusion Matrix data surfaced for technical auditing.

**Speaker Notes (30s)**:  
"I’ve deployed the entire system as a Streamlit dashboard. It uses the serialized `joblib` artifacts to run real-time inference, allowing reps to run 'What-If' simulations on customer profiles. 

More importantly, it surfaces the ROC curves and confusion matrices directly in the UI. This provides a transparent 'Health Check' for the models, allowing technical stakeholders to audit performance in real-time as new data flows into the system."

---

## Slide 9: Conclusion & Engineering Roadmap
**Title**: The Roadmap to Data Maturity  
**Key Points**:
- **Current State**: Proactive, imbalanced-aware retention system.
- **Future Improvements**:
    - **Survival Analysis**: Transitioning binary churn to duration modeling.
    - **Drift Detection**: Monitoring for model/concept drift in production.
    - **A/B Testing**: Validating 'Save' campaign efficacy.

**Speaker Notes (30s)**:  
"In conclusion, I have moved the business from reactive reporting to proactive, value-based retention. I have a live pipeline that is sensitivity-optimized and technically transparent. 

My roadmap includes moving toward **Survival Analysis** to model *when* a customer will churn, not just *if*, and implementing automated drift detection to maintain model integrity over time. This project is the foundation for a state-of-the-art retention strategy. Thank you."
