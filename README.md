#  Who Pays the Price?
## An Explainable & Responsible AI System for Credit Risk Decisions

---

##  Project Overview

Machine learning models are increasingly used to automate high-stakes decisions such as loan approvals. These systems are often evaluated using technical metrics like accuracy, but such metrics fail to answer a more important question:

> **When a model makes a mistake, who is harmed?**

This project builds an **explainable and fairness-aware AI system** to analyze credit risk decisions, focusing not just on prediction performance, but on the **human cost of model errors**.

Instead of optimizing accuracy alone, this project evaluates **who pays the price when automated decisions go wrong**.

---

##  Project Objectives

- Build a machine learning model to predict credit risk  
- Evaluate model performance **beyond accuracy**  
- Quantify **human harm caused by model errors**  
- Analyze whether harm is distributed evenly across groups  

---

##  Dataset

**German Credit Risk Dataset**
https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

- Source: UCI Machine Learning Repository  
- Records: 1,000 loan applicants  
- Features: Financial, employment, and demographic attributes  
- Target:
  - `1` → Good credit (low risk)
  - `2` → Bad credit (high risk)

The dataset includes an official **cost matrix**, making it well-suited for cost-sensitive and ethical analysis.

---

##  Defining Harm in AI Decisions

Not all model errors are equal.

| Error Type | Description | Impact |
|----------|------------|--------|
| False Positive | Approving a risky applicant | Financial loss |
| **False Negative** | Rejecting a safe applicant | **Human harm** |

This project focuses on **false negatives**, which represent unfair denial of opportunity to individuals.

---

##  Methodology

### 1️. Data Preparation
- Parsed raw academic data format  
- Assigned meaningful feature names  
- Encoded categorical variables  
- Created a binary risk target  
- Defined age-based groups to analyze fairness  

### 2️. Modeling
- Logistic Regression (interpretable baseline model)
- Compared unscaled vs scaled feature versions
- Selected model based on **error impact**, not just accuracy

### 3️. Evaluation Beyond Accuracy
- Confusion matrix analysis  
- Precision, recall, and F1-score  
- Explicit tracking of false positives and false negatives  

### 4️. Fairness & Harm Analysis
- Measured false rejection rates by age group  
- Identified which groups experienced higher unfair rejection  
- Demonstrated how accuracy alone can hide real-world harm  

---

##  Key Findings

- Accuracy alone is insufficient for evaluating AI systems  
- Technical improvements can unintentionally **increase human harm**  
- Model errors were not evenly distributed across age groups  
- Explainability is essential for trust, fairness, and accountability  

---


##  Recommendation

Based on findings, this project recommends:

- Using AI predictions as **decision support**, not final authority  
- Flagging low-confidence predictions for human review  
- Monitoring false rejection rates across demographic groups  
- Regular fairness audits during deployment  

---

##  Tools & Technologies

- **Python**
- **Pandas & NumPy** — data processing
- **Scikit-learn** — modeling & evaluation
- **Matplotlib & Seaborn** — visualization
- **VS Code**

---

