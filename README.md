# RetainAI
To develop a machine learning-based B2B SaaS MVP that identifies at-risk e-commerce customers before they churn, utilizing behavioral and transactional data to trigger automated retention strategies.

## RetainAI MVP (Mobile-First Edition)

*   **Project Name:** RetainAI (Customer Churn & Win-Back Prediction Engine)
*   **Objective:** To develop a machine learning-based B2B SaaS MVP that identifies at-risk e-commerce customers before they churn, utilizing behavioral and transactional data to trigger automated retention strategies.
*   **Core Data Source:** eCommerce Customer Behavior Dataset (25 features capturing engagement and transaction history).

### 1. Problem Statement & Solution Overview
E-commerce businesses lose significant revenue to silent customer churn. Traditional retention methods rely on reactive measures (e.g., emailing a customer *after* 90 days of inactivity), which often occur too late in the customer journey. 

**The Solution:** RetainAI acts as a proactive intelligence layer. By feeding the 25 behavioral features from the e-commerce dataset (such as session length, category views, and time since last purchase) into a predictive model, the system calculates a real-time "Flight Risk Score." This allows marketing teams to automate targeted incentives (like a 15% discount) strictly to users on the verge of churning, saving promotional margins while recovering lost revenue.

### 2. Target Audience & Value Proposition
*   **Target Users:** E-commerce store owners, CRM managers, and Retention Marketers (specifically on platforms like Shopify or WooCommerce).
*   **Value Proposition:** 
    *   **Cost Savings:** Stop offering blanket discounts to users who were going to buy anyway.
    *   **Revenue Recovery:** Win back at-risk users with pinpoint timing.
    *   **Actionable Insights:** Translate raw behavioral data into plain-English marketing directives using explainable AI.

### 3. Technical Architecture & Development Stack
The architecture is designed for rapid iteration, pairing robust Python data science frameworks with a modern, cross-platform mobile frontend.

*   **Development Environment:** Ubuntu 22.04 LTS utilizing Visual Studio Code. Code generation, UI component scaffolding, and debugging across both the Python ML pipeline and the React Native frontend can be heavily accelerated by indexing the project codebase with local open-source models via Ollama and Continue.
*   **Data Processing & ML:** Python (Pandas, Scikit-learn). Class imbalance in the churn dataset will be handled via SMOTE (Synthetic Minority Over-sampling Technique).
*   **Modeling:** XGBoost or LightGBM for the core classification task, chosen for their high performance on tabular behavioral data.
*   **Explainability:** SHAP (SHapley Additive exPlanations) to interpret model decisions and output human-readable reasoning for the risk scores.
*   **Backend & API:** FastAPI to serve the trained `.pkl` model as a high-performance REST endpoint.
*   **Frontend Application:** React Native paired with Tailwind CSS (via NativeWind). This creates a polished, modern, and responsive cross-platform application demo that gives the MVP a premium, mobile-accessible SaaS feel.

### 4. Core MVP Feature Set
1.  **Customer Risk Matrix:** A central mobile dashboard displaying active users sorted by their dynamic Flight Risk Score (0-100%), styled with clean Tailwind utility classes.
2.  **Behavioral Explainability Cards:** Tapping on a user profile expands a modal revealing the top three behavioral drivers influencing their score (e.g., *Driver 1: Cart abandonment last week; Driver 2: Session duration dropped by 40%*).
3.  **Threshold Automation Webhooks:** Configurable UI toggles that trigger API calls simulating a webhook payload to an email tool (like Klaviyo or Mailchimp) when a user crosses an 85% risk threshold.
4.  **Model Performance Monitor:** A dedicated admin tab showing the model's F1-Score, Recall, and Feature Importance graphs optimized for mobile viewing.

### 5. Implementation Roadmap

| Phase | Milestone | Deliverables | Estimated Timeline |
| :--- | :--- | :--- | :--- |
| **Phase 1: Data Strategy** | EDA & Preprocessing | Cleaned dataset, correlation matrices, handled null values, and feature engineering (e.g., calculating interaction frequency). | Week 1 |
| **Phase 2: ML Pipeline** | Model Training & Tuning | Train-test split, SMOTE application, baseline Logistic Regression, and tuned XGBoost model with SHAP integration. | Week 2 |
| **Phase 3: Backend API** | FastAPI Service | RESTful API endpoints (`/predict_churn`, `/get_user_explanations`) serving the serialized model. | Week 3 |
| **Phase 4: Frontend GUI** | React Native App | Modern, cross-platform UI demo built with React Native and Tailwind CSS, consuming the FastAPI endpoints for real-time risk scores. | Week 4 |
| **Phase 5: Refinement** | Testing & Polish | End-to-end local testing, API integration checks, UI responsiveness tuning, and documentation. | Week 5 |

### 6. Future Scalability (Post-MVP)
Once the MVP successfully demonstrates predictive capability using the static Kaggle dataset, the next iteration will focus on:
*   **Live Data Ingestion:** Building real-time webhooks to ingest live session data from Shopify or Google Analytics 4.
*   **A/B Testing Engine:** Automatically splitting at-risk users into control and test groups to measure the actual lift generated by the RetainAI discount triggers.