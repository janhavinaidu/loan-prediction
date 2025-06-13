from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# Load explainer
explainer = ClassifierExplainer.from_file("loan_approval_model.joblib")

# Launch dashboard
ExplainerDashboard(
    explainer,
    title="Loan Approval - ML Explainer Dashboard",
    port=8050,             # Make sure this is set if you're deploying
    mode="external"        # For environments like Render or Colab
).run()
