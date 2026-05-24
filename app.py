import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              f1_score, precision_recall_curve,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🔍",
    layout="wide"
)

# ─── Load Model & Data ─────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('models/xgb_final.pkl')
    config = joblib.load('models/model_config.pkl')
    return model, config

@st.cache_data
def load_data():
    import os
    import json
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split

    csv_path = 'data/raw/creditcard.csv'

    if not os.path.exists(csv_path):
        os.makedirs('data/raw', exist_ok=True)

        # Write kaggle.json
        kaggle_dir = os.path.expanduser('~/.kaggle')
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(f'{kaggle_dir}/kaggle.json', 'w') as f:
            json.dump({
                "username": st.secrets["KAGGLE_USERNAME"],
                "key": st.secrets["KAGGLE_KEY"]
            }, f)
        os.chmod(f'{kaggle_dir}/kaggle.json', 0o600)

        # Use kaggle Python API directly
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path='data/raw/',
            unzip=True
        )

        if not os.path.exists(csv_path):
            raise FileNotFoundError("Download failed — CSV still not found")

    df = pd.read_csv(csv_path)

    scaler = RobustScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])
    df = df.drop(['Time', 'Amount'], axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y)

    return X_test, y_test.values, X.columns.tolist()

model, config  = load_model()
threshold      = config['threshold']
X_test_df, y_test, feature_names = load_data()
y_prob         = model.predict_proba(X_test_df)[:, 1]
y_pred         = (y_prob >= threshold).astype(int)

# ─── Header ────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection")
st.markdown("**XGBoost + SHAP Explainability | Trained on 284K+ real transactions**")

# ─── Top KPI Metrics ───────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Transactions", f"{len(y_test):,}")
col2.metric("Fraud Cases", f"{y_test.sum()}")
col3.metric("PR-AUC", f"{average_precision_score(y_test, y_prob):.4f}")
col4.metric("Fraud Caught", f"{y_pred[y_test==1].sum()}/{y_test.sum()}")
col5.metric("False Alarms", f"{y_pred[y_test==0].sum()}")

st.divider()

# ─── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "📈 Model Performance",
    "🧠 Explainability",
    "🚨 Live Fraud Detector"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — Dataset Overview
# ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Class distribution
        counts = pd.Series(y_test).value_counts()
        fig = px.pie(
            values=counts.values,
            names=['Legitimate', 'Fraud'],
            title='Transaction Class Distribution',
            color_discrete_map={
                'Legitimate': '#2ecc71', 'Fraud': '#e74c3c'},
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Amount distribution
        amount_col = 'Amount_scaled'
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=X_test_df[amount_col][y_test==0],
            name='Legitimate', nbinsx=50,
            marker_color='#2ecc71', opacity=0.7))
        fig.add_trace(go.Histogram(
            x=X_test_df[amount_col][y_test==1],
            name='Fraud', nbinsx=50,
            marker_color='#e74c3c', opacity=0.7))
        fig.update_layout(
            title='Transaction Amount Distribution (Scaled)',
            barmode='overlay',
            xaxis_title='Scaled Amount',
            yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    # Feature correlation bar chart
    test_with_label = X_test_df.copy()
    test_with_label['Class'] = y_test
    correlations = test_with_label.corr()['Class'].drop(
        'Class').sort_values()

    fig = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        title='Feature Correlation with Fraud (Test Set)',
        color=correlations.values,
        color_continuous_scale='RdYlGn_r',
        height=500
    )
    fig.update_layout(yaxis_title='Feature',
                      xaxis_title='Correlation')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual",
                        color="Count"),
            x=['Legitimate', 'Fraud'],
            y=['Legitimate', 'Fraud'],
            text_auto=True,
            color_continuous_scale='Blues',
            title=f'Confusion Matrix (threshold={threshold:.3f})'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Precision-Recall curve
        precisions, recalls, thresholds_pr = precision_recall_curve(
            y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recalls, y=precisions,
            mode='lines', name='PR Curve',
            line=dict(color='#3498db', width=2)))
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"PR-AUC = {average_precision_score(y_test, y_prob):.4f}",
            showarrow=False, font=dict(size=14))
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision')
        st.plotly_chart(fig, use_container_width=True)

    # Threshold slider — the killer feature
    st.subheader("🎚️ Threshold Tuning")
    st.markdown("Adjust threshold to see real-time precision/recall tradeoff:")

    selected_threshold = st.slider(
        "Decision Threshold", 0.0, 1.0,
        float(threshold), 0.01)

    y_pred_custom = (y_prob >= selected_threshold).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 Score",
              f"{f1_score(y_test, y_pred_custom):.4f}")
    c2.metric("Fraud Caught",
              f"{y_pred_custom[y_test==1].sum()}/{y_test.sum()}")
    c3.metric("False Alarms",
              f"{y_pred_custom[y_test==0].sum()}")
    c4.metric("Precision",
              f"{y_pred_custom[y_test==1].sum() / max(y_pred_custom.sum(),1):.4f}")

    # Threshold curve
    f1_scores_t = []
    thresh_range = np.arange(0.01, 1.0, 0.01)
    for t in thresh_range:
        yp = (y_prob >= t).astype(int)
        f1_scores_t.append(f1_score(y_test, yp, zero_division=0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresh_range, y=f1_scores_t,
        mode='lines', name='F1',
        line=dict(color='green', width=2)))
    fig.add_vline(
        x=selected_threshold, line_dash="dash",
        line_color="red",
        annotation_text=f"Selected: {selected_threshold:.2f}")
    fig.update_layout(
        title='F1 Score vs Threshold',
        xaxis_title='Threshold',
        yaxis_title='F1 Score')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — Explainability
# ══════════════════════════════════════════════════════════
with tab3:
    st.subheader("SHAP Explainability")

    st.info("SHAP values explain WHY the model makes each prediction.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Global Feature Importance**")
        if st.button("Generate Summary Plot"):
            with st.spinner("Computing SHAP values..."):
                explainer = shap.TreeExplainer(model)
                # Use sample for speed
                sample_idx = np.random.choice(
                    len(X_test_df), 500, replace=False)
                shap_sample = explainer.shap_values(
                    X_test_df.iloc[sample_idx])

                fig, ax = plt.subplots(figsize=(8, 6))
                shap.summary_plot(
                    shap_sample,
                    X_test_df.iloc[sample_idx],
                    feature_names=feature_names,
                    max_display=12,
                    show=False)
                st.pyplot(fig)
                plt.close()

    with col2:
        st.markdown("**Pre-computed SHAP Images**")
        import os
        if os.path.exists('models/shap_summary.png'):
            st.image('models/shap_summary.png',
                     caption='SHAP Summary Plot')
        if os.path.exists('models/shap_waterfall.png'):
            st.image('models/shap_waterfall.png',
                     caption='Waterfall — Most Confident Fraud Case')

    # Feature importance table
    st.subheader("Top Fraud Signals (from SHAP analysis)")
    shap_importance = pd.DataFrame({
        'Feature': ['V14', 'V7', 'V10', 'V4', 'V28'],
        'Fraud Cases in Top-3': ['90/98 (92%)', '65/98 (66%)',
                                  '58/98 (59%)', '26/98 (27%)',
                                  '11/98 (11%)'],
        'Direction': ['Negative → Fraud', 'Negative → Fraud',
                      'Negative → Fraud', 'Positive → Legit',
                      'Varies']
    })
    st.dataframe(shap_importance, use_container_width=True,
                 hide_index=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — Live Fraud Detector
# ══════════════════════════════════════════════════════════
with tab4:
    st.subheader("🚨 Live Fraud Detector")
    st.markdown("Adjust transaction features and get an instant fraud prediction with explanation.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Key Features (most influential)**")
        v14 = st.slider("V14", -20.0, 10.0, 0.0, 0.1,
                         help="Most important fraud signal")
        v7  = st.slider("V7",  -25.0, 10.0, 0.0, 0.1,
                         help="2nd most important signal")
        v10 = st.slider("V10", -25.0, 25.0, 0.0, 0.1)
        v4  = st.slider("V4",  -5.0,  15.0, 0.0, 0.1)
        v12 = st.slider("V12", -20.0, 10.0, 0.0, 0.1)
        amount = st.slider("Amount (scaled)", -0.5, 10.0, 0.0, 0.1)

    with col2:
        st.markdown("**Other Features**")
        v3  = st.slider("V3",  -45.0, 10.0, 0.0, 0.1)
        v11 = st.slider("V11", -5.0,  12.0, 0.0, 0.1)
        v17 = st.slider("V17", -25.0, 10.0, 0.0, 0.1)
        v16 = st.slider("V16", -15.0, 10.0, 0.0, 0.1)
        v2  = st.slider("V2",  -75.0, 20.0, 0.0, 0.1)


    if st.button("🔍 Predict", type="primary"):
        # Build input with all features set to 0 except specified
        input_data = pd.DataFrame(
            np.zeros((1, len(feature_names))),
            columns=feature_names)

        input_data['V14'] = v14
        input_data['V7']  = v7
        input_data['V10'] = v10
        input_data['V4']  = v4
        input_data['V12'] = v12
        input_data['V3']  = v3
        input_data['V11'] = v11
        input_data['V17'] = v17
        input_data['V16'] = v16
        input_data['V2']  = v2
        input_data['Amount_scaled'] = amount

        prob = model.predict_proba(input_data)[0, 1]
        prediction = int(prob >= threshold)

        st.divider()

        # Big result display
        if prediction == 1:
            st.error(f"## 🚨 FRAUD DETECTED")
            st.error(f"Fraud Probability: **{prob*100:.1f}%**")
        else:
            st.success(f"## ✅ LEGITIMATE")
            st.success(f"Fraud Probability: **{prob*100:.1f}%**")
            st.caption(f"Decision threshold: {threshold:.3f} — "
                    f"model only flags fraud when >99.8% confident")

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Fraud Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c" if prediction==1
                        else "#2ecc71"},
                'steps': [
                    {'range': [0, 30],   'color': '#d5f5e3'},
                    {'range': [30, 70],  'color': '#fdebd0'},
                    {'range': [70, 100], 'color': '#fadbd8'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        # SHAP waterfall for this prediction
        with st.spinner("Generating SHAP explanation..."):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_data)

            fig2, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=input_data.iloc[0],
                feature_names=feature_names),
                show=False)
            st.pyplot(fig2)
            plt.close()
            st.caption("Red bars push toward FRAUD, "
                       "blue bars push toward LEGITIMATE")