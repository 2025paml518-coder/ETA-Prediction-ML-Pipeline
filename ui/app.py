# Week 1 Data Explorer Dashboard
# Visualise the ETA-Prediction data engineering pipeline outputs

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title='ETA Prediction – Week 1 Explorer', page_icon='🗺️', layout='wide')

st.title('🗺️ ETA Prediction – Week 1 Data Explorer')
st.markdown('Explore synthetic NYC trip data, validation results, feature engineering, and train/val/test splits.')
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load all pipeline outputs."""
    try:
        raw = pd.read_parquet('data/raw/trips_raw.parquet')
        validated = pd.read_parquet('data/interim/trips_validated.parquet')
        quarantined = pd.read_parquet('data/quarantine/quarantined_trips.parquet')
        train = pd.read_parquet('data/interim/train.parquet')
        val = pd.read_parquet('data/interim/val.parquet')
        test = pd.read_parquet('data/interim/test.parquet')
        train_features = pd.read_parquet('data/processed/train_features.parquet')
        
        return {
            'raw': raw,
            'validated': validated,
            'quarantined': quarantined,
            'train': train,
            'val': val,
            'test': test,
            'train_features': train_features,
        }
    except FileNotFoundError as e:
        st.error(f'❌ Data not found: {e}. Run `dvc repro` first.')
        st.stop()

data = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: PIPELINE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(['📊 Overview', '🔍 Validation', '✨ Features', '📈 Distributions', '🎯 Splits'])

with tab1:
    st.header('Pipeline Summary')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric('Raw Trips', f"{len(data['raw']):,}")
    with col2:
        st.metric('Valid Trips', f"{len(data['validated']):,}")
    with col3:
        quarantine_pct = (len(data['quarantined']) / len(data['raw']) * 100)
        st.metric('Quarantined', f"{len(data['quarantined']):,} ({quarantine_pct:.1f}%)")
    with col4:
        st.metric('Engineered Features', len(data['train_features'].columns))
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader('Data Quality')
        validated_pct = (len(data['validated']) / len(data['raw']) * 100)
        fig = go.Figure(data=[go.Bar(
            x=['Validated', 'Quarantined'],
            y=[validated_pct, 100 - validated_pct],
            marker=dict(color=['#2ecc71', '#e74c3c'])
        )])
        fig.update_layout(title='Data Retention', yaxis_title='%', showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader('Split Distribution')
        split_data = {
            'Train': len(data['train']),
            'Val': len(data['val']),
            'Test': len(data['test']),
        }
        fig = px.pie(
            values=list(split_data.values()),
            names=list(split_data.keys()),
            title='Train/Val/Test Split',
            color_discrete_map={'Train': '#3498db', 'Val': '#f39c12', 'Test': '#9b59b6'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader('Trip Duration Range')
        duration_range = data['validated']['trip_duration_min'].describe()
        fig = go.Figure(data=[go.Box(y=data['validated']['trip_duration_min'], name='Duration (min)')])
        fig.update_layout(title='Duration Distribution', height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: VALIDATION RESULTS
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.header('Data Validation Summary')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Defect Breakdown')
        if len(data['quarantined']) > 0:
            reason_counts = data['quarantined']['quarantine_reason'].value_counts().head(10)
            fig = px.bar(
                x=reason_counts.values,
                y=reason_counts.index,
                orientation='h',
                title='Top Quarantine Reasons',
                labels={'x': 'Count', 'y': 'Reason'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No quarantined data found.')
    
    with col2:
        st.subheader('Validation Metrics')
        metrics = {
            'Total Rows': len(data['raw']),
            'Valid': len(data['validated']),
            'Quarantined': len(data['quarantined']),
            'Valid %': f"{(len(data['validated']) / len(data['raw']) * 100):.1f}%",
            'Quarantine %': f"{(len(data['quarantined']) / len(data['raw']) * 100):.1f}%",
        }
        for key, value in metrics.items():
            st.write(f'**{key}:** {value}')
    
    st.divider()
    
    st.subheader('Sample Quarantined Rows')
    if len(data['quarantined']) > 0:
        st.dataframe(
            data['quarantined'][['trip_id', 'quarantine_reason', 'pickup_datetime']].head(10),
            use_container_width=True
        )
    else:
        st.info('No quarantined data.')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ENGINEERED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header('Feature Engineering')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Feature Categories')
        features_df = data['train_features']
        
        categories = {
            'Geometry': ['haversine_km', 'manhattan_km', 'bearing_sin', 'bearing_cos'],
            'Time': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos'],
            'Calendar': ['is_weekend', 'is_rush_hour', 'is_night', 'is_holiday'],
            'Weather': ['traffic_index', 'temperature_c', 'precipitation_mm', 'wind_kph', 'weather_severity'],
            'Metadata': ['passenger_count', 'store_and_forward'],
            'Zone': ['pickup_zone_lat', 'pickup_zone_lon', 'dropoff_zone_lat', 'dropoff_zone_lon', 'same_zone'],
            'Priors': ['zone_hour_prior_min', 'route_speed_kmph_prior', 'zone_baseline_duration_min'],
        }
        
        cat_counts = {}
        for cat, cols in categories.items():
            count = sum(1 for col in cols if col in features_df.columns)
            if count > 0:
                cat_counts[cat] = count
        
        if cat_counts:
            fig = px.bar(
                x=list(cat_counts.keys()),
                y=list(cat_counts.values()),
                title='Features by Category',
                labels={'x': 'Category', 'y': 'Count'},
                color=list(cat_counts.keys()),
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader('Feature Statistics')
        st.write(f'**Total Features:** {len(features_df.columns)}')
        st.write(f'**Numeric Features:** {features_df.select_dtypes(include=["number"]).shape[1]}')
        st.write(f'**Null Values:** {features_df.isnull().sum().sum()}')
        
        st.divider()
        st.subheader('Top Features by Variance')
        # Select only numeric columns for variance calculation
        numeric_df = features_df.select_dtypes(include=['number'])
        variance = numeric_df.var().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=variance.values,
            y=variance.index,
            orientation='h',
            title='Features with Highest Variance',
            labels={'x': 'Variance', 'y': 'Feature'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.header('Feature Distributions (Train Set)')
    
    features_df = data['train_features']
    numeric_cols = features_df.select_dtypes(include=['number']).columns.tolist()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_feature = st.selectbox('Select Feature', numeric_cols[:10])
    
    with col2:
        if selected_feature:
            fig = px.histogram(
                features_df,
                x=selected_feature,
                nbins=50,
                title=f'Distribution of {selected_feature}',
                labels={selected_feature: 'Value', 'count': 'Frequency'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.subheader('Correlation Heatmap (Top 15 Features)')
    
    # Select only numeric columns for variance calculation
    numeric_df = features_df.select_dtypes(include=['number'])
    top_var_cols = numeric_df.var().sort_values(ascending=False).head(15).index.tolist()
    corr_matrix = features_df[top_var_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns))
    fig.update_layout(title='Feature Correlations', height=500, width=700)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: TRAIN/VAL/TEST SPLITS
# ─────────────────────────────────────────────────────────────────────────────

with tab5:
    st.header('Temporal Split Analysis')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Train Rows', f"{len(data['train']):,}")
        st.write(f"**%:** {len(data['train']) / len(data['validated']) * 100:.1f}%")
    
    with col2:
        st.metric('Val Rows', f"{len(data['val']):,}")
        st.write(f"**%:** {len(data['val']) / len(data['validated']) * 100:.1f}%")
    
    with col3:
        st.metric('Test Rows', f"{len(data['test']):,}")
        st.write(f"**%:** {len(data['test']) / len(data['validated']) * 100:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Trip Duration by Split')
        splits = []
        for name, df in [('Train', data['train']), ('Val', data['val']), ('Test', data['test'])]:
            for duration in df['trip_duration_min']:
                splits.append({'Split': name, 'Duration (min)': duration})
        
        splits_df = pd.DataFrame(splits)
        fig = px.box(splits_df, x='Split', y='Duration (min)', title='Duration Distribution by Split')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader('Temporal Coverage')
        
        train_dates = pd.to_datetime(data['train']['pickup_datetime'])
        val_dates = pd.to_datetime(data['val']['pickup_datetime'])
        test_dates = pd.to_datetime(data['test']['pickup_datetime'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_dates.dt.date, y=[1]*len(train_dates), mode='markers', name='Train', marker=dict(size=3)))
        fig.add_trace(go.Scatter(x=val_dates.dt.date, y=[2]*len(val_dates), mode='markers', name='Val', marker=dict(size=3)))
        fig.add_trace(go.Scatter(x=test_dates.dt.date, y=[3]*len(test_dates), mode='markers', name='Test', marker=dict(size=3)))
        
        fig.update_layout(
            title='Temporal Split (by Date)',
            xaxis_title='Date',
            yaxis_title='Split',
            height=400,
            yaxis=dict(tickvals=[1, 2, 3], ticktext=['Train', 'Val', 'Test'])
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    """
    **Week 1 Data Explorer** — ETA Prediction ML Pipeline  
    Generated by `src/data/` and `src/features/` pipeline stages.  
    Run `dvc repro` to update outputs. See `docs/design_decisions.md` for implementation details.
    """
)
