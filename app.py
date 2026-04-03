import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Water Filter Shop — Sales Forecasting Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 28px; font-weight: 700; color: #1e3a5f;
        padding: 10px 0 4px; border-bottom: 3px solid #3b82f6; margin-bottom: 18px;
    }
    .sub-header {
        font-size: 16px; font-weight: 600; color: #374151; margin: 18px 0 10px;
    }
    .metric-card {
        background: #f0f9ff; border-radius: 10px; padding: 14px 18px;
        border-left: 5px solid #3b82f6; margin-bottom: 10px;
    }
    .best-badge {
        background: #dcfce7; color: #166534; font-weight: 700;
        padding: 3px 10px; border-radius: 20px; font-size: 12px;
    }
    .insight-box {
        background: #fefce8; border-left: 4px solid #f59e0b;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0;
        font-size: 14px; color: #78350f;
    }
</style>
""", unsafe_allow_html=True)

# ── Load & prepare data ───────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('water_filter_shop_sales_Data_2025.csv')
    month_map = {
        'January':1,'February':2,'March':3,'April':4,
        'May':5,'June':6,'July':7,'August':8,
        'September':9,'October':10,'November':11,'December':12
    }
    df['month_num'] = df['month'].map(month_map)
    df['date']      = pd.to_datetime({'year':2025,'month':df['month_num'],'day':1})
    return df

@st.cache_data
def build_time_series(df):
    monthly = df.groupby('date')['Total_Sale'].sum().sort_index()
    weekly  = monthly.resample('W').mean().interpolate(method='linear')
    return monthly, weekly

@st.cache_data
def run_models(weekly_series):
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import pmdarima as pm

    train_size = int(len(weekly_series) * 0.8)
    train = weekly_series.iloc[:train_size]
    test  = weekly_series.iloc[train_size:]

    # ARIMA
    auto_m        = pm.auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
    arima_fit     = ARIMA(train, order=auto_m.order).fit()
    arima_pred    = arima_fit.forecast(steps=len(test))
    arima_mae     = mean_absolute_error(test, arima_pred)
    arima_rmse    = np.sqrt(mean_squared_error(test, arima_pred))
    arima_mape    = np.mean(np.abs((test.values - arima_pred.values) / test.values)) * 100

    # SARIMA
    sarima_fit    = ARIMA(train, order=(2,0,0), seasonal_order=(1,0,1,12)).fit()
    sarima_pred   = sarima_fit.forecast(steps=len(test))
    sarima_mae    = mean_absolute_error(test, sarima_pred)
    sarima_rmse   = np.sqrt(mean_squared_error(test, sarima_pred))
    sarima_mape   = np.mean(np.abs((test.values - sarima_pred.values) / test.values)) * 100

    # Prophet
    from prophet import Prophet
    train_p = pd.DataFrame({'ds': train.index, 'y': train.values})
    test_p  = pd.DataFrame({'ds': test.index,  'y': test.values})
    prophet = Prophet(changepoint_prior_scale=0.01, seasonality_prior_scale=0.01,
                      yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    prophet.fit(train_p)
    future_p      = prophet.make_future_dataframe(periods=len(test), freq='W')
    forecast_p    = prophet.predict(future_p)
    prophet_pred  = forecast_p['yhat'].tail(len(test)).values
    prophet_mae   = mean_absolute_error(test.values, prophet_pred)
    prophet_rmse  = np.sqrt(mean_squared_error(test.values, prophet_pred))
    prophet_mape  = np.mean(np.abs((test.values - prophet_pred) / test.values)) * 100

    # Future forecast (8 weeks)
    future_8w      = prophet.make_future_dataframe(periods=len(test)+8, freq='W')
    forecast_8w    = prophet.predict(future_8w)
    future_only    = forecast_8w.tail(8)

    return {
        'train': train, 'test': test,
        'arima_order': auto_m.order,
        'arima_pred': arima_pred, 'arima_mae': arima_mae, 'arima_rmse': arima_rmse, 'arima_mape': arima_mape,
        'sarima_pred': sarima_pred,'sarima_mae': sarima_mae,'sarima_rmse': sarima_rmse,'sarima_mape': sarima_mape,
        'prophet_pred': prophet_pred,'prophet_mae': prophet_mae,'prophet_rmse': prophet_rmse,'prophet_mape': prophet_mape,
        'future_only': future_only, 'prophet_model': prophet,
    }

# ── Load everything ───────────────────────────────────────────────
df               = load_data()
monthly, weekly  = build_time_series(df)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/824/824239.png", width=60)
    st.markdown("## 💧 Water Filter Shop")
    st.markdown("**Sales Forecasting Dashboard**")
    st.markdown("---")
    st.markdown("**Student ID:** 20221250")
    st.markdown("**Dataset:** 3,459 invoices")
    st.markdown("**Period:** Jan–Dec 2025")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview & EDA",
        "🤖 Model Comparison",
        "🔮 Forecast",
        "📈 Business Insights"
    ])
    st.markdown("---")
    st.markdown("**Best Model:** Prophet (Tuned)")
    st.markdown("**Best MAPE:** 3.72%")

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════
if page == "📊 Overview & EDA":

    st.markdown('<div class="main-header">📊 Sales Overview & Exploratory Analysis</div>', unsafe_allow_html=True)

    # KPI row
    monthly_vals = monthly.values
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Total Revenue 2025",    f"LKR {sum(monthly_vals)/1e6:.2f}M")
    c2.metric("📈 Peak Month",            "May 2025",         f"LKR {max(monthly_vals)/1e6:.2f}M")
    c3.metric("📉 Lowest Month",          "February 2025",    f"LKR {min(monthly_vals)/1e6:.2f}M")
    c4.metric("📊 Jan→Dec Growth",        "+9.2%",            "Overall 2025")

    st.markdown("---")

    # Monthly sales bar chart
    st.markdown('<div class="sub-header">Monthly Sales Performance (2025)</div>', unsafe_allow_html=True)
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    bar_colors   = ['#ef4444' if v==max(monthly_vals) else
                    '#3b82f6' if v==min(monthly_vals) else
                    '#93c5fd' for v in monthly_vals]

    fig_monthly = go.Figure(go.Bar(
        x=month_labels, y=monthly_vals,
        marker_color=bar_colors, marker_line_color='#1e3a5f', marker_line_width=1.2,
        text=[f'LKR {v/1e6:.2f}M' for v in monthly_vals],
        textposition='outside', textfont_size=11
    ))
    fig_monthly.add_hline(y=np.mean(monthly_vals), line_dash='dash',
                          line_color='orange', line_width=2,
                          annotation_text=f"Average: LKR {np.mean(monthly_vals)/1e6:.2f}M",
                          annotation_position="top right")
    fig_monthly.update_layout(
        title='Total Sales by Month — 2025',
        yaxis_title='Total Sales (LKR)', xaxis_title='Month',
        yaxis_tickformat=',.0f', height=420, plot_bgcolor='white',
        yaxis=dict(gridcolor='#e5e7eb')
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Two charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Sales by Product Type</div>', unsafe_allow_html=True)
        prod = df.groupby('Product_Type')['Total_Sale'].sum().sort_values(ascending=True)
        fig_prod = px.bar(x=prod.values, y=prod.index, orientation='h',
                          color=prod.values, color_continuous_scale='Blues',
                          labels={'x':'Total Sales (LKR)','y':'Product'},
                          text=[f'LKR {v/1e6:.1f}M' for v in prod.values])
        fig_prod.update_traces(textposition='outside')
        fig_prod.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                               plot_bgcolor='white', yaxis=dict(gridcolor='#e5e7eb'))
        st.plotly_chart(fig_prod, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Sales by Location</div>', unsafe_allow_html=True)
        loc = df.groupby('Location')['Total_Sale'].sum().sort_values(ascending=True)
        fig_loc = px.bar(x=loc.values, y=loc.index, orientation='h',
                         color=loc.values, color_continuous_scale='Purples',
                         labels={'x':'Total Sales (LKR)','y':'Location'},
                         text=[f'LKR {v/1e6:.1f}M' for v in loc.values])
        fig_loc.update_traces(textposition='outside')
        fig_loc.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                              plot_bgcolor='white')
        st.plotly_chart(fig_loc, use_container_width=True)

    # Weekly time series
    st.markdown('<div class="sub-header">Weekly Sales Time Series (Interpolated)</div>', unsafe_allow_html=True)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=weekly.index, y=weekly.values,
        mode='lines', name='Weekly Sales',
        line=dict(color='#3b82f6', width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.08)'
    ))
    fig_ts.update_layout(
        yaxis_title='Total Sales (LKR)', xaxis_title='Date',
        height=350, plot_bgcolor='white',
        yaxis=dict(gridcolor='#e5e7eb', tickformat=',.0f')
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Product × Month heatmap
    st.markdown('<div class="sub-header">Product × Month Sales Heatmap</div>', unsafe_allow_html=True)
    pms   = df.groupby(['Product_Type','month_num'])['Total_Sale'].sum().reset_index()
    pivot = pms.pivot(index='Product_Type', columns='month_num', values='Total_Sale')
    pivot.columns = month_labels
    fig_heat = px.imshow(pivot, color_continuous_scale='YlGnBu',
                         labels=dict(color='Sales (LKR)'),
                         title='Product-wise Monthly Sales Pattern')
    fig_heat.update_layout(height=300)
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":

    st.markdown('<div class="main-header">🤖 Time Series Model Comparison</div>', unsafe_allow_html=True)

    with st.spinner("Training ARIMA, SARIMA and Prophet models... (first load takes ~2 min)"):
        results = run_models(weekly)

    train, test = results['train'], results['test']

    # Model KPI row
    c1, c2, c3 = st.columns(3)
    c1.metric(f"ARIMA{results['arima_order']}",
              f"MAPE: {results['arima_mape']:.2f}%",
              f"Accuracy: {100-results['arima_mape']:.2f}%")
    c2.metric("SARIMA(2,0,0)(1,0,1,12)",
              f"MAPE: {results['sarima_mape']:.2f}%",
              f"Accuracy: {100-results['sarima_mape']:.2f}%")
    c3.metric("Prophet Tuned  ⭐ BEST",
              f"MAPE: {results['prophet_mape']:.2f}%",
              f"Accuracy: {100-results['prophet_mape']:.2f}%")

    st.markdown("---")

    # Metrics bar charts
    st.markdown('<div class="sub-header">Evaluation Metrics — All 3 Models</div>', unsafe_allow_html=True)

    model_names = [f"ARIMA{results['arima_order']}", "SARIMA\n(2,0,0)(1,0,1,12)", "Prophet\n(Tuned)"]
    colors_m    = ['#3b82f6', '#8b5cf6', '#10b981']

    fig_metrics = make_subplots(rows=1, cols=3,
                                subplot_titles=['MAE (LKR) — Lower is Better',
                                                'RMSE (LKR) — Lower is Better',
                                                'MAPE (%) — Lower is Better'])

    for col_i, (vals, name) in enumerate([
        ([results['arima_mae'], results['sarima_mae'], results['prophet_mae']], 'MAE'),
        ([results['arima_rmse'],results['sarima_rmse'],results['prophet_rmse']],'RMSE'),
        ([results['arima_mape'],results['sarima_mape'],results['prophet_mape']],'MAPE'),
    ], 1):
        fig_metrics.add_trace(
            go.Bar(x=model_names, y=vals, marker_color=colors_m,
                   marker_line_color='black', marker_line_width=1.2,
                   text=[f'{v:,.0f}' if col_i < 3 else f'{v:.2f}%' for v in vals],
                   textposition='outside', showlegend=False),
            row=1, col=col_i
        )

    fig_metrics.update_layout(height=420, plot_bgcolor='white')
    fig_metrics.update_yaxes(gridcolor='#e5e7eb')
    if col_i == 3:
        fig_metrics.add_hline(y=10, line_dash='dash', line_color='red',
                              annotation_text='10% threshold', row=1, col=3)
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Model selector for actual vs predicted
    st.markdown('<div class="sub-header">Actual vs Predicted — Test Period</div>', unsafe_allow_html=True)

    selected_model = st.selectbox("Select model to view:", [
        f"ARIMA{results['arima_order']}",
        "SARIMA(2,0,0)(1,0,1,12)",
        "Prophet (Tuned) ⭐ Best"
    ])

    pred_map = {
        f"ARIMA{results['arima_order']}": (results['arima_pred'],   '#3b82f6', results['arima_mape']),
        "SARIMA(2,0,0)(1,0,1,12)":        (results['sarima_pred'],  '#8b5cf6', results['sarima_mape']),
        "Prophet (Tuned) ⭐ Best":         (results['prophet_pred'], '#10b981', results['prophet_mape']),
    }
    pred_vals, pred_color, pred_mape = pred_map[selected_model]

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=train.index, y=train.values, name='Training Data',
        line=dict(color='#93c5fd', width=1.5), opacity=0.6))
    fig_avp.add_trace(go.Scatter(
        x=test.index, y=test.values, name='Actual Sales',
        line=dict(color='black', width=2.5), mode='lines+markers',
        marker=dict(size=7)))
    fig_avp.add_trace(go.Scatter(
        x=test.index, y=pred_vals, name=f'{selected_model} Predicted',
        line=dict(color=pred_color, width=2.5, dash='dash'), mode='lines+markers',
        marker=dict(symbol='square', size=7)))
    fig_avp.add_vline(x=test.index[0].timestamp() * 1000, line_dash='dot',
                      line_color='gray', annotation_text='Test Start')
    fig_avp.update_layout(
        title=f'{selected_model} — Actual vs Predicted  |  MAPE: {pred_mape:.2f}%  |  Accuracy: {100-pred_mape:.2f}%',
        yaxis_title='Total Sales (LKR)', xaxis_title='Date',
        height=420, plot_bgcolor='white', yaxis=dict(gridcolor='#e5e7eb', tickformat=',.0f'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    # Comparison table
    st.markdown('<div class="sub-header">Model Summary Table</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame({
        'Model'      : [f"ARIMA{results['arima_order']}", "SARIMA(2,0,0)(1,0,1,12)", "Prophet (Tuned) ⭐"],
        'MAE (LKR)'  : [f"{results['arima_mae']:,.0f}",  f"{results['sarima_mae']:,.0f}",  f"{results['prophet_mae']:,.0f}"],
        'RMSE (LKR)' : [f"{results['arima_rmse']:,.0f}", f"{results['sarima_rmse']:,.0f}", f"{results['prophet_rmse']:,.0f}"],
        'MAPE (%)'   : [f"{results['arima_mape']:.2f}%", f"{results['sarima_mape']:.2f}%", f"{results['prophet_mape']:.2f}%"],
        'Accuracy'   : [f"{100-results['arima_mape']:.2f}%", f"{100-results['sarima_mape']:.2f}%", f"{100-results['prophet_mape']:.2f}%"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — FORECAST
# ══════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":

    st.markdown('<div class="main-header">🔮 Sales Forecast — Prophet (Best Model)</div>', unsafe_allow_html=True)

    with st.spinner("Generating forecast..."):
        results = run_models(weekly)

    future_only = results['future_only']

    # Forecast weeks slider
    n_weeks = st.slider("Number of weeks to forecast:", min_value=4, max_value=16, value=8, step=1)

    # Reforecast with selected weeks
    from prophet import Prophet
    train = results['train']
    train_p = pd.DataFrame({'ds': train.index, 'y': train.values})
    prophet_f = Prophet(changepoint_prior_scale=0.01, seasonality_prior_scale=0.01,
                        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    prophet_f.fit(train_p)
    future_n   = prophet_f.make_future_dataframe(periods=len(results['test'])+n_weeks, freq='W')
    forecast_n = prophet_f.predict(future_n)
    future_sel = forecast_n.tail(n_weeks)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weekly Average Forecast", f"LKR {future_sel['yhat'].mean():,.0f}")
    c2.metric(f"Total {n_weeks}-Week Forecast", f"LKR {future_sel['yhat'].sum()/1e6:.2f}M")
    c3.metric("Lower Bound (95%)", f"LKR {future_sel['yhat_lower'].mean():,.0f}")
    c4.metric("Upper Bound (95%)", f"LKR {future_sel['yhat_upper'].mean():,.0f}")

    st.markdown("---")

    # Forecast chart
    avg_hist = weekly.mean()

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=weekly.index, y=weekly.values,
        name='Historical Sales', line=dict(color='#3b82f6', width=2),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.07)'))
    fig_fc.add_trace(go.Scatter(
        x=future_sel['ds'], y=future_sel['yhat'],
        name='Forecast', line=dict(color='#10b981', width=2.5, dash='dash'),
        mode='lines+markers', marker=dict(size=8)))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([future_sel['ds'], future_sel['ds'][::-1]]),
        y=pd.concat([future_sel['yhat_upper'], future_sel['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(16,185,129,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'))
    fig_fc.add_hline(y=avg_hist*1.10, line_dash='dot', line_color='green',
                     annotation_text='High Season Zone (+10%)')
    fig_fc.add_hline(y=avg_hist*0.90, line_dash='dot', line_color='red',
                     annotation_text='Low Season Zone (-10%)')
    fig_fc.add_hline(y=avg_hist, line_dash='dash', line_color='gray',
                     annotation_text=f'Avg: LKR {avg_hist/1e6:.2f}M', line_width=1)
    fig_fc.add_vline(x=future_sel['ds'].iloc[0].timestamp() * 1000, line_dash='dot',
                     line_color='gray', annotation_text='Forecast Starts')
    fig_fc.update_layout(
        title=f'Prophet Sales Forecast — Next {n_weeks} Weeks with 95% Confidence Interval',
        yaxis_title='Total Sales (LKR)', xaxis_title='Date',
        height=480, plot_bgcolor='white',
        yaxis=dict(gridcolor='#e5e7eb', tickformat=',.0f'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Forecast table
    st.markdown('<div class="sub-header">Weekly Forecast Table</div>', unsafe_allow_html=True)
    ft = pd.DataFrame({
        'Week'            : future_sel['ds'].dt.strftime('%Y-%m-%d'),
        'Forecast (LKR)'  : future_sel['yhat'].round(0).astype(int),
        'Lower Bound'     : future_sel['yhat_lower'].round(0).astype(int),
        'Upper Bound'     : future_sel['yhat_upper'].round(0).astype(int),
        'vs Avg'          : [f"{(y-avg_hist)/avg_hist*100:+.1f}%" for y in future_sel['yhat']]
    })
    st.dataframe(ft, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 4 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Business Insights":

    st.markdown('<div class="main-header">📈 Business Insights & Recommendations</div>', unsafe_allow_html=True)

    with st.spinner("Loading insights..."):
        results = run_models(weekly)

    monthly_vals = list(monthly.values)
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    avg_monthly  = np.mean(monthly_vals)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual Revenue 2025",  f"LKR {sum(monthly_vals)/1e6:.2f}M")
    c2.metric("Peak Month",           "May",        f"LKR {max(monthly_vals)/1e6:.2f}M")
    c3.metric("Lowest Month",         "February",   f"LKR {min(monthly_vals)/1e6:.2f}M")
    c4.metric("Annual Growth",        "+9.2%",      "Jan → Dec 2025")

    st.markdown("---")

    # Monthly bar + MoM growth
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sub-header">Monthly Sales with Growth Zones</div>', unsafe_allow_html=True)
        bar_c = ['#ef4444' if v==max(monthly_vals) else '#3b82f6' if v==min(monthly_vals) else '#93c5fd' for v in monthly_vals]
        fig_m = go.Figure(go.Bar(x=month_labels, y=monthly_vals, marker_color=bar_c,
                                 marker_line_color='#1e3a5f', marker_line_width=1,
                                 text=[f'{v/1e6:.2f}M' for v in monthly_vals], textposition='outside'))
        fig_m.add_hline(y=avg_monthly, line_dash='dash', line_color='orange',
                        annotation_text=f'Avg LKR {avg_monthly/1e6:.2f}M')
        fig_m.update_layout(height=380, plot_bgcolor='white', yaxis=dict(gridcolor='#e5e7eb', tickformat=',.0f'),
                            yaxis_title='Sales (LKR)', xaxis_title='Month')
        st.plotly_chart(fig_m, use_container_width=True)

    with col2:
        st.markdown('<div class="sub-header">Month-over-Month Growth (%)</div>', unsafe_allow_html=True)
        growth_pct = [(monthly_vals[i]-monthly_vals[i-1])/monthly_vals[i-1]*100 for i in range(1, len(monthly_vals))]
        g_colors   = ['#10b981' if g >= 0 else '#ef4444' for g in growth_pct]
        fig_g = go.Figure(go.Bar(x=month_labels[1:], y=growth_pct, marker_color=g_colors,
                                 marker_line_color='black', marker_line_width=1,
                                 text=[f'{g:+.1f}%' for g in growth_pct], textposition='outside'))
        fig_g.add_hline(y=0, line_color='black', line_width=1)
        fig_g.update_layout(height=380, plot_bgcolor='white', yaxis=dict(gridcolor='#e5e7eb'),
                            yaxis_title='Growth (%)', xaxis_title='Month')
        st.plotly_chart(fig_g, use_container_width=True)

    # Product revenue
    st.markdown('<div class="sub-header">Revenue by Product Type</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    prod_sales = df.groupby('Product_Type')['Total_Sale'].sum().sort_values(ascending=False)
    colors_pie = ['#3b82f6','#8b5cf6','#10b981','#f59e0b','#ef4444']

    with col3:
        fig_pie = px.pie(values=prod_sales.values, names=prod_sales.index,
                         color_discrete_sequence=colors_pie, hole=0.35)
        fig_pie.update_traces(textinfo='label+percent', textfont_size=11)
        fig_pie.update_layout(height=360, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col4:
        fig_pb = px.bar(x=prod_sales.values, y=prod_sales.index, orientation='h',
                        color=prod_sales.index, color_discrete_sequence=colors_pie,
                        text=[f'LKR {v/1e6:.1f}M ({v/prod_sales.sum()*100:.1f}%)' for v in prod_sales.values])
        fig_pb.update_traces(textposition='outside')
        fig_pb.update_layout(height=360, showlegend=False, plot_bgcolor='white',
                             xaxis=dict(gridcolor='#e5e7eb', tickformat=',.0f'))
        st.plotly_chart(fig_pb, use_container_width=True)

    # Recommendations
    st.markdown("---")
    st.markdown('<div class="sub-header">Business Recommendations</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("""
        <div class="insight-box">
        <strong>📦 Inventory Planning</strong><br><br>
        • Stock up 4–6 weeks before April (peak season)<br>
        • Reduce slow stock in January–February<br>
        • Never stockout Industrial Filter or Water Purifier<br>
        • These two products drive highest revenue
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="insight-box">
        <strong>💰 Pricing Strategy</strong><br><br>
        • Premium pricing during April–May peak<br>
        • Bundle promotions in Jan–Feb slow season<br>
        • Loyalty discounts for Corporate customers<br>
        • May demand is 29.7% above February
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div class="insight-box">
        <strong>🎯 Forecast Reliability</strong><br><br>
        • Best model: Prophet (Tuned)<br>
        • MAPE: {results['prophet_mape']:.2f}% → ±{results['prophet_mape']:.1f}% accuracy<br>
        • 95% CI: LKR 7.4M–9.0M per week<br>
        • Stable growth expected Q1 2026
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    > **Limitation:** Weekly data was produced by linear interpolation of monthly totals,
    not real weekly transaction records. Actual weekly data would improve forecast precision.
    """)