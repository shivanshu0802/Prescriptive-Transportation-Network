import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pulp

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategic Supply Chain Optimizer", layout="wide")

# --- DATA LOADING (REAL CASE DATA) ---
@st.cache_data
def load_all_data():
    # 1. Demand Points
    df_d = pd.read_csv('DataSet.csv')
    df_d.columns = df_d.columns.str.strip()
    
    # 2. Road Distance Matrix (Real OSRM data from paper)
    df_dm_p = pd.read_csv('Section_1_Group_1_Books_DM.csv', index_col=0)
    df_dm_p.columns = df_dm_p.columns.str.strip()
    
    return df_d, df_dm_p

df_demand, df_dm_paper = load_all_data()

# --- DISTANCE LOGIC ---
def haversine_road_est(lat1, lon1, lat2, lon2):
    R = 6371.0 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))) * 1.25 

# --- SIDEBAR ---
st.sidebar.title("🏢 Strategic Planning")
st.sidebar.info("Solving the Transportation Problem using PuLP/Simplex Algorithm.")

mode = st.sidebar.radio("Optimization Mode", ["Single Category", "Combined Supply Chain"])
demand_cols = [c for c in df_demand.columns if "Demand -" in c]

if mode == "Single Category":
    target_prod = st.sidebar.selectbox("Select Product", demand_cols)
    df_demand['Active_Demand'] = df_demand[target_prod]
else:
    df_demand['Active_Demand'] = df_demand[demand_cols].sum(axis=1)

# CUSTOM HUB OPTIONS
hub_options = {
    "WH1 (NCR/North)": [28.545470, 77.340203],
    "WH2 (Hyderabad/South)": [17.151102, 78.290139],
    "WH3 (Mumbai/West)": [19.063115, 73.021483],
    "Kolkata (East)": [22.57, 88.36],
    "Ahmedabad (West)": [23.02, 72.57],
    "Bangalore (South)": [12.97, 77.59]
}
selected_hubs = st.sidebar.multiselect("Select Warehouses to Open", 
                                       list(hub_options.keys()), 
                                       default=["WH1 (NCR/North)", "WH2 (Hyderabad/South)", "WH3 (Mumbai/West)"])

unit_cost = st.sidebar.slider("Transport Cost (INR per KM/Unit)", 0.5, 3.0, 1.0)

# --- THE PRESCRIPTIVE MODEL ---
def run_optimization(df_d, hubs, cost_rate):
    model = pulp.LpProblem("Supply_Chain_Minimization", pulp.LpMinimize)
    
    tot_d = df_d['Active_Demand'].sum()
    s_bar = (tot_d / len(hubs)) * 1.25 # 25% Capacity Buffer
    
    wh_list = []
    for h in hubs:
        wh_list.append({'Warehouse': h, 'lat': hub_options[h][0], 'lon': hub_options[h][1], 'Capacity': int(s_bar)})
    df_w = pd.DataFrame(wh_list)
    
    cust_names = df_d['Demand Point Name'].tolist()
    x = pulp.LpVariable.dicts("ship", (hubs, cust_names), lowBound=0)
    
    obj_terms = []
    for i, wh in enumerate(hubs):
        for j, cust in enumerate(cust_names):
            wh_key = wh.split(" (")[0]
            if wh_key in df_dm_paper.columns:
                dist = df_dm_paper.loc[j, wh_key] # REAL ROAD DATA
            else:
                dist = haversine_road_est(df_w.iloc[i]['lat'], df_w.iloc[i]['lon'], 
                                         df_d.iloc[j]['Masked Latitude'], df_d.iloc[j]['Masked Longitude'])
            obj_terms.append(dist * cost_rate * x[wh][cust])
            
    model += pulp.lpSum(obj_terms)
    
    for i, wh in enumerate(hubs):
        model += pulp.lpSum([x[wh][c] for c in cust_names]) <= df_w.iloc[i]['Capacity']
    for j, cust in enumerate(cust_names):
        model += pulp.lpSum([x[wh][cust] for wh in hubs]) >= df_d.iloc[j]['Active_Demand']
        
    model.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[model.status] == 'Optimal':
        results = []
        for i, wh in enumerate(hubs):
            for j, cust in enumerate(cust_names):
                vol = pulp.value(x[wh][cust])
                if vol and vol > 0.1:
                    results.append({
                        'wh': wh, 'cust': cust, 'vol': vol, 
                        'w_lat': df_w.iloc[i]['lat'], 'w_lon': df_w.iloc[i]['lon'],
                        'c_lat': df_d.iloc[j]['Masked Latitude'], 'c_lon': df_d.iloc[j]['Masked Longitude']
                    })
        return results, pulp.value(model.objective), df_w
    return None, None, None

# --- UI RENDER ---
st.title("🚛 LogiSpatial Strategic Optimizer")

if not selected_hubs:
    st.warning("Select warehouses to calculate the Optimal solution.")
    st.stop()

with st.spinner("Solving Prescriptive Model..."):
    flows, total_cost, final_wh_df = run_optimization(df_demand, selected_hubs, unit_cost)

if flows:
    # 1. Metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Nodes", f"{len(df_demand)}")
    k2.metric("Total Load", f"{df_demand['Active_Demand'].sum():,}")
    k3.metric("System Slack", f"{final_wh_df['Capacity'].sum() - df_demand['Active_Demand'].sum():,}")
    k4.metric("Optimal Cost", f"₹ {total_cost/1e6:.2f} M")

    # 2. High Contrast Map
    st.markdown("#### **India Distribution Network: Optimal Prescriptive Flows**")
    fig = go.Figure()
    
    # Supply Lines (Neon Yellow)
    for f in flows:
        fig.add_trace(go.Scattergeo(lon=[f['w_lon'], f['c_lon']], lat=[f['w_lat'], f['c_lat']],
                                   mode='lines', line=dict(width=1.2, color='#FFFF00'), opacity=0.4, showlegend=False))
    
    # Hubs (Neon Red Stars)
    fig.add_trace(go.Scattergeo(lon=final_wh_df['lon'], lat=final_wh_df['lat'], 
                               marker=dict(size=15, color='#FF0000', symbol='star', line=dict(width=1, color='white')), 
                               name="Warehouse Hubs", text=final_wh_df['Warehouse']))
    
    # Customers (Neon Cyan)
    fig.add_trace(go.Scattergeo(lon=df_demand['Masked Longitude'], lat=df_demand['Masked Latitude'],
                               marker=dict(size=4, color='#00FFFF', opacity=0.7), name="Demand Points"))

    fig.update_layout(geo=dict(scope='asia', center=dict(lat=22, lon=78), projection_scale=4.5, 
                               showland=True, landcolor="#121212", bgcolor="black", subunitcolor="#555"),
                      margin={"r":0,"t":0,"l":0,"b":0}, height=550, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # 3. Tables
    st.subheader("Operational Breakdown")
    df_flows = pd.DataFrame(flows)
    wh_sum = df_flows.groupby('wh')['vol'].sum().reset_index()
    
    c_l, c_r = st.columns(2)
    with c_l:
        st.write("**Utilization Table**")
        final_wh_df['Used_Load'] = final_wh_df['Warehouse'].map(wh_sum.set_index('wh')['vol']).fillna(0).astype(int)
        final_wh_df['% Load'] = (final_wh_df['Used_Load'] / final_wh_df['Capacity'] * 100).round(1)
        st.table(final_wh_df[['Warehouse', 'Capacity', 'Used_Load', '% Load']])
    with c_r:
        st.write("**Network Volume Distribution**")
        st.bar_chart(wh_sum.set_index('wh'))

else:
    st.error("Optimization failed. Try adding more warehouses.")
