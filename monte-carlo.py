import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="Interactive Risk Simulator")

# --- 1. INITIALIZE SESSION STATE ---
# This is crucial for a dynamic app. It stores the user's inputs.
if 'threat_paths' not in st.session_state:
    st.session_state.threat_paths = []
if 'consequences' not in st.session_state:
    st.session_state.consequences = []

# --- 2. THE MONTE CARLO ENGINE (Slightly modified to accept new dimensions) ---

def run_monte_carlo_simulation(bow_tie_model, num_simulations, volatility_multiplier, connectivity_failure_chance):
    annual_losses = []
    
    for i in range(num_simulations):
        total_loss_for_this_year = 0
        
        # MODELING CONNECTIVITY: Check for a cascade failure from an external risk
        is_cascade_failure_year = np.random.rand() < connectivity_failure_chance
        control_failure_multiplier = 3.0 if is_cascade_failure_year else 1.0

        # --- Simulate the Left Side (Threats -> Top Event) ---
        for path in bow_tie_model["threat_and_prevention_paths"]:
            # MODELING VOLATILITY: Adjust threat frequency based on the environment
            adjusted_frequency = path["threat"]["frequency_per_year"] * volatility_multiplier
            num_threat_events = np.random.poisson(adjusted_frequency)

            if num_threat_events > 0:
                for _ in range(num_threat_events):
                    top_event_occurs = True
                    for control in path["preventive_controls"]:
                        # Apply connectivity impact
                        adjusted_pof = min(1.0, control["probability_of_failure"] * control_failure_multiplier)
                        if np.random.rand() > adjusted_pof:
                            top_event_occurs = False
                            break
                    
                    if top_event_occurs:
                        loss_from_this_event = 0
                        for consequence in bow_tie_model["consequences_and_mitigation"]:
                            dist = consequence["impact_distribution"]
                            if dist[0] == 'triangular':
                                initial_impact = np.random.triangular(left=dist[1], mode=dist[2], right=dist[3])
                            
                            current_impact = initial_impact
                            for m_control in consequence["mitigative_controls"]:
                                # MODELING VELOCITY: Simpler proxy - effectiveness of mitigative control
                                # A more advanced model could make this time-based
                                if np.random.rand() < m_control["probability_of_success"]:
                                    current_impact *= (1 - m_control["impact_reduction_factor"])
                            
                            loss_from_this_event += current_impact
                        total_loss_for_this_year += loss_from_this_event

        annual_losses.append(total_loss_for_this_year)
    return annual_losses


# --- 3. HELPER FUNCTION TO BUILD THE MODEL FROM THE UI ---

def build_bowtie_from_session_state():
    threat_paths_data = []
    for path in st.session_state.threat_paths:
        threat_data = {"name": path['threat_name'], "frequency_per_year": path['threat_freq']}
        controls_data = [{"name": c['name'], "probability_of_failure": c['pof']} for c in path['controls']]
        threat_paths_data.append({"threat": threat_data, "preventive_controls": controls_data})

    consequences_data = []
    for cons in st.session_state.consequences:
        impact_dist = ("triangular", cons['impact_min'], cons['impact_mode'], cons['impact_max'])
        mit_controls_data = [{"name": mc['name'], "probability_of_success": mc['pos'], "impact_reduction_factor": mc['irf']} for mc in cons['mit_controls']]
        consequences_data.append({"name": cons['name'], "impact_distribution": impact_dist, "mitigative_controls": mit_controls_data})

    return {
        "risk_name": st.session_state.risk_name,
        "threat_and_prevention_paths": threat_paths_data,
        "consequences_and_mitigation": consequences_data
    }

# --- 4. THE STREAMLIT USER INTERFACE ---

st.title("Interactive Bowtie Risk Simulator")

# --- GLOBAL INPUTS & RISK DIMENSIONS ---
st.sidebar.header("Simulation Parameters")
num_simulations = st.sidebar.number_input("Number of Simulations (Years)", 1000, 100000, 10000, 1000)
st.session_state.risk_name = st.text_input("**Name of the Risk (Top Event)**", "Uncontrolled Transfer of Funds from Hot Wallet")
st.sidebar.markdown("---")

st.sidebar.header("Risk Dimensions")

# VOLATILITY
st.sidebar.subheader("4. Volatility")
volatility_multiplier = st.sidebar.slider(
    "Environmental Volatility Multiplier", 
    min_value=0.5, max_value=10.0, value=1.0, step=0.5,
    help="Models the overall environment. > 1.0 for a crisis/hostile scenario (threats are more frequent), < 1.0 for a stable one."
)

# VELOCITY (Proxy)
st.sidebar.subheader("3. Velocity (Proxy)")
st.sidebar.info("Velocity is complex. We use a proxy: the effectiveness of your mitigative controls to reduce damage quickly. Higher effectiveness = higher velocity of control.", icon="💡")


# CONNECTIVITY
st.sidebar.subheader("5. Connectivity")
connectivity_failure_chance = st.sidebar.slider(
    "Connectivity Cascade Failure Chance", 
    min_value=0.0, max_value=1.0, value=0.05, step=0.01,
    help="The probability in any given year that an external risk (e.g., infrastructure collapse) causes all control effectiveness to worsen."
)


# --- UI FOR BUILDING THE BOWTIE MODEL ---
st.header("Build Your Bowtie Model")

# --- LEFT SIDE: THREATS AND PREVENTIVE CONTROLS ---
st.subheader("Left Side: Threats & Preventive Controls")
if st.button("Add Threat Path"):
    st.session_state.threat_paths.append({'threat_name': '', 'threat_freq': 1.0, 'controls': []})

for i, path in enumerate(st.session_state.threat_paths):
    with st.container(border=True):
        st.markdown(f"**Threat Path {i+1}**")
        cols = st.columns([3, 2])
        path['threat_name'] = cols[0].text_input("Threat Name", key=f"t_name_{i}")
        # LIKELIHOOD
        path['threat_freq'] = cols[1].number_input("**Likelihood**: Freq/Year", min_value=0.0, value=1.0, step=0.1, key=f"t_freq_{i}", help="How many times per year this threat occurs, on average.")
        
        st.markdown("**Preventive Controls for this Path**")
        if st.button("Add Preventive Control", key=f"add_pc_{i}"):
            path['controls'].append({'name': '', 'pof': 0.1})
            
        for j, control in enumerate(path['controls']):
            c_cols = st.columns([3, 2, 1])
            control['name'] = c_cols[0].text_input("Control Name", key=f"pc_name_{i}_{j}")
            control['pof'] = c_cols[1].slider("Probability of Failure", 0.0, 1.0, 0.1, 0.01, key=f"pc_pof_{i}_{j}")
            if c_cols[2].button("🗑️", key=f"del_pc_{i}_{j}"):
                path['controls'].pop(j)
                st.rerun()

        if st.button("Delete Threat Path", key=f"del_path_{i}", type="secondary"):
            st.session_state.threat_paths.pop(i)
            st.rerun()


# --- RIGHT SIDE: CONSEQUENCES AND MITIGATIVE CONTROLS ---
st.subheader("Right Side: Consequences & Mitigative Controls")
if st.button("Add Consequence"):
    st.session_state.consequences.append({'name': '', 'impact_min': 0, 'impact_mode': 100000, 'impact_max': 1000000, 'mit_controls': []})

for i, cons in enumerate(st.session_state.consequences):
    with st.container(border=True):
        st.markdown(f"**Consequence {i+1}**")
        cons['name'] = st.text_input("Consequence Name", key=f"c_name_{i}")
        
        st.markdown("**Severity**: Potential Impact Range (Triangular Distribution)")
        s_cols = st.columns(3)
        cons['impact_min'] = s_cols[0].number_input("Min Impact ($)", value=0, key=f"c_min_{i}")
        cons['impact_mode'] = s_cols[1].number_input("Most Likely Impact ($)", value=100000, key=f"c_mode_{i}")
        cons['impact_max'] = s_cols[2].number_input("Max Impact ($)", value=1000000, key=f"c_max_{i}")

        st.markdown("**Mitigative Controls for this Consequence**")
        if st.button("Add Mitigative Control", key=f"add_mc_{i}"):
            cons['mit_controls'].append({'name': '', 'pos': 0.8, 'irf': 0.5})

        for j, m_control in enumerate(cons['mit_controls']):
            mc_cols = st.columns([3, 2, 2, 1])
            m_control['name'] = mc_cols[0].text_input("Control Name", key=f"mc_name_{i}_{j}")
            m_control['pos'] = mc_cols[1].slider("Prob. of Success", 0.0, 1.0, 0.8, 0.01, key=f"mc_pos_{i}_{j}")
            m_control['irf'] = mc_cols[2].slider("Impact Reduction Factor", 0.0, 1.0, 0.5, 0.05, key=f"mc_irf_{i}_{j}", help="If successful, how much of the remaining damage is removed (e.g., 0.8 = 80% reduction).")
            if mc_cols[3].button("🗑️", key=f"del_mc_{i}_{j}"):
                 cons['mit_controls'].pop(j)
                 st.rerun()
        
        if st.button("Delete Consequence", key=f"del_cons_{i}", type="secondary"):
            st.session_state.consequences.pop(i)
            st.rerun()

st.divider()

# --- RUN SIMULATION ---
if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
    if not st.session_state.threat_paths or not st.session_state.consequences:
        st.error("Please add at least one Threat Path and one Consequence before running the simulation.")
    else:
        final_bowtie_model = build_bowtie_from_session_state()
        with st.spinner(f"Running {num_simulations:,} simulations..."):
            losses = run_monte_carlo_simulation(final_bowtie_model, num_simulations, volatility_multiplier, connectivity_failure_chance)
            results_df = pd.DataFrame(losses, columns=['Annual Loss'])

        st.success("Simulation Complete!")
        st.header("Simulation Results")

        col1, col2, col3, col4 = st.columns(4)
        mean_loss = results_df['Annual Loss'].mean()
        prob_loss = (results_df['Annual Loss'] > 0).mean() * 100
        var_95 = results_df['Annual Loss'].quantile(0.95)
        
        col1.metric("Mean Annual Loss (ALE)", f"${mean_loss:,.0f}")
        col2.metric("Probability of Any Loss in a Year", f"{prob_loss:.1f}%")
        col3.metric("95th Percentile Loss (VaR)", f"${var_95:,.0f}", help="There is a 5% chance that the annual loss will exceed this amount.")
        col4.metric("Max Loss in Simulation", f"${results_df['Annual Loss'].max():,.0f}")

        fig = px.histogram(results_df, x='Annual Loss', nbins=50, title=f'Distribution of Potential Annual Losses')
        st.plotly_chart(fig, use_container_width=True)
