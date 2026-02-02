import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import io
import numpy as np
import webbrowser
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional
from minefield_logic import (
    MinefieldPoint, 
    MineStrip,
    Landmark,
    SafeLane,
    calculate_coords, 
    calculate_strip_coords, 
    generate_minefield_data,
    plot_survey_chain_interactive,
    generate_user_defined_trace,
    generate_kml,
    save_record,
    load_record,
    list_records,
    calculate_stopping_power,
    predict_density_for_pk,
    generate_tactical_layout,
    calculate_cumulative_stopping_power,
    generate_kml_from_layout,
    generate_folium_map,
    plot_survey_chain,
    dms_to_decimal,
    KNOWN_TANKS,
    KNOWN_MINES
)

# --- Login Logic & CSS ---

def get_img_as_base64(file_path):
    import base64
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def login_page():
    # Load Logo
    try:
        if os.path.exists("assets/damp_logo.jpg"):
            img_b64 = get_img_as_base64("assets/damp_logo.jpg")
            bg_image_css = f"""
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url("data:image/jpeg;base64,{img_b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            """
        else:
             # Fallback
             bg_image_css = "background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);"
    except:
        bg_image_css = "background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);"

    # Custom CSS for "Premium / Tactical" Theme
    st.markdown(f"""
    <style>
    /* Gradient Background */
    .stApp {{
        {bg_image_css}
    }}
    
    /* Login Container (Compact & Readable) */
    .login-container {{
        background: rgba(0, 0, 0, 0.75); /* Darker for better contrast */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 30px 25px; /* Reduced Padding */
        border-radius: 12px;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.6);
        text-align: center;
        width: 100%;
        max-width: 380px; /* Slightly Narrower */
        margin: auto;
    }}
    
    /* Typography */
    .damp-title {{
        color: #e6b800 !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 32px; /* Explicit size */
        letter-spacing: 3px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.9);
        margin-bottom: 5px;
        line-height: 1.2;
    }}
    .damp-subtitle {{
        color: #f0f0f0 !important;
        font-family: 'Consolas', monospace;
        font-size: 14px; /* Reduced */
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 20px;
        opacity: 0.9;
    }}
    
    .regiment-tag {{
        color: #e6b800; /* Gold */
        font-family: 'Arial', sans-serif;
        font-size: 12px;
        font-weight: bold;
        text-align: center;
        margin-top: 5px;
        letter-spacing: 1px;
        width: 100%;
    }}

    /* Input Fields */
    .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.95);
        color: #111;
        border: 2px solid #555;
        border-radius: 6px;
        padding: 10px;
        font-size: 14px;
        margin-bottom: 5px;
    }}
    
    /* Button */
    .stButton > button {{
        background: linear-gradient(90deg, #b8860b 0%, #ffd700 100%);
        color: #000;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: none;
        border-radius: 6px; /* Match input */
        padding: 12px;
        font-size: 15px;
        margin-top: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.7);
        filter: brightness(1.1);
    }}
    
    /* Footer */
    .footer-text {{
        color: rgba(255, 255, 255, 0.5);
        font-size: 11px;
        margin-top: 25px;
        font-family: monospace;
        text-shadow: 0 1px 2px black;
    }}
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        # Using div classes instead of h1/h3 to avoid Streamlit anchors
        st.markdown("""
        <div class='login-container'>
            <div class='damp-title'>D A M P</div>
            <div class='damp-subtitle'>Digitized & Automated Mine Planning</div>
            <div style="height: 1px; background-color: rgba(255,255,255,0.2); margin: 15px auto; width: 60%;"></div>
            <div class='regiment-tag'>19 MADRAS</div>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Operator ID", placeholder="Enter Service Number")
        password = st.text_input("Access Code", type="password", placeholder="Enter Password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("INITIALIZE SYSTEM"):
            # Credential Check
            if username == "CARNATIC@123" and password == "PASSWORD":
                st.session_state['authenticated'] = True
                st.success("AUTHENTICATION SUCCESSFUL. LOADING TACTICAL MODULES...")
                st.rerun()
            else:
                st.error("ACCESS DENIED. INCORRECT CREDENTIALS.")

        st.markdown("<div class='footer-text'>RESTRICTED ACCESS // AUTHORIZED PERSONNEL ONLY</div>", unsafe_allow_html=True)

def main_app():
    # Global Theme CSS
    st.markdown("""
    <style>
    .stApp {
        background-color: #F8F9FA;
        background-image: none !important; /* Force remove login background */
        color: #212529;
    }
    h1, h2, h3 {
        color: #333333 !important;
    }
    /* Standard Input Styling */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stDateInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ced4da;
    }
    </style>
    """, unsafe_allow_html=True)

    # Regular App Logic
    
    # Sidebar removed as per user request to clean up the interface.


    tab1, tab3, tab4 = st.tabs(["Quick Calculator", "Strategic Planner", "Digital Mapping"])

    # === TAB 1: Quick Calculator ===
    with tab1:
        st.header("DIGITIZED AND AUTOMATED MINE PLANNING (DAMP)")
        st.caption("Plan Resources and Location Geometry")
        
        
        # --- STEP 1: DIMENSIONS & RESOURCES ---
        st.subheader("Step 1: Dimensions & Resources")
        c1, c2 = st.columns(2)
        with c1:
            frontage = st.number_input("Field Frontage (m)", min_value=1.0, value=500.0)
            depth = st.number_input("Field Depth (m)", min_value=1.0, value=200.0)
        with c2:
            calc_option = st.radio("Input Method", ["Density of Mines", "Total Mines Available"], horizontal=True)
            mine_inputs = {}
            if calc_option == "Total Mines Available":
                mine_inputs['ap'] = st.number_input("Total AP", min_value=0, value=45)
                mine_inputs['at'] = st.number_input("Total AT", min_value=0, value=15)
                mine_inputs['frag'] = st.number_input("Total Frag", min_value=0, value=7)
            else:
                mine_inputs['ap'] = st.number_input("Spacing AP (m)", min_value=0.1, value=1.0)
                mine_inputs['at'] = st.number_input("Spacing AT (m)", min_value=0.1, value=3.0)
                mine_inputs['frag'] = st.number_input("Spacing Frag (m)", min_value=0.1, value=12.0)
        
        st.divider()
        
        # --- STEP 2: STRIP CONFIGURATION ---
        st.subheader("Step 2: Strip Configuration")
        c_s1, c_s2 = st.columns(2)
        num_strips = c_s1.number_input("Number of Strips", min_value=1, max_value=10, value=1)
        
        strip_depth = depth / num_strips if num_strips > 0 else 30
        c_s2.info(f"Approx Spacing/Depth: {strip_depth:.1f} m")
        
        if strip_depth < 25 or strip_depth > 150:
             c_s2.warning("⚠️ Strip Spacing should be between 25m and 150m!")
        
        st.divider()



            

        
        # --- STEP 3: GEODETIC LOCATION (STRIP DEFINITION) ---
        st.subheader("Step 3: Geodetic Layout")
        
        # 0. Direction (Moved Up for Safety Checks)
        global_enemy_bearing = st.number_input("Enemy Direction (Bearing)", 0, 360, 0, help="Used for Safety Checks (Ensure SSM is Ahead of LM).")

        # Landmark
        c_lm1, c_lm2 = st.columns(2)
        lm_e = c_lm1.number_input("Landmark Easting", value=1000.0)
        lm_n = c_lm2.number_input("Landmark Northing", value=1000.0)
        
        strip_configs = []
        
        # Loop for each strip
        for s in range(1, num_strips + 1):
            with st.expander(f"Strip {s} Layout", expanded=(s==1)):
                c_ssm1, c_ssm2 = st.columns(2)
                
                # SSM Definition
                ssm_bear = c_ssm1.number_input(f"Bearing LM -> SSM-{s}", 0.0, 360.0, 45.0, key=f"b_ssm_{s}")
                
                # Safety 1: Min Distance Check (50m)
                ssm_dist = c_ssm2.number_input(f"Distance LM -> SSM-{s}", 0.0, 5000.0, 100.0 + (s-1)*60, key=f"d_ssm_{s}")
                
                if ssm_dist < 50.0:
                     st.error(f"⛔ SAFETY ERROR: SSM-{s} is too close to Landmark ({ssm_dist}m). Minimum 50m required.")
                
                # Safety 2: Forward Check (Ahead of Landmark)
                # Angle difference between Enemy Bearing and SSM Bearing should be < 90 degrees (General Front)
                # normalizing angle diff
                angle_diff = abs((ssm_bear - global_enemy_bearing + 180) % 360 - 180)
                if angle_diff > 90:
                    st.warning(f"⚠️ TACTICAL WARNING: SSM-{s} is located BEHIND the Landmark (Relative to Enemy Dir {global_enemy_bearing}°). Verify safety.")

                if ssm_dist > 150:
                    st.warning("⚠️ Recommendation: Keep SSM distance < 150m from reference.")

                # Safety Gap Check
                if s > 1:
                     # Heuristic: Compare with previous strip's roughly estimated distance
                     # ideally we should store prev ssm coords, but here we can just use the input logic
                     # assuming linear progression away from LM.
                     # Dist diff is a proxy for spacing if bearing is similar.
                     # Let's try to get prev value from session state if possible, or just re-calc.
                     # Simpler: Just check difference in 'Distance LM -> SSM' input if bearings are same.
                     # If bearings differ, we need full coord calc.
                     prev_dist = 100.0 + (s-2)*60 # Default fallback
                     # Try to fetch actual input from widget state if available, hard to access dynamically in loop before rerun.
                     # Actually, Streamlit widget keys: f"d_ssm_{s-1}"
                     # We can access st.session_state[f"d_ssm_{s-1}"]
                     prev_key = f"d_ssm_{s-1}"
                     if prev_key in st.session_state:
                         prev_val = st.session_state[prev_key]
                         gap = abs(ssm_dist - prev_val)
                         # Also check bearing difference? If bearings are same, gap is purely distance diff.
                         # If bearings close, it holds.
                         if gap < 55.0:
                             st.error(f"⛔ SAFETY VIOLATION: Strip {s} is too close to Strip {s-1} (Gap: {gap:.1f}m). Minimum 55m required (45m clear gap).")
                
                # Calculate SSM Coords
                ssm_e, ssm_n = calculate_coords(lm_e, lm_n, ssm_bear, ssm_dist)
                st.write(f"📍 **SSM-{s}**: E {ssm_e:.1f}, N {ssm_n:.1f}")
                
                # Strip Trace (Straight or User Defined Zig-Zag)
                st.markdown("**Strip Trace (Zig-Zag B, B+20)**")
                c_ax1, c_ax2 = st.columns(2)
                
                first_leg_bearing = c_ax1.number_input(f"First Leg Bearing (Deg)", 0.0, 360.0, 90.0, key=f"first_b_{s}")
                # Optional: Override frontage per strip? keeping uniform for now
                strip_frontage = c_ax2.number_input(f"Strip Frontage", value=frontage, key=f"sf_{s}", disabled=True)
                
                # Single Row Constraint (Hidden/Implicit)
                rows_per_strip = 1 
                
                from minefield_logic import generate_user_defined_trace 
                
                # Generate Trace (Now with Strip ID)
                trace = generate_user_defined_trace(ssm_e, ssm_n, first_leg_bearing, frontage, strip_id=s)
                
                # Show Trace Info
                st.caption(f"Generated {len(trace)-1} Legs (B={first_leg_bearing:.0f}, B+20). Max Leg < 150m.")
                
                strip_configs.append({
                    "trace": trace,
                    "row_count": rows_per_strip,
                    "depth": strip_depth,
                    "is_first": (s == 1), # Only first strip gets M16
                    "ssm_coords": (ssm_e, ssm_n) # For setting out vis
                })

        # Global Settings
        st.markdown("### Tactical Settings")
        st.caption("Enemy Direction set in previous step.")

        st.divider()
        
        # --- STEP 4: GENERATE ---
        if st.button("Generate Tactical Plan", type="primary", use_container_width=True):
            
            # Call Generator (New Signature)
            # detail_view logic: keep simple/default
            
            fig, _, df_stores, all_mines = generate_minefield_data(
                strip_configs, mine_inputs, frontage=frontage, depth=depth
            )
            
            # Persist for Digital Mapping (Tab 4)
            st.session_state['latest_plan'] = {
                "all_mines": all_mines,
                "strip_configs": strip_configs,
                "lm_coords": (lm_e, lm_n),
                "frontage": frontage,
                "depth": depth
            }
            
            st.success("Plan Generated Successfully!")
            
            # --- RESULTS ---
            
            # 1. Geodetic Map (Combined)
            st.subheader("Tactical Map")
            
            # Combine all traces for plotting
            all_traces_flat = []
            setting_out_lines = []
            
            for cfg in strip_configs:
                all_traces_flat.extend(cfg['trace'])
                # Add Line from LM -> SSM
                ssm_x, ssm_y = cfg['ssm_coords']
                setting_out_lines.append((lm_e, lm_n, ssm_x, ssm_y))
            
            # Use Interactive Plot - Now taking enemy_bearing?
            fig_geo = plot_survey_chain_interactive(
                all_traces_flat, 
                mine_points=all_mines,
                landmark_coord=(lm_e, lm_n),
                setting_out_lines=setting_out_lines
            )
            
            # Add Enemy Arrow
            if fig_geo:
                # Arrow length logic
                # Vector for direction
                rad_en = math.radians(global_enemy_bearing)
                # Plotly annotations use paper coords or plot coords.
                # Let's put a big red arrow in the corner or center?
                # Corner is better.
                fig_geo.add_annotation(
                    text=f"ENEMY ({global_enemy_bearing}°)",
                    xref="paper", yref="paper",
                    x=0.05, y=0.95,
                    showarrow=False, # We'll draw the arrow visually or just text? 
                    # Arrow is hard with arbitrary angle in annotations unless using 'ax', 'ay'.
                    # Let's just state the direction for now as text.
                    font=dict(size=14, color="red", weight="bold"),
                    bgcolor="white", bordercolor="red"
                )
            
            st.plotly_chart(fig_geo, use_container_width=True)
            
            # 2. Stores & Logistics
            st.subheader("Logistics Requirements")
            st.dataframe(df_stores.T, use_container_width=True)
            
            # --- SEPARATE REPORTS PER STRIP ---
            st.subheader("Field Minefield Record (Per Strip)")
            
            # Create a Tabs view for report? Or consecutive tables?
            # User wants "A REPORT SEPERATELY FOR EACH STRIP".
            # Expanding or Tabs works best.
            
            tabs = st.tabs([f"Strip {s}" for s in range(1, len(strip_configs) + 1)])
            
            for s_idx, tab in enumerate(tabs):
                s_num = s_idx + 1
                with tab:
                    st.markdown(f"#### Strip {s_num} Survey Control")
                    
                    # 1. Control Data (LM, SSM, TPs, ESM)
                    control_data = []
                    # LM (Always included as reference)
                    control_data.append({"Point": "Landmark (LM)", "Easting": lm_e, "Northing": lm_n, "Note": "Reference"})
                    
                    # SSM
                    # Find SSM in trace
                    trace = strip_configs[s_idx]['trace']
                    for pt in trace:
                        l, e, n = pt
                        if "SSM" in l or "TP" in l or "ESM" in l:
                            control_data.append({"Point": l, "Easting": round(e, 1), "Northing": round(n, 1), "Note": "Control"})
                            
                    # DPs (from mines)
                    dps_strip = [m for m in all_mines if m.get('strip_id') == s_num and m.get('type') == 'DP']
                    for dp in dps_strip:
                         control_data.append({"Point": dp['label'], "Easting": dp['Grid Easting'], "Northing": dp['Grid Northing'], "Note": "Dir Picquet"})
                    
                    df_ctrl = pd.DataFrame(control_data)
                    st.table(df_ctrl)
                    
                    st.markdown(f"#### Strip {s_num} Mine Inventory")
                    # Filter mines for this strip (exclude DPs from this list? user said "EACH MINE")
                    mines_strip = [m for m in all_mines if m.get('strip_id') == s_num and m.get('type') != 'DP']
                    
                    if mines_strip:
                        df_m = pd.DataFrame(mines_strip)
                        # Columns: mine_id, type, Grid Easting, Grid Northing
                        cols = ["mine_id", "type", "Grid Easting", "Grid Northing"]
                        st.dataframe(df_m[cols], use_container_width=True, hide_index=True)
                        
                        # Download Button for THIS strip
                        csv_strip = df_m[cols].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            f"📥 Download Strip {s_num} Mines", 
                            csv_strip, 
                            f"strip_{s_num}_mines.csv", 
                            "text/csv"
                        )
                    else:
                        st.info("No mines generated for this strip.")

            # 3. Downloads (Individual)
            c_d1, c_d3 = st.columns(2)
            
            csv_stores = df_stores.to_csv(index=False).encode('utf-8')
            c_d1.download_button("📥 Download Stores", csv_stores, "stores.csv", "text/csv", use_container_width=True)
            
            # Export Mines (Full Detail)
            df_mines = pd.DataFrame(all_mines)
            csv_mines = df_mines.to_csv(index=False).encode('utf-8')
            c_d3.download_button("📥 Download Full Inventory", csv_mines, "mine_inventory.csv", "text/csv", use_container_width=True)

            st.divider()
            
            # --- COMPREHENSIVE REPORT SECTION ---
            st.header("📄 Consolidated Minefield Report")
            st.caption("Printable View for Mine Laying Party")
            
            # Use checkbox to keep report visible during interaction (downloads, etc.)
            if st.checkbox("Show Final Report View", value=False):
                st.markdown("---")
                st.subheader("1. Stores Required")
                st.table(df_stores.T)
                
                st.subheader("2. Mine Quantity vs Type")
                # Simple Bar Chart
                chart_data = pd.DataFrame({
                    "Type": ["Anti-Personnel", "Anti-Tank", "Fragmentation"],
                    "Quantity": [
                        df_stores['ap'][0], 
                        df_stores['at'][0], 
                        df_stores['frag'][0]
                    ]
                })
                st.bar_chart(chart_data, x="Type", y="Quantity", color="#ff4b4b")
                
                st.subheader("3. Full Geodetic Record (GR)")
                # Re-generate the survey dataframe here for display
                full_survey_data = []
                full_survey_data.append({"Type": "LANDMARK", "ID": "LM", "Easting": lm_e, "Northing": lm_n})
                
                for s_idx, cfg in enumerate(strip_configs):
                    strip_id = f"Strip-{s_idx+1}"
                    for pt in cfg['trace']:
                         l, e, n = pt
                         full_survey_data.append({"Type": "SURVEY", "ID": l, "Easting": e, "Northing": n, "Strip": strip_id})
                
                # Add All Mines
                for m in all_mines:
                    full_survey_data.append({
                        "Type": m['type'].upper(),
                        "ID": m.get('mine_id', m.get('label')),
                        "Easting": m['Grid Easting'],
                        "Northing": m['Grid Northing'],
                        "Strip": f"Strip-{m.get('strip_id', '?')}"
                    })
                    
                df_full_report = pd.DataFrame(full_survey_data)
                st.dataframe(df_full_report, use_container_width=True)
                
                # Download Button for this Master Record
                csv_report = df_full_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Entire Control & Mine Record", 
                    csv_report, 
                    "master_minefield_record.csv", 
                    "text/csv",
                    type="primary"
                )

            # 4. M16 Details (Only Strip 1)
            m16_rows = [m for m in all_mines if m['type'] == 'frag']
            if m16_rows:
                st.divider()
                st.markdown("### M16 Fragmentation Mines (Strip 1 Only)")
                st.dataframe(pd.DataFrame(m16_rows)[['label', 'Grid Easting', 'Grid Northing']], hide_index=True)
                
        st.divider()

    

    
    # === TAB 3: Strategic Planner ===
    with tab3:
        st.header("Strategic Minefield Planner")
        st.caption("Automated Planning based on Desired Effect (Stopping Power)")
        
        # Display Methodology Guide
        with st.expander("ℹ️ Process Methodology", expanded=False):
            st.markdown("""
            **Step 1: User Input Collection** - Enter tactical specifications (Target, Mine Model, Dimensions).
            **Step 2: Stopping Power Calculation** - System computes probability of casualty (P_total) based on track width vs mine overlap.
            **Step 3: Minefield Pattern Prediction** - Determines optimal density and spacing.
            **Step 4: 2D Planning (Top View)** - Generates visual layout of the field.
            **Step 5: Output Generation** - Produces final report with values and projections.
            **Step 6: Real-World Projection** - Exports the plan to Google Earth (KML).
            """)

        col_p1, col_p2 = st.columns([1, 2])
        
        with col_p1:
            st.subheader("Step 1: Specifications")
            
            # Inputs
            st.markdown("**Target & Munition**")
            target_vehicle = st.selectbox("Target Vehicle", list(KNOWN_TANKS.keys()))
            mine_model = st.selectbox("Mine Model", list(KNOWN_MINES.keys()))
            
            # Determine mine type from model (Simple inference for now)
            # ND MK1 -> Anti-Personnel? ND MK3 -> Anti-Tank?
            # User diagram suggests ND MK1 is small (0.09m), MK3 is large (0.76m).
            # Let's map them for the visualizer colors.
            mine_viz_type = "Anti-Tank" if "MK3" in mine_model else "Anti-Personnel"
            
            st.markdown("**Area of Deployment**")
            c1, c2 = st.columns(2)
            frontage = c1.number_input("Frontage (m)", min_value=10.0, value=500.0, key="plan_frontage")
            depth = c2.number_input("Depth (m)", min_value=10.0, value=200.0, key="plan_depth")
            
            st.markdown("---")
            st.markdown("**Target Effect**")
            target_pk = st.slider("Desired Stopping Power (%)", 1, 99, 80) / 100.0
            
            # Manual Density Override (Optional)
            use_manual_density = st.checkbox("Manual Density Input")
            manual_density = 1.0
            if use_manual_density:
                manual_density = st.number_input("Density (Mines/m)", 0.1, 5.0, 1.0)
            
            if st.button("GENERATE PLAN", type="primary"):
                # Step 2: Stopping Power Calculation
                # If Reverse (Predict req density for Pk):
                # The advanced formula is complex to inverse algebraically. 
                # We can iterate or use the simple inverse as an estimate, then refinement?
                # For now, let's use the provided 'forward' formula on the proposed density.
                
                # If Manual Density is NOT used, we need to find density for Target Pk.
                # Since the formula is P_total = 1 - Prod(1-P_row), and P_row depends on density.
                # Let's simple-search or use the manual density if provided.
                # Start with a guess or 1.0
                
                calc_density = manual_density
                
                # Calculate P_total for this density (Assuming 3 rows for standard mixed)
                # Standard Mixed: 2 rows of AT? Or just 1?
                # User diagram: Row 1 AP, Row 2 AT, Row 3 Frag. 
                # Against a TANK, only MK3 (AT) works? Or MK1 hurts tracks?
                # Let's assume uniform rows of the selected mine for the main stopping power calc.
                p_total = calculate_cumulative_stopping_power(target_vehicle, mine_model, [calc_density])
                
                # If we want to FIND density for Pk, we could loop:
                if not use_manual_density:
                    # Simple iterative solver
                    for d in [x/10.0 for x in range(1, 50)]: # 0.1 to 5.0
                        p = calculate_cumulative_stopping_power(target_vehicle, mine_model, [d])
                        if p >= target_pk:
                            calc_density = d
                            p_total = p
                            break
                
                total_mines = int(calc_density * frontage)
                spacing = 1.0 / calc_density if calc_density > 0 else 0
                
                st.session_state['plan_results'] = {
                    "density": calc_density,
                    "spacing": spacing,
                    "pk": p_total,
                    "total": total_mines,
                    "target": target_vehicle,
                    "mine_model": mine_model,
                    "mine_type": mine_viz_type # For visualization colors
                }

        with col_p2:
            st.subheader("Tactical Output")
            
            res = st.session_state.get('plan_results')
            
            if res:
                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Target Vehicle", f"{res['target']}")
                m2.metric("Stopping Power", f"{res['pk']*100:.1f}%")
                m3.metric("Rec. Density", f"{res['density']:.1f} m/m")
                m4.metric("Spacing", f"{res['spacing']:.2f} m")
                st.divider()
                
                # Step 4: 2D Planning (Top View)
                # 4. Display Top-View Layout
                st.markdown(f"**Step 4: Tactical Deployment Guide (Top View)**")
                st.caption("Detailed laying pattern: Superimposed Mixed Strip.")
                
                # Retrieve Mine Type from results
                viz_mine_type = res.get('mine_type', 'Anti-Tank')
                
                # Generate Tactical Layout
                layout_data = generate_tactical_layout(frontage, depth, res['density'], viz_mine_type)
                df_mines = layout_data['mines']
                df_markers = layout_data['markers']
                df_annotations = layout_data.get('annotations', pd.DataFrame())
                cl = layout_data['centerline']
                meta = layout_data.get('meta', {})
                
                fig, ax = plt.subplots(figsize=(14, 8)) 
                ax.set_facecolor('#ffffff')
                
                # Plot Centerline
                ax.plot([cl[0][0], cl[1][0]], [cl[0][1], cl[1][1]], color='orange', linestyle='-.', linewidth=2, label='Zero Line', alpha=0.5)
                
                # Plot Mines by Color
                if not df_mines.empty:
                    # Group by Type/Color for legend
                    unique_types = df_mines['type'].unique()
                    for m_type in unique_types:
                         subset = df_mines[df_mines['type'] == m_type]
                         if not subset.empty:
                             color = subset.iloc[0]['color']
                             ax.scatter(subset['x'], subset['y'], c=color, s=40, alpha=0.9, label=m_type, edgecolors='k')
                
                # Plot Markers
                for _, m in df_markers.iterrows():
                    color = m['color']
                    marker = m.get('marker', 'o')
                    ax.plot(m['x'], m['y'], marker=marker, color=color, markersize=10, linestyle='None', label="_nolegend_")
                    ax.text(m['x'], m['y'] - 1, m['id'], fontsize=8, ha='center', color=color, weight='bold')

                # Plot Dimensions / Annotations
                if not df_annotations.empty:
                    for _, ann in df_annotations.iterrows():
                        ax.annotate(
                            "", xy=(ann['x1'], ann['y1']), xytext=(ann['x2'], ann['y2']),
                            arrowprops=dict(arrowstyle='<->', color=ann['color'], lw=1.5)
                        )
                        mid_x = (ann['x1'] + ann['x2']) / 2
                        ax.text(mid_x, ann['y1'] + 0.5, ann['text'], ha='center', fontsize=9, color=ann['color'], weight='bold')

                # Laying Direction Arrow
                if not df_mines.empty:
                    row_y = df_mines.iloc[0]['y']
                    ax.annotate("Laying Direction", xy=(frontage/5, row_y + 6), xytext=(0, row_y + 6),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8), fontsize=10, weight='bold')
                                
                ax.set_title(f"Tactical Layout Plan: Mixed Strip Pattern ({viz_mine_type})", pad=20, fontsize=14)
                ax.set_xlabel("Frontage (m)")
                ax.set_ylabel("Depth (m)")
                ax.set_xlim(-5, frontage + 5)
                ax.set_ylim(0, depth + 10)
                ax.grid(True, linestyle=':', alpha=0.3)
                ax.legend(loc='upper right', frameon=True)
                
                st.pyplot(fig)
                
                st.divider()
                
                # Step 5: Report (Moved to Expander/Button as requested)
                st.markdown("**Step 5: Output Generation**")
                
                if st.button("📄 Generate Analysis Report"):
                    st.success("Report Generated!")
                    st.markdown(f"""
                    ### 📂 Strategic Analysis Report
                    
                    **1. Operational Requirements**
                    - **Objective**: Neutralize enemy forces ({res['target']}) with {res['pk']*100:.1f}% probability.
                    - **Area**: {frontage}m x {depth}m.
                    
                    **2. Pattern Specifications**
                    - **Primary Mine**: {mine_model} ({mine_viz_type})
                    - **Inter-Mine Spacing**: {meta['mine_spacing']:.2f}m
                    - **Strip Spacing**: {meta['row_spacing']}m
                    - **Pattern Type**: {'Mixed (AT + AP)' if meta['mixed'] else 'Uniform'}
                    """)
                
            else:
                st.info("Define parameters in Step 1 to generate the strategic plan.")

    # === TAB 4: Digital Projector ===
    with tab4:
        st.header("Digital Map Projector")
        st.caption("Export Tactical Plans to Google Earth (KML)")
        
        st.info("This module converts your local Minefield Grid (Meters) to Global Coordinates (Lat/Lon) for visualization in Google Earth.")
        
        plan = st.session_state.get('latest_plan')
        
        if plan:
            st.success(f"✅ Plan Loaded: {len(plan['all_mines'])} Mines found.")
            
            # UI for KML Inputs
            col_k1, col_k2 = st.columns(2)
            
            with col_k1:
                st.subheader("1. Reference Point (Landmark)")
                st.markdown(f"**Plan Landmark**: Grid {plan['lm_coords'][0]:.0f}, {plan['lm_coords'][1]:.0f}")
                
                input_mode = st.radio("Input Format", ["Decimal Lat/Lon", "Military Grid (MGRS)"], horizontal=True)
                
                if input_mode == "Decimal Lat/Lon":
                    c_lat, c_lon = st.columns(2)
                    with c_lat:
                        ref_lat = st.number_input("Latitude", value=31.5000, format="%.6f", step=0.0001)
                    with c_lon:
                        ref_lon = st.number_input("Longitude", value=74.3000, format="%.6f", step=0.0001)
                else:
                    # MGRS Input
                    mgrs_str = st.text_input("MGRS String", "42S UB 12345 67890")
                    try:
                        import mgrs
                        m = mgrs.MGRS()
                        lat, lon = m.toLatLon(mgrs_str.replace(" ", ""))
                        st.success(f"✅ Converted: **{lat:.6f}, {lon:.6f}**")
                        ref_lat, ref_lon = lat, lon
                    except ImportError:
                        st.error("⚠️ MGRS Library not installed (Mobile Mode). Please use Lat/Lon.")
                        ref_lat, ref_lon = 0.0, 0.0
                    except Exception as e:
                        st.error(f"Invalid MGRS String: {e}")
                        ref_lat, ref_lon = 0.0, 0.0

            with col_k2:
                st.subheader("2. Visualization")
                
                # Mode Selection
                viz_mode = st.radio(
                    "Select Platform:",
                    ["Interactive Local Map (Fastest)", "Google Earth (Web / KML)"],
                    index=0,
                    help="Local Map opens instantly. Google Earth Web requires internet & drag-and-drop."
                )
                
                st.markdown("---")
                
                # Unified Launch Button
                if st.button("🚀 PROJECT & LAUNCH", type="primary", use_container_width=True):
                    
                    # 1. Prepare Data
                    lm_e, lm_n = plan['lm_coords']
                    survey_points = [{"label": "Landmark (LM)", "e": lm_e, "n": lm_n}]
                    for s_idx, cfg in enumerate(plan['strip_configs']):
                        for pt in cfg['trace']:
                            l, e, n = pt
                            survey_points.append({"label": l, "e": e, "n": n})

                    # 2. Logic based on selection
                    if "Local Map" in viz_mode:
                        # Folium Generation
                        map_path = generate_folium_map(
                            plan['all_mines'], 
                            survey_points,
                            ref_lat, ref_lon,
                            lm_e, lm_n
                        )
                        
                        try:
                            abs_path = os.path.abspath(map_path)
                            webbrowser.open(f'file:///{abs_path}')
                            st.success(f"🚀 **Map Launched!** Check your browser.")
                            st.caption(f"File: `{abs_path}`")
                        except Exception as e:
                            st.error(f"Launch failed. File saved at: {os.path.abspath(map_path)}")
                            
                    else:
                        # Google Earth (KML) Generation
                        kml_content = generate_kml(
                            plan['all_mines'],
                            survey_points,
                            ref_lat, ref_lon,
                            lm_e, lm_n
                        )
                        
                        # Save KML file
                        kml_filename = "minefield_plan.kml"
                        full_path = os.path.abspath(kml_filename)
                        with open(kml_filename, "w") as f:
                            f.write(kml_content)
                            
                        # Web Launch Logic
                        try:
                            # Open Google Earth Web
                            webbrowser.open("https://earth.google.com/web/")
                            st.success(f"🌎 **Google Earth Web Opened!**")
                            st.info(f"📂 **Action Required**: Drag the file `{kml_filename}` from your folder into the browser window.")
                            st.code(full_path, language="text")
                            
                            # Also provide download just in case
                            st.download_button("Download KML", kml_content, file_name="minefield_plan.kml")
                            
                        except Exception as e:
                            st.error(f"Could not open browser: {e}")
                            st.download_button("Download KML File", kml_content, file_name="minefield_plan.kml")
                    
            # Result Section
            if 'kml_str' in st.session_state:
                st.divider()
                st.success("KML Generated Successfully!")
                
                st.download_button(
                    label="⬇️ Download Minefield.kml",
                    data=st.session_state['kml_str'],
                    file_name="minefield_plan.kml",
                    mime="application/vnd.google-earth.kml+xml",
                    type="primary",
                    use_container_width=True
                )
                
                st.info("Instructions: Open Google Earth -> File -> Open -> Select this .kml file.")

        else:
            st.warning("⚠️ No Tactical Plan found.")
            st.markdown("Please go to **Tab 1 (Quick Calculator)** and click **'Generate Tactical Plan'** to create the data first.")

# --- Execution Flow ---

st.set_page_config(page_title="Digital Minefield Record", layout="wide")

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    login_page()
else:
    main_app()


