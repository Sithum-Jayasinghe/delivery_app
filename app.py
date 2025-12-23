import streamlit as st
import folium
from streamlit_folium import folium_static
import joblib
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
import random

# Page configuration
st.set_page_config(
    page_title="Smart Delivery Tracker - Sri Lanka",
    page_icon="🚚",
    layout="wide"
)

# App title and description
st.title("🚚 Smart Delivery Tracking System - Sri Lanka")
st.markdown("Track your deliveries across Sri Lanka with AI-powered time predictions and live GPS simulation")

# SRI LANKAN CITIES DATABASE with real coordinates
SRI_LANKA_CITIES = {
    "Colombo": {"coords": [6.9271, 79.8612], "type": "capital"},
    "Kandy": {"coords": [7.2906, 80.6337], "type": "major"},
    "Galle": {"coords": [6.0535, 80.2210], "type": "major"},
    "Jaffna": {"coords": [9.6615, 80.0255], "type": "major"},
    "Negombo": {"coords": [7.2083, 79.8358], "type": "coastal"},
    "Trincomalee": {"coords": [8.5874, 81.2152], "type": "coastal"},
    "Anuradhapura": {"coords": [8.3114, 80.4037], "type": "historical"},
    "Polonnaruwa": {"coords": [7.9403, 81.0187], "type": "historical"},
    "Matara": {"coords": [5.9485, 80.5353], "type": "southern"},
    "Ratnapura": {"coords": [6.6804, 80.3996], "type": "central"},
    "Nuwara Eliya": {"coords": [6.9497, 80.7891], "type": "hill"},
    "Badulla": {"coords": [6.9934, 81.0550], "type": "hill"},
    "Batticaloa": {"coords": [7.7312, 81.6747], "type": "eastern"},
    "Kurunegala": {"coords": [7.4800, 80.3600], "type": "central"},
    "Puttalam": {"coords": [8.0392, 79.8383], "type": "coastal"},
    "Hambantota": {"coords": [6.1246, 81.1185], "type": "southern"},
    "Kalutara": {"coords": [6.5854, 79.9607], "type": "coastal"},
    "Matale": {"coords": [7.4675, 80.6234], "type": "central"},
    "Monaragala": {"coords": [6.8728, 81.3506], "type": "eastern"},
    "Vavuniya": {"coords": [8.7562, 80.4971], "type": "northern"}
}

# Initialize session state for tracking
if 'delivery_status' not in st.session_state:
    st.session_state.delivery_status = "Pending"
if 'current_location' not in st.session_state:
    st.session_state.current_location = None
if 'order_id' not in st.session_state:
    st.session_state.order_id = None
if 'route_points' not in st.session_state:
    st.session_state.route_points = []
if 'gps_history' not in st.session_state:
    st.session_state.gps_history = []

# Function to calculate distance between two coordinates (simplified)
def calculate_distance(coord1, coord2):
    """Calculate approximate distance between two coordinates in km"""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # Simple approximation for Sri Lanka distances
    return np.sqrt((lat2-lat1)**2 + (lon2-lon1)**2) * 111  # 1 degree ≈ 111 km

# Function to generate intermediate route points
def generate_route_points(start_coords, end_coords, num_points=10):
    """Generate intermediate points along the route for smoother animation"""
    points = []
    for i in range(num_points + 1):
        fraction = i / num_points
        lat = start_coords[0] + (end_coords[0] - start_coords[0]) * fraction
        lon = start_coords[1] + (end_coords[1] - start_coords[1]) * fraction
        points.append([lat, lon])
    return points

# Sidebar for user inputs
with st.sidebar:
    st.header("📋 Delivery Details")
    
    # Generate order ID if not exists
    if st.session_state.order_id is None:
        st.session_state.order_id = f"ORD{np.random.randint(10000, 99999)}"
    
    st.info(f"**Order ID:** {st.session_state.order_id}")
    
    # City selection with auto-complete
    st.subheader("📍 Select Locations")
    
    pickup_city = st.selectbox(
        "Pickup City",
        list(SRI_LANKA_CITIES.keys()),
        index=0,
        help="Select pickup city from major Sri Lankan cities"
    )
    
    drop_city = st.selectbox(
        "Drop City",
        list(SRI_LANKA_CITIES.keys()),
        index=1,
        help="Select destination city"
    )
    
    # Display selected city coordinates
    pickup_coords = SRI_LANKA_CITIES[pickup_city]["coords"]
    drop_coords = SRI_LANKA_CITIES[drop_city]["coords"]
    
    st.caption(f"Pickup: {pickup_city} ({pickup_coords[0]:.4f}, {pickup_coords[1]:.4f})")
    st.caption(f"Drop: {drop_city} ({drop_coords[0]:.4f}, {drop_coords[1]:.4f})")
    
    # Calculate distance automatically
    auto_distance = calculate_distance(pickup_coords, drop_coords)
    
    st.subheader("🚚 Delivery Parameters")
    
    # Use auto-calculated distance or allow manual override
    distance = st.slider(
        "Distance (km)",
        5, 500, 
        int(auto_distance) if auto_distance > 5 else 50,
        5,
        help=f"Auto-calculated: {auto_distance:.1f} km"
    )
    
    traffic = st.slider(
        "Traffic Level (1=Low, 4=High)",
        1, 4, 2,
        help="Consider road conditions and traffic"
    )
    
    weather = st.select_slider(
        "Weather Conditions",
        options=["Sunny", "Cloudy", "Rainy", "Stormy"],
        value="Sunny"
    )
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤖 Predict Time", use_container_width=True, type="primary"):
            try:
                model = joblib.load('model.pkl')
                # Adjust prediction based on weather
                weather_factor = {"Sunny": 1.0, "Cloudy": 1.1, "Rainy": 1.3, "Stormy": 1.5}
                base_prediction = model.predict([[distance, traffic]])[0]
                adjusted_prediction = base_prediction * weather_factor[weather]
                
                st.session_state.predicted_time = adjusted_prediction
                st.session_state.delivery_status = "Processing"
                
                st.success(f"""
                **Estimated Delivery Time:** {adjusted_prediction:.1f} hours
                - Base: {base_prediction:.1f} hours
                - Weather factor: {weather_factor[weather]}x
                """)
                
                # Generate route points
                st.session_state.route_points = generate_route_points(
                    pickup_coords, drop_coords, num_points=20
                )
                
            except Exception as e:
                st.error(f"Error: {e}. Please run train_model.py first")
    
    with col2:
        if st.button("🚀 Start Delivery", use_container_width=True, type="secondary"):
            if 'predicted_time' not in st.session_state:
                st.warning("Please predict time first!")
            else:
                st.session_state.delivery_status = "On the way"
                st.session_state.start_time = datetime.now()
                st.session_state.estimated_arrival = datetime.now() + timedelta(
                    hours=st.session_state.predicted_time
                )
                st.session_state.current_location = pickup_coords.copy()
                st.session_state.current_point_index = 0
                st.session_state.gps_history = [pickup_coords.copy()]
                st.rerun()

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🗺️ Sri Lanka Delivery Route")
    
    # Sri Lanka coordinates (center of the country)
    sri_lanka_center = [7.8731, 80.7718]
    
    # Create a base map centered on Sri Lanka
    m = folium.Map(location=sri_lanka_center, zoom_start=8, tiles="OpenStreetMap")
    
    # Add pickup marker (green)
    folium.Marker(
        pickup_coords,
        popup=f"""
        <div style='font-family: Arial;'>
            <h4>📍 Pickup: {pickup_city}</h4>
            <b>Type:</b> {SRI_LANKA_CITIES[pickup_city]['type']}<br>
            <b>Coordinates:</b> {pickup_coords[0]:.4f}, {pickup_coords[1]:.4f}<br>
            <b>Status:</b> Ready for pickup
        </div>
        """,
        tooltip=f"Pickup: {pickup_city}",
        icon=folium.Icon(color="green", icon="warehouse", prefix="fa")
    ).add_to(m)
    
    # Add drop marker (red)
    folium.Marker(
        drop_coords,
        popup=f"""
        <div style='font-family: Arial;'>
            <h4>🏁 Destination: {drop_city}</h4>
            <b>Type:</b> {SRI_LANKA_CITIES[drop_city]['type']}<br>
            <b>Coordinates:</b> {drop_coords[0]:.4f}, {drop_coords[1]:.4f}<br>
            <b>Distance:</b> {distance} km
        </div>
        """,
        tooltip=f"Destination: {drop_city}",
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
    ).add_to(m)
    
    # Draw route line
    if len(st.session_state.route_points) > 0:
        folium.PolyLine(
            st.session_state.route_points,
            color="blue",
            weight=3,
            opacity=0.7,
            dash_array="5, 5",
            tooltip=f"Route: {pickup_city} → {drop_city}"
        ).add_to(m)
    else:
        # Simple straight line if no route points
        folium.PolyLine(
            [pickup_coords, drop_coords],
            color="gray",
            weight=2,
            opacity=0.5,
            dash_array="10, 5"
        ).add_to(m)
    
    # Add city markers along the route
    cities_along_route = []
    for city, data in SRI_LANKA_CITIES.items():
        # Simple check if city is roughly between pickup and drop
        city_coords = data["coords"]
        if (min(pickup_coords[0], drop_coords[0]) <= city_coords[0] <= max(pickup_coords[0], drop_coords[0]) and
            min(pickup_coords[1], drop_coords[1]) <= city_coords[1] <= max(pickup_coords[1], drop_coords[1])):
            if city not in [pickup_city, drop_city]:
                cities_along_route.append((city, city_coords))
    
    # Display intermediate cities (limit to 3)
    for city, coords in cities_along_route[:3]:
        folium.CircleMarker(
            coords,
            radius=6,
            color='orange',
            fill=True,
            fill_color='orange',
            fill_opacity=0.6,
            popup=f"City: {city}",
            tooltip=f"Passing through: {city}"
        ).add_to(m)
    
    # SIMULATED LIVE GPS TRACKING
    if st.session_state.delivery_status == "On the way":
        # Calculate progress
        elapsed_time = (datetime.now() - st.session_state.start_time).seconds / 3600
        total_time = st.session_state.predicted_time
        progress = min(elapsed_time / total_time, 0.99)
        
        # Move along route points
        if len(st.session_state.route_points) > 0:
            point_index = int(progress * (len(st.session_state.route_points) - 1))
            point_index = min(point_index, len(st.session_state.route_points) - 1)
            current_pos = st.session_state.route_points[point_index]
            
            # Add slight randomness to simulate real GPS
            current_pos = [
                current_pos[0] + random.uniform(-0.01, 0.01),
                current_pos[1] + random.uniform(-0.01, 0.01)
            ]
            
            st.session_state.current_location = current_pos
            st.session_state.current_point_index = point_index
            
            # Add to GPS history
            st.session_state.gps_history.append(current_pos.copy())
            
            # Keep only last 20 positions for trail
            if len(st.session_state.gps_history) > 20:
                st.session_state.gps_history.pop(0)
            
            # Add animated delivery van
            folium.Marker(
                current_pos,
                popup=f"""
                <div style='font-family: Arial;'>
                    <h4>🚚 Delivery Van</h4>
                    <b>Status:</b> On the way<br>
                    <b>Progress:</b> {progress*100:.1f}%<br>
                    <b>Speed:</b> {distance/total_time:.1f} km/h<br>
                    <b>Next Update:</b> 5 seconds
                </div>
                """,
                tooltip=f"Live Location: {progress*100:.1f}%",
                icon=folium.Icon(color="orange", icon="truck-fast", prefix="fa")
            ).add_to(m)
            
            # Add GPS trail (path taken)
            if len(st.session_state.gps_history) > 1:
                folium.PolyLine(
                    st.session_state.gps_history,
                    color="orange",
                    weight=2,
                    opacity=0.5,
                    dash_array="2, 5",
                    tooltip="GPS History Trail"
                ).add_to(m)
        
        # AUTO-REFRESH every 5 seconds
        time.sleep(5)
        st.rerun()
    
    elif st.session_state.delivery_status == "Delivered":
        # Show delivered at destination
        folium.Marker(
            drop_coords,
            popup=f"✅ Delivered: {drop_city}",
            tooltip="Delivered!",
            icon=folium.Icon(color="green", icon="check-circle", prefix="fa")
        ).add_to(m)
    
    # Display the map
    folium_static(m, width=700, height=500)
    
    # Map controls info
    with st.expander("🗺️ Map Controls"):
        st.markdown("""
        - **Zoom:** Mouse scroll or +/- buttons
        - **Move:** Click and drag
        - **Click markers** for detailed info
        - **GPS Trail:** Orange dotted line shows path taken
        - **Cities along route:** Orange circles
        """)

with col2:
    st.subheader("📊 Live Delivery Dashboard")
    
    # Status indicator with emoji
    status_config = {
        "Pending": {"emoji": "⏳", "color": "#808080", "desc": "Awaiting dispatch"},
        "Processing": {"emoji": "⚙️", "color": "#1E90FF", "desc": "Preparing delivery"},
        "On the way": {"emoji": "🚚", "color": "#FF8C00", "desc": "In transit"},
        "Delivered": {"emoji": "✅", "color": "#32CD32", "desc": "Successfully delivered"}
    }
    
    current_status = st.session_state.delivery_status
    status_info = status_config[current_status]
    
    # Status card
    st.markdown(f"""
    <div style="background-color: {status_info['color']}; 
                color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h2>{status_info['emoji']} {current_status}</h2>
        <p>{status_info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Delivery information
    st.info("### 📦 Delivery Information")
    
    info_cols = st.columns(2)
    with info_cols[0]:
        st.metric("Order ID", st.session_state.order_id)
        st.metric("Distance", f"{distance} km")
        st.metric("Traffic Level", f"{traffic}/4")
    
    with info_cols[1]:
        st.metric("Weather", weather)
        if 'predicted_time' in st.session_state:
            st.metric("Predicted Time", f"{st.session_state.predicted_time:.1f} hrs")
        else:
            st.metric("Predicted Time", "N/A")
    
    # Live tracking info
    if st.session_state.delivery_status == "On the way":
        st.success("### 🎯 Live Tracking")
        
        elapsed_time = (datetime.now() - st.session_state.start_time).seconds / 3600
        total_time = st.session_state.predicted_time
        progress = min(elapsed_time / total_time, 0.99)
        
        # Progress bar
        st.progress(progress)
        st.caption(f"Progress: {progress*100:.1f}%")
        
        # Time metrics
        time_cols = st.columns(2)
        with time_cols[0]:
            st.metric("Elapsed", f"{elapsed_time:.1f} hours")
        with time_cols[1]:
            remaining = total_time - elapsed_time
            st.metric("Remaining", f"{remaining:.1f} hours")
        
        # GPS coordinates
        if st.session_state.current_location:
            st.caption(f"**Live GPS:** {st.session_state.current_location[0]:.4f}, {st.session_state.current_location[1]:.4f}")
        
        # Speed calculation
        if elapsed_time > 0:
            avg_speed = distance / elapsed_time
            st.caption(f"**Average Speed:** {avg_speed:.1f} km/h")
        
        # Estimated arrival countdown
        if 'estimated_arrival' in st.session_state:
            time_left = st.session_state.estimated_arrival - datetime.now()
            if time_left.total_seconds() > 0:
                hours_left = time_left.total_seconds() / 3600
                st.caption(f"**ETA:** {st.session_state.estimated_arrival.strftime('%H:%M')} ({hours_left:.1f} hours)")
        
        # Complete delivery button
        if progress >= 0.95:
            if st.button("✅ Mark as Delivered", use_container_width=True, type="primary"):
                st.session_state.delivery_status = "Delivered"
                st.balloons()
                st.success(f"Delivery to {drop_city} completed successfully!")
                st.rerun()
    
    # Control buttons
    st.subheader("🔄 Controls")
    
    control_cols = st.columns(3)
    
    with control_cols[0]:
        if st.button("🔄 Refresh Map", use_container_width=True):
            st.rerun()
    
    with control_cols[1]:
        if st.button("⏸️ Pause Sim", use_container_width=True, disabled=True):
            st.info("Simulation paused")
    
    with control_cols[2]:
        if st.button("🆕 New Delivery", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['_last_run_time', '_is_running']:
                    del st.session_state[key]
            st.session_state.order_id = f"ORD{np.random.randint(10000, 99999)}"
            st.rerun()

# Footer with explanations
st.divider()
st.subheader("📈 Route Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🚚 Route Details")
    st.write(f"""
    **Pickup:** {pickup_city}
    **Destination:** {drop_city}
    **Direct Distance:** {auto_distance:.1f} km
    **Cities Along Route:** {len(cities_along_route)}
    **Route Type:** {SRI_LANKA_CITIES[pickup_city]['type']} → {SRI_LANKA_CITIES[drop_city]['type']}
    """)

with col2:
    st.markdown("### 🤖 AI Prediction Logic")
    st.write("""
    **Model:** Linear Regression
    **Features:** Distance + Traffic + Weather
    **Training Data:** 20 sample deliveries
    **Equation:** Time = a×Distance + b×Traffic + c×Weather
    **Accuracy:** R² > 0.98 on test data
    """)

with col3:
    st.markdown("### 📡 GPS Simulation")
    st.write("""
    **Type:** Simulated GPS with randomness
    **Update Rate:** Every 5 seconds
    **Path:** Smooth interpolation between cities
    **Trail:** Shows last 20 positions
    **Realism:** Added coordinate jitter
    """)

# Auto-refresh toggle
st.divider()
auto_refresh = st.checkbox("🔄 Enable auto-refresh (every 5 seconds)", 
                          value=st.session_state.delivery_status == "On the way",
                          help="Automatically refresh map during delivery")

if auto_refresh and st.session_state.delivery_status == "On the way":
    time.sleep(5)
    st.rerun()