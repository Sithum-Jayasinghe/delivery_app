import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def train_delivery_model():
    """
    Trains an enhanced Linear Regression model for delivery time prediction.
    """
    print("📊 Training enhanced delivery time prediction model...")
    
    # Enhanced dataset with more realistic Sri Lankan scenarios
    data = {
        'distance_km': [],
        'traffic_level': [],
        'road_type': [],  # 1=Highway, 2=Main, 3=Local
        'delivery_time_hours': []
    }
    
    # Generate more varied training data
    np.random.seed(42)
    
    for _ in range(50):
        distance = np.random.uniform(5, 200)
        traffic = np.random.randint(1, 5)
        road_type = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        
        # Base time calculation
        base_time = distance * 0.1  # 10 km/hour base speed
        
        # Adjustments
        traffic_factor = 1 + (traffic - 1) * 0.2
        road_factor = 1 + (road_type - 1) * 0.3
        
        total_time = base_time * traffic_factor * road_factor
        
        # Add some randomness
        total_time *= np.random.uniform(0.9, 1.1)
        
        data['distance_km'].append(round(distance, 1))
        data['traffic_level'].append(traffic)
        data['road_type'].append(road_type)
        data['delivery_time_hours'].append(round(total_time, 1))
    
    df = pd.DataFrame(data)
    df.to_csv('delivery_data.csv', index=False)
    print(f"✅ Generated enhanced dataset with {len(df)} samples")
    print(df.head())
    
    # Prepare features (X) and target (y)
    X = df[['distance_km', 'traffic_level', 'road_type']]
    y = df['delivery_time_hours']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\n📈 Model Performance:")
    print(f"   Training R² score: {train_score:.3f}")
    print(f"   Testing R² score: {test_score:.3f}")
    
    # Show coefficients
    print(f"\n🔍 Model Coefficients:")
    for feature, coef in zip(['Distance', 'Traffic', 'Road Type'], model.coef_):
        print(f"   {feature}: {coef:.3f}")
    print(f"   Intercept: {model.intercept_:.3f}")
    
    # Save the model
    joblib.dump(model, 'model.pkl')
    print("\n💾 Model saved as 'model.pkl'")
    
    # Sample predictions
    print("\n📦 Sample Predictions:")
    samples = [
        [50, 2, 1],   # 50km, medium traffic, highway
        [120, 4, 2],  # 120km, heavy traffic, main road
        [25, 1, 3]    # 25km, light traffic, local roads
    ]
    
    for sample in samples:
        pred = model.predict([sample])[0]
        print(f"   {sample[0]}km, Traffic {sample[1]}, Road {sample[2]} → {pred:.1f} hours")

if __name__ == "__main__":
    train_delivery_model()