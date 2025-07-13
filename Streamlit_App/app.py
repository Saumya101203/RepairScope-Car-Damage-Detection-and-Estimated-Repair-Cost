# app.py
import streamlit as st
from model_helper import predict
import matplotlib.pyplot as plt

st.title("\U0001F699 RepairScope: Instant Vehicle Damage & Cost Check")

# Car brand and model selector
car_brands = ["Maruti Suzuki", "Hyundai", "Toyota", "Tata", "Mahindra", "Kia", "MG"]
car_models = {
    "Maruti Suzuki": ["Swift", "Baleno", "WagonR", "Dzire"],
    "Hyundai": ["i20", "Creta", "Venue"],
    "Toyota": ["Innova", "Fortuner", "Glanza"],
    "Tata": ["Punch", "Nexon", "Harrier"],
    "Mahindra": ["XUV300", "XUV700", "Scorpio N"],
    "Kia": ["Seltos", "Sonet"],
    "MG": ["Hector", "Astor"]
}

brand = st.selectbox("Select Car Brand", car_brands)
model = st.selectbox("Select Car Model", car_models[brand])

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png", "webp"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded File", use_container_width=True)

    if st.button("Predict & Estimate Cost"):
        prediction, probs, class_names = predict(image_path)
        st.info(f"Predicted Class: {prediction}")

        # Plot confidence
        st.subheader("Prediction Confidence")
        fig, ax = plt.subplots()
        ax.barh(class_names, probs)
        ax.invert_yaxis()
        ax.set_xlabel("Confidence")
        st.pyplot(fig)

        # Cost estimation
        from model_helper import get_cost_range
        low, high = get_cost_range(brand, prediction)
        if low == 0 and high == 0:
            st.success("No visible damage. No repair required.")
        else:
            st.success(f"Estimated Repair Cost for {brand} {model}: ₹{low:,} – ₹{high:,} (approx.)")