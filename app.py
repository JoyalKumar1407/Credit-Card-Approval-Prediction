from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Applicant Name (only for display)
        name = request.form["name"]

        # Read form values
        gender = int(request.form["gender"])
        car = int(request.form["car"])
        house = int(request.form["house"])
        children = int(request.form["children"])
        income = float(request.form["income"])
        income_type = int(request.form["income_type"])
        education = int(request.form["education"])
        family_status = int(request.form["family_status"])
        housing = int(request.form["housing"])

        age = int(request.form["age"])
        experience = int(request.form["experience"])

        mobile = int(request.form["mobile"])
        work_phone = int(request.form["work_phone"])
        phone = int(request.form["phone"])
        email = int(request.form["email"])
        occupation = int(request.form["occupation"])
        family_members = float(request.form["family"])

        # Convert Age and Experience into dataset format
        days_birth = -(age * 365)
        days_employed = -(experience * 365)

        # Feature order must match training
        features = np.array([[
            gender,
            car,
            house,
            children,
            income,
            income_type,
            education,
            family_status,
            housing,
            days_birth,
            days_employed,
            mobile,
            work_phone,
            phone,
            email,
            occupation,
            family_members
        ]])

        prediction = model.predict(features)[0]

        if prediction == 0:
            result = "✅ Good Customer"
        else:
            result = "❌ Bad Customer"

        return render_template(
            "result.html",
            prediction=result,
            applicant=name
        )

    except Exception as e:
        return f"Error : {e}"


if __name__ == "__main__":
    app.run(debug=True)