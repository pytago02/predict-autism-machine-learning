from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load mô hình đã lưu
lr_model = joblib.load("diabetes_model.pkl")
gb_model = joblib.load("diabetes_model.pkl")
knn_model = joblib.load("diabetes_model.pkl")
rf_model = joblib.load("diabetes_model.pkl")
dt_model = joblib.load("diabetes_model.pkl")
svm_model = joblib.load("diabetes_model.pkl")
nb_model = joblib.load("diabetes_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Lấy dữ liệu từ form trên trang web
    features = [float(x) for x in request.form.values() if x != request.form["model"]]
    final_features = np.array([features])

    # Lấy lựa chọn mô hình từ form
    model_choice = request.form["model"]

    # Dự đoán từ mô hình
    lr_prediction = lr_model.predict(final_features)
    knn_prediction = knn_model.predict(final_features)
    rf_prediction = rf_model.predict(final_features)
    dt_prediction = dt_model.predict(final_features)
    svm_prediction = svm_model.predict(final_features)
    nb_prediction = nb_model.predict(final_features)

    # Hiển thị kết quả
    if model_choice == "Logistic Regression":
        if lr_prediction[0] == 1:
            return render_template("false.html")
        else:
            return render_template("success.html")
    elif model_choice == "KNN":
        if knn_prediction[0] == 1:
            return render_template("false.html")
        else:
            return render_template("success.html")
    elif model_choice == "Random Forest":
        if rf_prediction[0] == 1:
            return render_template("false.html")
        else:
            return render_template("success.html")
    elif model_choice == "Decision Tree":
        if dt_prediction[0] == 1:
            return render_template("false.html")
        else:
            return render_template("success.html")
    elif model_choice == "SVM":
        if svm_prediction[0] == 1:
            return render_template("false.html")
        else:
            return render_template("success.html")
    elif model_choice == "Gaussian Naive Bayes":
        if nb_prediction[0] == 1:
            return render_template("false.html")
        else:
            return render_template("success.html")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
