import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("Binary Classification for Mushrooms")
    st.sidebar.title("Controls")
    st.markdown("## Can you eat this fun-gi? 🍄 ➡️ 🍳")

    @st.cache(persist=True)
    def load_data():
        df = pd.read_csv(r"C:\Users\Mitsy\Projects\fuzzy-potato\mushrooms.csv")
        df = df.rename(columns={'class':'type'})
        label = LabelEncoder()
        for col in df.columns:
            df[col] = label.fit_transform(df[col])
        return df


    #target column is class not type
    @st.cache(persist=True)
    def split(df):
        y = df.type
        X = df.drop(columns=['type'])
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2, random_state=1)
        return X_tr, X_te,y_tr,y_te

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, X_te,y_te,display_labels=class_names)
            st.pyplot()

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_te,y_te)
            st.pyplot()
        if "Precision-Recall Curve" in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, X_te, y_te)
            st.pyplot()

    df = load_data()
    X_tr, X_te,y_tr,y_te = split(df)
    class_names = ["Edible","Poisonous"]
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("SVM","Logistic Regression","Random Forest"))

    if classifier=="SVM":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key="C")
        kernel = st.sidebar.radio("Kernel",("rbf","linear"),key="kernel")
        gamma = st.sidebar.radio("Gamma",("scale","auto"),key="gamma")

        metric = st.sidebar.multiselect("Metrics to Plot",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Support Vector Machine Results")
            model = SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(X_tr,y_tr)
            accuracy = model.score(X_te,y_te)
            yhat = model.predict(X_te)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_te,yhat,labels=class_names))
            st.write("Recall: ", recall_score(y_te,yhat,labels=class_names).round(2))
            plot_metrics(metric)

    if classifier=="Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01, 10.0, step=0.01,key="C")
        max_iter = st.sidebar.slider("Maximum number of iterations",100,500,key="max_iter")

        metric = st.sidebar.multiselect("Metrics to Plot",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter,)
            model.fit(X_tr,y_tr)
            accuracy = model.score(X_te,y_te)
            yhat = model.predict(X_te)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_te,yhat,labels=class_names))
            st.write("Recall: ", recall_score(y_te,yhat,labels=class_names).round(2))
            plot_metrics(metric)

    if classifier=="Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key="n_estimators")
        max_depth = st.sidebar.number_input("The maximum depth of the tree",1,20,step=1,key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees",("True","False"),key="bootstrap")

        metric = st.sidebar.multiselect("Metrics to Plot",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap, n_jobs=-1)
            model.fit(X_tr,y_tr)
            accuracy = model.score(X_te,y_te)
            yhat = model.predict(X_te)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_te,yhat,labels=class_names))
            st.write("Recall: ", recall_score(y_te,yhat,labels=class_names).round(2))
            plot_metrics(metric)

    if st.sidebar.checkbox("Show Raw Data",False):
        st.subheader("Mushroom Data Set")
        st.write(df)





if __name__ == '__main__':
    main()
