from preprocess import preprocess_text
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px


class Model: # Defining the model class
    def __init__(self, X_train, y_train, X_test,y_test, model_architecture, vectorizer, class_labels) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_instance = model_architecture
        self.vectorizer = vectorizer
        self.class_labels = class_labels

        self.confusion_matrix_kwargs = dict(
        text_auto=True, 
        title="Confusion Matrix", width=1000, height=800,
        labels=dict(x="Predicted", y="True Label"),
        x=self.class_labels,
        y=self.class_labels,
        color_continuous_scale='Blues'
        )

        self.pipeline = Pipeline([ # Defining the pipeline with the parameter passed in the class 
            ("vect", self.vectorizer(preprocessor=self.preprocess)), # Preprocessor
            ("model", self.model_instance)]) # Model


    def preprocess(self, text): # The preprocess function reapplied here
        text = preprocess_text(text)
        return text
    
    def fit(self): # Fit function to the train data
        self.pipeline.fit(self.X_train,self.y_train)

    def predict(self): # Predict function on test data
        return self.pipeline.predict(self.X_test)

    
    def predict_proba(self): # Predict probabilities function on test data
        return self.pipeline.predict_proba(self.X_test)

    def report(self, y_pred, class_labels): # Function for classification report and confusion matrix reapplied here

        print(classification_report(self.y_test, y_pred, target_names=class_labels))
        cf_mat = confusion_matrix(self.y_test,y_pred)
        fig = px.imshow(
        cf_mat, 
        **self.confusion_matrix_kwargs
        )
        fig.show()