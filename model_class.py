from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
from preprocess import preprocessor


# Class that will work as the baseline of out model
class Model_Baseline: 
    def __init__(self, X, y, model_architecture, vectorizer, words_to_remove, class_labels, test_size = 0.2, random_seed = 42) -> None:
        # Define initial parameter
        self.X = X
        self.y = y
        self.model_instance = model_architecture # The specified model
        self.vectorizer = vectorizer # The specified vectorizer
        self.test_size = test_size
        self.random_seed = random_seed

        self.words_to_remove = words_to_remove # Parameter for the preprocessor
        self.class_labels = class_labels # Define the label for the classes

        # Defines the parameters for the visual of the confusion matrix
        self.confusion_matrix_kwargs = dict(
            text_auto=True, 
            title="Confusion Matrix", width=1000, height=800,
            labels=dict(x="Predicted", y="True Label"),
            x=self.class_labels,
            y=self.class_labels,
            color_continuous_scale='Blues'
            )

        # Defining the pipeline with the parameter passed in the class 
        self.pipeline = Pipeline([ 
            ("vect", self.vectorizer(preprocessor=lambda x: preprocessor(x, self.words_to_remove))), # We call our custom parameter
            ("model", self.model_instance)]) # Model
        

        # Finally we split our data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y,test_size=self.test_size, random_state = self.random_seed)

    
    def fit(self): # Fit function to the train data
        self.pipeline.fit(self.X_train,self.y_train)

    def score(self):
        score = self.pipeline.score(self.X_test, self. y_test)
        return score


    def predict(self): # Predict function on test data
        return self.pipeline.predict(self.X_test)
    
    def predict_proba(self): # Predict probabilities function on test data
        return self.pipeline.predict_proba(self.X_test)

    def report(self, y_pred, class_labels): # Function to print classification report and confusion matrix reapplied here

        print(classification_report(self.y_test, y_pred, target_names=class_labels))
        cf_mat = confusion_matrix(self.y_test,y_pred)
        fig = px.imshow(
        cf_mat, 
        **self.confusion_matrix_kwargs
        )
        fig.show()