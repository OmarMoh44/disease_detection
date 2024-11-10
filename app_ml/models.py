import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#reading dataset and extract features and target
data = pd.read_csv('./student_version.csv')
x_train = data.iloc[:,:-1]
y_train = data.iloc[:,-1]

#get features with the same data type in list
categorical_features = x_train.select_dtypes(include="object").columns.tolist()
numerical_features = x_train.select_dtypes(exclude="object").columns.tolist()

#get target classes for prediction
classes = y_train.unique()

#extract features that have classes
for feature in numerical_features:
    if len(x_train[feature].unique()) < 8:
        categorical_features.append(feature)
        numerical_features.remove(feature)

#handling outliers
for feature in numerical_features:
    q1 = x_train[feature].quantile(0.25)
    q3 = x_train[feature].quantile(0.75)
    IQR = q3 - q1
    lower_bound = q1 - 0.5 * IQR
    upper_bound = q3 + 0.5 * IQR
    mean_value = x_train[feature].mean()
    x_train[feature] = x_train[feature].astype('float64')
    x_train.loc[(x_train[feature] < lower_bound) | (x_train[feature] > upper_bound), feature] = mean_value
    
#get classes of categorical features for input validation
categorical_unique_values = []
for feature in categorical_features:
    categorical_unique_values.append(data[feature].unique().tolist())
    
#get min and max of numerical features for input validation
min_max = []
for feature in numerical_features:
    min_max.append([x_train[feature].min(),x_train[feature].max()])
    

#scaling for features
preprocessor = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),numerical_features),
        ("cat",OneHotEncoder(),categorical_features)
    ]
)
x_train_scaled = preprocessor.fit_transform(x_train)

#KNN model 
neighbors = 5
KNN_model = KNeighborsClassifier(n_neighbors=neighbors, metric="minkowski", p=2)
KNN_model.fit(x_train_scaled,y_train)

#maching learing algorithm (logistic regression)
logistic_model = LogisticRegression()
logistic_model.fit(x_train_scaled,y_train)

#deep learing (keras)
keras_model = Sequential([
    Dense(16,activation='relu'),
    Dense(8,activation='relu'),
    Dense(1,activation='sigmoid')
])
keras_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
keras_model.fit(x_train_scaled,y_train,epochs=10,batch_size=32)

def predict_target(request):
    try:
        columns = x_train.columns.tolist()
        data = {}
        wanted_model = (request.form)['model']
        target_class = None
        for col in columns:
            x = (request.form)[col]
            if x.isnumeric():
                x = float(x)
            data = {**data, col:[x]}
        df = pd.DataFrame(data)
        df_scaled = preprocessor.transform(df)
        match wanted_model:
            case 'KNN':
                target_class = (KNN_model.predict(df_scaled))
            case 'Logistic':
                target_class = logistic_model.predict(df_scaled)
            case 'Keras':
                target_class = keras_model.predict(df_scaled)
        return {'target':target_class[0]}
    except Exception as e:
        return {'error':str(e)}