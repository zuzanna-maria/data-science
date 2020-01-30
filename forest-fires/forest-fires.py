import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

url="http://www.dsi.uminho.pt/~pcortez/forestfires/forestfires.csv"
df = pd.read_csv(url)

def predict_temp():
    x_train, x_test, y_train, y_test = train_test_split(
        df[['wind', 'RH', 'FFMC', "ISI", "DC", 'DMC']],
        df['temp'],
        test_size=0.1,
        random_state=0)
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(x_train, y_train)
    y_test_predictions = regressor.predict(x_test)
    print('R^2 for true vs. predicted test set forest temperature: {:0.2f}'.format(r2_score(y_test, y_test_predictions)))

predict_temp()