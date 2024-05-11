import pandas as pd

class read_data:

    def load_data():
       path = 'mobile_recommendation_system_dataset.csv'
       data = pd.read_csv(path)
       return data