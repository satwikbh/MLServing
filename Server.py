from typing import Union

from fastapi import FastAPI

import MLModel
from RequestBody import RequestBody

app = FastAPI()
iris, clf = MLModel.main()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/predict")
def predict(data: RequestBody):
    test_data = [
        [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    ]
    class_idx = clf.predict(test_data)[0]
    return {"class": iris.target_names[class_idx]}
