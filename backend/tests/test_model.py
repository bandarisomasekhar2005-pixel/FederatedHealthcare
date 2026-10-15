from backend.app.ml.model_service import ModelService
def test_prediction():
    m=ModelService()
    patient={"age":55,"sex":1,"cp":3,"trestbps":130,"chol":240,"fbs":0,"restecg":0,
             "thalach":150,"exang":0,"oldpeak":1.0,"slope":2,"ca":0,"thal":3}
    p,pred,x=m.predict(patient)
    assert 0<=p<=1
    assert pred in (0,1)
    assert x.shape==(1,22)
